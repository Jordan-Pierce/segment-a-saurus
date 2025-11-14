import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.v2 as T 
import matplotlib.pyplot as plt
import time


class DinoSegmenter:
    """
    A class for high-resolution, interactive segmentation using
    DINOv3, AnyUp, and real-time similarity search.
    
    Processes the full image by resizing and padding, not cropping.
    """
    
    def __init__(self, 
                 dino_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m", 
                 device: str = None):
        """
        Initializes the segmenter by loading both DINOv3 and AnyUp models.
        
        Args:
            dino_model_name (str): The DINOv3 model to use as the feature extractor.
            device (str, optional): The device to run models on ('cuda', 'cpu').
        """
        print(f"Initializing DinoSegmenter with {dino_model_name}...")
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.is_vit = "vit" in dino_model_name.lower()
        attn_impl = "sdpa" if self.is_vit else "eager"

        # 1. Load DINOv3 Model
        try:
            print(f"Loading {dino_model_name}...")
            self.processor = AutoImageProcessor.from_pretrained(dino_model_name)
            self.model = AutoModel.from_pretrained(
                dino_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation=attn_impl,
            ).to(self.device).eval()
            
            if self.is_vit:
                self.patch_size = self.model.config.patch_size
            else:
                self.patch_size = 32
                
        except Exception as e:
            print(f"Error loading DINOv3 model: {e}")
            raise

        # 2. Load AnyUp Model
        try:
            print("Loading AnyUp model from torch.hub...")
            self.upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(self.device).eval()
        except Exception as e:
            print(f"Error loading AnyUp model: {e}")
            raise
        
        # --- Initialize state for segmentation ---
        self.hr_features = None # Will store the [C, H, W] pixel features
        self.prompts = []       # Will store user clicks
        self.transform_info = {} # Will store info to map clicks
            
        print("DinoSegmenter initialized successfully.")

    @torch.no_grad()
    def set_image(self, image: np.ndarray, target_res: int = 1024):
        """
        Runs the full DINOv3 + AnyUp pipeline on an image and
        stores the resulting high-resolution feature map.
        
        Image is resized (smallest edge to target_res) and then
        padded to be divisible by the patch size.
        
        Args:
            image (np.ndarray): The input image as an RGB numpy array [H, W, 3].
            target_res (int): The resolution for the smallest edge.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")

        print(f"Setting image: resizing smallest edge to {target_res} and padding...")
        img_pil = Image.fromarray(image).convert("RGB")
        W_orig, H_orig = img_pil.size
        
        # --- 1. Create transforms ---
        mean = self.processor.image_mean
        std = self.processor.image_std
        
        # A. Transforms *before* padding
        transforms_pre = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(size=target_res, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        ])
        
        # B. Apply pre-transforms to find unpadded size
        img_tensor_unpadded = transforms_pre(img_pil)
        _, H_proc_unpadded, W_proc_unpadded = img_tensor_unpadded.shape
        
        # C. Calculate padding needed to be multiple of patch_size
        H_proc = (H_proc_unpadded + self.patch_size - 1) // self.patch_size * self.patch_size
        W_proc = (W_proc_unpadded + self.patch_size - 1) // self.patch_size * self.patch_size
        
        padding_right = W_proc - W_proc_unpadded
        padding_bottom = H_proc - H_proc_unpadded
        
        # D. Create final transforms (pad and normalize)
        # We pad with 0, which becomes (0-mean)/std after normalization
        transforms_post = T.Compose([
            T.Pad(padding=(0, 0, padding_right, padding_bottom), fill=0),
            T.Normalize(mean=mean, std=std),
        ])

        # --- 2. Apply transforms ---
        hr_image_tensor_f32 = transforms_post(img_tensor_unpadded).to(self.device).unsqueeze(0)
        B, C, H_proc_check, W_proc_check = hr_image_tensor_f32.shape
        assert H_proc == H_proc_check and W_proc == W_proc_check

        h_grid = H_proc // self.patch_size
        w_grid = W_proc // self.patch_size
        n_patches = h_grid * w_grid

        # --- 3. DINOv3 Pass (Get Low-Res Features) ---
        print("Running DINOv3 pass...")
        out = self.model(pixel_values=hr_image_tensor_f32.to(torch.float16))
        hs = out.last_hidden_state 
        patch_features = hs.squeeze(0)[-n_patches:, :] 
        lr_features_f16 = patch_features.permute(1, 0).reshape(1, -1, h_grid, w_grid).contiguous()

        # --- 4. AnyUp Pass (Get High-Res Features) ---
        print("Running AnyUp pass...")
        hr_features = self.upsampler(
            hr_image_tensor_f32,
            lr_features_f16.to(torch.float32),
            q_chunk_size=256
        )
        
        # --- 5. Store state ---
        self.hr_features = hr_features.squeeze(0).permute(1, 2, 0) 
        self.prompts = []
        
        # Store info needed to map clicks and un-pad the mask
        scale_factor = target_res / min(W_orig, H_orig)
        
        self.transform_info = {
            "original_size": (W_orig, H_orig),
            "processed_size": (H_proc, W_proc), # Padded size
            "unpadded_size": (H_proc_unpadded, W_proc_unpadded), # Resized-only size
            "resize_scale": scale_factor,
        }
        
        print(f"Feature extraction complete. Stored features: {self.hr_features.shape}")
        
    @torch.no_grad()
    def _calculate_similarity(self, all_features, prompt_features_list) -> torch.Tensor:
        """Helper to compute max similarity against a list of prompt features."""
        n_pixels, c = all_features.shape
        if not prompt_features_list:
            return torch.zeros(n_pixels, device=self.device)
            
        prompt_stack = torch.stack(prompt_features_list)
        
        all_norm = F.normalize(all_features, p=2, dim=1)
        prompt_norm = F.normalize(prompt_stack, p=2, dim=1)
        
        similarity = torch.matmul(all_norm, prompt_norm.T)
        max_similarity, _ = torch.max(similarity, dim=1)
        
        return max_similarity

    def _transform_coords(self, original_coords: tuple) -> tuple | None:
        """Transforms (y, x) from original image to processed feature map."""
        y_click, x_click = original_coords
        info = self.transform_info
        
        # 1. Simulate the resize
        y_proc = int(y_click * info["resize_scale"])
        x_proc = int(x_click * info["resize_scale"])
        
        # 2. Check if click is inside the un-padded area
        H_unpadded, W_unpadded = info["unpadded_size"]
        if 0 <= y_proc < H_unpadded and 0 <= x_proc < W_unpadded:
            return (y_proc, x_proc)
        else:
            return None # Click was outside the processed area

    # --- INTERACTIVE METHODS ---

    def add_prompt(self, original_coords: tuple, is_positive: bool):
        """Adds a single point prompt (positive or negative) to the list."""
        processed_coords = self._transform_coords(original_coords)
        
        if processed_coords:
            label = 1 if is_positive else 0
            self.prompts.append({"coords": processed_coords, "label": label})
            # Suppress print statements for drawing
            # print(f"Added {'positive' if is_positive else 'negative'} prompt at {processed_coords}.")
        else:
            # print("Click was outside the processed image area, prompt ignored.")
            pass

    def clear_prompts(self):
        """Removes all active prompts."""
        self.prompts = []
        print("All prompts cleared.")

    def undo_prompt(self):
        """Removes the most recently added prompt."""
        if self.prompts:
            removed = self.prompts.pop()
            print(f"Removed last prompt at {removed['coords']}.")
        else:
            print("No prompts to undo.")

    @torch.no_grad()
    def predict_scores(self) -> (np.ndarray, np.ndarray):
        """
        Calculates the raw cosine similarity maps for positive and negative
        prompts.
        """
        if self.hr_features is None:
            raise RuntimeError("An image must be set with set_image() before predicting.")

        H_proc, W_proc, C = self.hr_features.shape
        all_features = self.hr_features.reshape(-1, C)

        if len(self.prompts) == 0:
            print("No prompts provided. Returning empty maps.")
            return (
                np.zeros((H_proc, W_proc), dtype=np.float32),
                np.zeros((H_proc, W_proc), dtype=np.float32)
            )

        pos_features = []
        neg_features = []
        for prompt in self.prompts:
            y, x = prompt['coords']
            if prompt['label'] == 1:
                pos_features.append(self.hr_features[y, x])
            else:
                neg_features.append(self.hr_features[y, x])

        if pos_features:
            pos_map_flat = self._calculate_similarity(all_features, pos_features)
            pos_map_flat = (pos_map_flat + 1.0) / 2.0
        else:
            pos_map_flat = torch.zeros(H_proc * W_proc, device=self.device)

        if neg_features:
            neg_map_flat = self._calculate_similarity(all_features, neg_features)
            neg_map_flat = (neg_map_flat + 1.0) / 2.0
        else:
            neg_map_flat = torch.zeros(H_proc * W_proc, device=self.device)

        pos_map = pos_map_flat.reshape(H_proc, W_proc).cpu().numpy()
        neg_map = neg_map_flat.reshape(H_proc, W_proc).cpu().numpy()

        return pos_map, neg_map
        
    def post_process_mask(self, confidence_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Aligns the processed confidence map with the original image.
        
        Takes the padded confidence_map, applies a threshold,
        un-pads it, and then resizes it back to the original image's
        dimensions.
        """
        if self.transform_info is None:
            raise RuntimeError("set_image() must be called before post-processing.")
            
        # 1. Apply threshold
        binary_mask = (confidence_map > threshold)
        
        # 2. Get transform info
        info = self.transform_info
        W_orig, H_orig = info["original_size"]
        H_unpadded, W_unpadded = info["unpadded_size"]
        
        # 3. Un-pad the mask by cropping it
        uncropped_mask = binary_mask[0:H_unpadded, 0:W_unpadded]
        
        # 4. Resize this correctly-sized mask to the *original* image size
        mask_img = Image.fromarray((uncropped_mask * 255).astype(np.uint8))
        mask_resized = np.array(mask_img.resize((W_orig, H_orig), Image.NEAREST))
        
        # 5. Return as boolean
        return (mask_resized > 0)


# (Main test function is disabled)
if __name__ == "__main__":
    print("This file is intended to be used as a module.")