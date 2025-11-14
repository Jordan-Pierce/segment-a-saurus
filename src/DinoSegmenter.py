import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.v2 as T 
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LogisticRegression


class DinoSegmenter:
    """
    A class for high-resolution, interactive segmentation using
    DINOv3, AnyUp, and a real-time classifier.
    """
    
    def __init__(self, 
                 dino_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m", 
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
        
        # --- NEW: Initialize state for segmentation ---
        self.hr_features = None # Will store the [C, H, W] pixel features
        self.prompts = []       # Will store user clicks
        self.transform_info = {} # Will store info to map clicks
            
        print("DinoSegmenter initialized successfully.")

    @torch.no_grad()
    def set_image(self, image: np.ndarray, target_res: int = 1024):
        """
        Runs the full DINOv3 + AnyUp pipeline on an image and
        stores the resulting high-resolution feature map.
        
        Args:
            image (np.ndarray): The input image as an RGB numpy array [H, W, 3].
            target_res (int): The resolution to process the image at.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
            
        if target_res % self.patch_size != 0:
            target_res = (target_res // self.patch_size) * self.patch_size
            print(f"Warning: target_res adjusted to {target_res} to be a multiple of {self.patch_size}")

        print(f"Setting image: extracting features at {target_res}x{target_res} resolution...")
        img_pil = Image.fromarray(image).convert("RGB")
        W_orig, H_orig = img_pil.size
        
        # --- 1. Create the manual transform ---
        mean = self.processor.image_mean
        std = self.processor.image_std
        
        transforms = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(size=target_res, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(target_res),
            T.Normalize(mean=mean, std=std),
        ])

        # --- 2. Apply transforms ---
        hr_image_tensor_f32 = transforms(img_pil).to(self.device).unsqueeze(0)
        B, C, H_proc, W_proc = hr_image_tensor_f32.shape
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
        # Squeeze batch dim and permute to [H, W, C] for easier indexing
        self.hr_features = hr_features.squeeze(0).permute(1, 2, 0) 
        self.prompts = []
        
        # Store info needed to map clicks from original image to processed image
        scale_factor = target_res / min(W_orig, H_orig)
        new_W = int(round(W_orig * scale_factor))  
        new_H = int(round(H_orig * scale_factor)) 
        
        self.transform_info = {
            "original_size": (W_orig, H_orig),
            "processed_size": (H_proc, W_proc),
            "resize_scale": scale_factor,
            "crop_top": (new_H - target_res) // 2,
            "crop_left": (new_W - target_res) // 2,
        }
        
        print(f"Feature extraction complete. Stored features: {self.hr_features.shape}")
        
    @torch.no_grad()
    def _calculate_similarity(self, all_features, prompt_features_list) -> torch.Tensor:
        """Helper to compute max similarity against a list of prompt features."""
        n_pixels, c = all_features.shape
        if not prompt_features_list:
            return torch.zeros(n_pixels, device=self.device)
            
        prompt_stack = torch.stack(prompt_features_list)
        
        # Normalize for cosine similarity
        all_norm = F.normalize(all_features, p=2, dim=1)
        prompt_norm = F.normalize(prompt_stack, p=2, dim=1)
        
        # (N_pixels, C) @ (C, N_prompts) -> (N_pixels, N_prompts)
        similarity = torch.matmul(all_norm, prompt_norm.T)
        
        # Take the max similarity for each pixel against all prompt features
        max_similarity, _ = torch.max(similarity, dim=1)
        
        return max_similarity

    def _transform_coords(self, original_coords: tuple) -> tuple | None:
        """Transforms (y, x) from original image to processed feature map."""
        y_click, x_click = original_coords
        
        # 1. Simulate the resize
        y_resized = int(y_click * self.transform_info["resize_scale"])
        x_resized = int(x_click * self.transform_info["resize_scale"])
        
        # 2. Simulate the center crop
        y_proc = y_resized - self.transform_info["crop_top"]
        x_proc = x_resized - self.transform_info["crop_left"]
        
        # 3. Check if click is inside the cropped area
        H_proc, W_proc = self.transform_info["processed_size"]
        if 0 <= y_proc < H_proc and 0 <= x_proc < W_proc:
            return (y_proc, x_proc)
        else:
            return None # Click was outside the processed area

    # --- NEW INTERACTIVE METHODS ---

    def add_prompt(self, original_coords: tuple, is_positive: bool):
        """
        Adds a single point prompt (positive or negative) to the list.
        
        Args:
            original_coords (tuple): (y, x) coordinates from the original image.
            is_positive (bool): True for foreground (label 1), False for background (label 0).
        """
        processed_coords = self._transform_coords(original_coords)
        
        if processed_coords:
            label = 1 if is_positive else 0
            self.prompts.append({"coords": processed_coords, "label": label})
            print(f"Added {'positive' if is_positive else 'negative'} prompt at {processed_coords}.")
        else:
            print("Click was outside the processed image area, prompt ignored.")

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
    def predict_mask(self) -> np.ndarray:
        """
        Predicts a pixel-perfect mask by training a real-time
        classifier on the current prompts.
                               
        Returns:
            np.ndarray: A [H_proc, W_proc] float array (0.0 to 1.0)
                        representing the foreground confidence map.
        """
        if self.hr_features is None:
            raise RuntimeError("An image must be set with set_image() before predicting.")
        
        H_proc, W_proc, C = self.hr_features.shape
        
        if len(self.prompts) == 0:
            print("No prompts provided. Returning empty map.")
            return np.zeros((H_proc, W_proc), dtype=np.float32)
            
        labels = set(p['label'] for p in self.prompts)
        
        if len(labels) == 1:
            # --- MODE 1: Similarity Search (Only one class of prompts) ---
            if 1 in labels:
                print("Only positive prompts found. Running similarity search...")
                
                # 1. Get all positive features
                pos_features = []
                for prompt in self.prompts:
                    if prompt['label'] == 1:
                        y, x = prompt['coords']
                        pos_features.append(self.hr_features[y, x])
                
                all_features = self.hr_features.reshape(-1, C)
                confidence_map = self._calculate_similarity(all_features, pos_features)
                
                min_val, max_val = torch.min(confidence_map), torch.max(confidence_map)
                if min_val == max_val:
                    normalized_map = (confidence_map > 0).float()
                else:
                    normalized_map = (confidence_map - min_val) / (max_val - min_val)
                
                print("Similarity search complete.")
                return normalized_map.reshape(H_proc, W_proc).cpu().numpy()
            
            else:
                # Only negative prompts
                print("Only negative prompts found. Returning empty map.")
                return np.zeros((H_proc, W_proc), dtype=np.float32)

        else:
            # --- MODE 2: Classifier (Positive AND Negative prompts) ---
            print("Training real-time classifier...")
            
            X_train_list = []
            y_train = []
            for prompt in self.prompts:
                y, x = prompt['coords']
                label = prompt['label']
                X_train_list.append(self.hr_features[y, x])
                y_train.append(label)
                
            X_train = torch.stack(X_train_list).cpu().numpy()
            y_train = np.array(y_train)

            clf = LogisticRegression(
                class_weight='balanced', 
                random_state=0, 
                max_iter=1000
            )
            clf.fit(X_train, y_train)

            print("Predicting on full high-res feature map...")
            all_features = self.hr_features.reshape(-1, C).cpu().numpy()
            confidence_map = clf.predict_proba(all_features)[:, 1]
            
            print("Prediction complete.")
            return confidence_map.reshape(H_proc, W_proc).astype(np.float32)
        
    def post_process_mask(self, confidence_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Aligns the processed confidence map with the original image.
        ...
        """
        if self.transform_info is None:
            raise RuntimeError("set_image() must be called before post-processing.")
            
        # 1. Apply threshold
        binary_mask = (confidence_map > threshold)
        
        # 2. Get transform info
        info = self.transform_info
        W_orig, H_orig = info["original_size"]
        scale = info["resize_scale"]
        
        # 3. Calculate the "resized-but-not-cropped" dimensions
        new_H = int(round(H_orig * scale)) # Use round() to match torchvision
        new_W = int(round(W_orig * scale)) # Use round() to match torchvision
        
        # 4. Create a blank canvas with these dimensions
        uncropped_mask = np.zeros((new_H, new_W), dtype=np.uint8)
        
        # 5. Paste the binary_mask into the center
        top, left = info["crop_top"], info["crop_left"]
        
        # --- THIS IS THE FIX ---
        # Instead of getting H_proc/W_proc from info,
        # get the *actual* shape of the mask you are pasting.
        mask_to_paste = (binary_mask * 255).astype(np.uint8)
        H_mask, W_mask = mask_to_paste.shape
        
        # Make sure the paste doesn't go out of bounds
        # (This handles rounding errors)
        H_end = min(top + H_mask, new_H)
        W_end = min(left + W_mask, new_W)
        
        uncropped_mask[top : H_end, left : W_end] = mask_to_paste[0 : H_end - top, 0 : W_end - left]
        # --- END FIX ---
        
        # 6. Resize this correctly aligned mask to the *original* image size
        mask_img = Image.fromarray(uncropped_mask)
        mask_resized = np.array(mask_img.resize((W_orig, H_orig), Image.NEAREST))
        
        # 7. Return as boolean
        return (mask_resized > 0)


# --- Simple command-line test ---
if __name__ == "__main__":
    import os
    
    TEST_IMAGE_PATH = "data/veggies.jpg" 
    PROCESSING_RESOLUTION = 448
    DINOV3_MODEL = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"  # Patch size 16
    
    point = (110, 310)  # y, x coordinate of a hanging tomato
        
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at '{TEST_IMAGE_PATH}'")
    else:
        try:
            # 1. Load image
            print(f"Loading image: {TEST_IMAGE_PATH}")
            image = Image.open(TEST_IMAGE_PATH).convert("RGB")
            image_np = np.array(image)

            # 2. Initialize segmenter
            t0 = time.time()
            segmenter = DinoSegmenter(dino_model_name=DINOV3_MODEL)
            print(f"Model init time: {time.time() - t0:.2f}s")
            
            # 3. Run set_image (the slow step)
            t0 = time.time()
            segmenter.set_image(image_np, target_res=PROCESSING_RESOLUTION)
            print(f"set_image (DINOv3 + AnyUp) time: {time.time() - t0:.2f}s")
            
            # 4. Add dummy prompts
            H, W, _ = image_np.shape
            segmenter.add_prompt(original_coords=point, is_positive=True)  # Hanging tomato
            segmenter.add_prompt(original_coords=(10, 10), is_positive=False)  # Background

            # 5. Run prediction
            t0 = time.time()
            confidence_map = segmenter.predict_mask()
            print(f"predict_mask (train + predict) time: {time.time() - t0:.2f}s")
                        
            # 6. Post-process the mask
            t0 = time.time()
            final_mask = segmenter.post_process_mask(confidence_map, threshold=0.90)
            print(f"post_process_mask time: {time.time() - t0:.2f}s")
            
            # 7. Plot
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            axs[0].plot(point[1], point[0], 'ro', markersize=3)  # Mark the prompt
            axs[0].imshow(image_np)
            axs[0].set_title("Original Image")
            axs[0].axis("off")
            
            axs[1].imshow(confidence_map, cmap='inferno')
            axs[1].set_title(f"Confidence Map ({confidence_map.shape[0]}x{confidence_map.shape[1]})")
            axs[1].axis("off")

            # Create an overlay
            overlay = image_np.copy()
            overlay[final_mask] = [255, 255, 255]  # Highlight mask in white
            
            axs[2].imshow(image_np)
            axs[2].imshow(overlay, alpha=0.5)
            axs[2].set_title("Final Mask Overlay (Aligned)")
            axs[2].axis("off")
            
            plt.suptitle(f"DINOv3 + AnyUp + LogisticRegression Segmentation")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()