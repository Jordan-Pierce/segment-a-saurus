import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.v2 as T 
import matplotlib.pyplot as plt
import time


pretrained_model_names = [
    
    "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
]


class DinoSegmenter:
    """
    A class to encapsulate the DINOv3 + AnyUp feature extraction pipeline
    for high-resolution, pixel-level semantic features.
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
                torch_dtype=torch.float16, # Load DINOv3 in float16
                device_map="auto",
                attn_implementation=attn_impl,
            ).to(self.device).eval()
            
            if self.is_vit:
                self.patch_size = self.model.config.patch_size
            else:
                self.patch_size = 32 # Total stride for ConvNeXt
                
        except Exception as e:
            print(f"Error loading DINOv3 model: {e}")
            raise

        # 2. Load AnyUp Model
        try:
            print("Loading AnyUp model from torch.hub...")
            # AnyUp loads in float32 by default
            self.upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(self.device).eval()
        except Exception as e:
            print(f"Error loading AnyUp model: {e}")
            raise
            
        print("DinoSegmenter initialized successfully.")

    @torch.no_grad()
    def extract_features(self, image: np.ndarray, target_res: int = 1024):
        """
        Runs the full DINOv3 + AnyUp pipeline on an image.
        
        Args:
            image (np.ndarray): The input image as an RGB numpy array [H, W, 3].
            target_res (int): The resolution to process the image at. Must be a
                              multiple of the model's patch size.
                              
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - lr_features: Low-res features (float16) [1, C, H_grid, W_grid]
                - hr_features: High-res features (float32) [1, C, H_proc, W_proc]
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
            
        if target_res % self.patch_size != 0:
            raise ValueError(f"target_res ({target_res}) must be a multiple of patch_size ({self.patch_size}).")

        print(f"Extracting features at {target_res}x{target_res} resolution...")
        img_pil = Image.fromarray(image).convert("RGB")
        
        # --- 1. Create the manual transform ---
        mean = self.processor.image_mean
        std = self.processor.image_std
        
        transforms = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True), # <-- Create as float32
            T.Resize(size=target_res, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(target_res),
            T.Normalize(mean=mean, std=std),
        ])

        # --- 2. Apply transforms ---
        # Create the tensor as float32
        hr_image_tensor_f32 = transforms(img_pil).to(self.device).unsqueeze(0)
        
        B, C, H_proc, W_proc = hr_image_tensor_f32.shape
        h_grid = H_proc // self.patch_size
        w_grid = W_proc // self.patch_size
        n_patches = h_grid * w_grid

        # --- 3. DINOv3 Pass (Get Low-Res Features) ---
        print("Running DINOv3 pass...")
        # Convert to float16 *only* for the DINOv3 model pass
        out = self.model(pixel_values=hr_image_tensor_f32.to(torch.float16))
        hs = out.last_hidden_state # Output is float16
        
        patch_features = hs.squeeze(0)[-n_patches:, :] # [N, C] (float16)
        
        if patch_features.shape[0] != n_patches:
             raise RuntimeError("Feature extraction failed! Mismatch in patch count.")
        
        # Reshape to [B, C, H_grid, W_grid] for AnyUp
        lr_features_f16 = patch_features.permute(1, 0).reshape(1, -1, h_grid, w_grid).contiguous()

        # --- 4. AnyUp Pass (Get High-Res Features) ---
        print("Running AnyUp pass...")
        
        # --- START: THE FIX ---
        # Convert *both* inputs to float32 to match AnyUp's weights
        hr_features = self.upsampler(
            hr_image_tensor_f32,                # Pass the float32 image
            lr_features_f16.to(torch.float32),  # Convert features to float32
            q_chunk_size=256
        )
        # --- END: THE FIX ---
        
        print("Feature extraction complete.")
        # Return lr_features as float16 and hr_features as float32
        return lr_features_f16, hr_features

    @staticmethod
    @torch.no_grad()
    def visualize_features(lr_features: torch.Tensor, hr_features: torch.Tensor):
        """
        Performs a joint PCA on low-res and high-res features
        and returns them as plottable RGB images.
        """
        print("Performing joint PCA for visualization...")
        
        # --- This logic is adapted directly from the AnyUp demo script ---
        #
        
        # Convert both to float32 for PCA
        lr_features = lr_features.to(torch.float32)
        hr_features = hr_features.to(torch.float32)

        # 1. Get shapes
        _, C, h_grid, w_grid = lr_features.shape
        _, _, h_proc, w_proc = hr_features.shape
        n_lr = h_grid * w_grid
        
        # 2. Flatten both feature maps
        lr_flat = lr_features[0].permute(1, 2, 0).reshape(-1, C)
        hr_flat = hr_features[0].permute(1, 2, 0).reshape(-1, C)
        all_feats = torch.cat([lr_flat, hr_flat], dim=0)

        # 3. Compute PCA
        mean = all_feats.mean(dim=0, keepdim=True)
        X = all_feats - mean
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        pcs = Vh[:3].T # Top 3 principal components
        proj_all = X @ pcs

        # 4. Split back and normalize
        proj_lr = proj_all[:n_lr].reshape(h_grid, w_grid, 3)
        proj_hr = proj_all[n_lr:].reshape(h_proc, w_proc, 3)
        
        cmin = proj_all.min(dim=0).values
        cmax = proj_all.max(dim=0).values
        crng = (cmax - cmin).clamp(min=1e-6)

        lr_rgb = ((proj_lr - cmin) / crng).cpu().numpy()
        hr_rgb = ((proj_hr - cmin) / crng).cpu().numpy()
        
        print("PCA visualization complete.")
        return lr_rgb, hr_rgb
    
    
# --- Simple command-line test ---
if __name__ == "__main__":
    import os
    
    # --- CONFIGURATION ---
    # Put a test image in this path
    TEST_IMAGE_PATH = "data/veggies.jpg" 
    
    # Resolution to process at. Must be a multiple of 16 (for vits16) or 32 (for convnext)
    # Use a smaller value (e.g., 448 or 512) for faster testing
    PROCESSING_RESOLUTION = 1024 
    
    # Model to test
    # DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m" # Patch size 16
    DINOV3_MODEL = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m" # Patch size 32
    # ---------------------

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at '{TEST_IMAGE_PATH}'")
        print("Please create this file to run the test.")
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
            
            # Adjust resolution if needed for ConvNeXt
            if PROCESSING_RESOLUTION % segmenter.patch_size != 0:
                old_res = PROCESSING_RESOLUTION
                PROCESSING_RESOLUTION = (old_res // segmenter.patch_size) * segmenter.patch_size
                print(f"Warning: Adjusted resolution from {old_res} to {PROCESSING_RESOLUTION} "
                      f"to be a multiple of patch size {segmenter.patch_size}")

            # 3. Run pipeline
            t0 = time.time()
            lr_features, hr_features = segmenter.extract_features(image_np, target_res=PROCESSING_RESOLUTION)
            print(f"Full feature extraction time: {time.time() - t0:.2f}s")
            
            print(f"LR features shape: {lr_features.shape}")
            print(f"HR features shape: {hr_features.shape}")

            # 4. Visualize
            t0 = time.time()
            lr_viz, hr_viz = DinoSegmenter.visualize_features(lr_features, hr_features)
            print(f"PCA visualization time: {time.time() - t0:.2f}s")

            # 5. Plot
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(lr_viz)
            axs[0].set_title(f"Low-Res DINOv3 Features ({lr_viz.shape[0]}x{lr_viz.shape[1]})")
            axs[0].axis("off")
            
            axs[1].imshow(hr_viz)
            axs[1].set_title(f"High-Res AnyUp Features ({hr_viz.shape[0]}x{hr_viz.shape[1]})")
            axs[1].axis("off")
            
            plt.suptitle(f"DINOv3 + AnyUp PCA Visualization\nModel: {DINOV3_MODEL.split('/')[-1]}")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()