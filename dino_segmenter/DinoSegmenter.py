import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.v2 as T 


class DinoSegmenter:
    """
    An API for visualizing DINOv3 feature similarity,
    using the exact logic from the provided Gradio app
    AND processing the image at a user-defined max resolution.
    """
    
    def __init__(self, device: str = None, max_resolution: int = 1024):
        """
        Initializes the segmenter by loading the DINOv3 model.
        
        Args:
            device (str, optional): The device to run models on ('cuda', 'cpu').
            max_resolution (int): The maximum dimension (W or H) to process.
                                  Images larger than this will be scaled down
                                  before feature extraction to save memory/time.
        """
        print("Initializing DinoSegmenter with DINOv3 (ViT-S/16)...")
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.max_resolution = max_resolution

        # 1. Load DINOv3 Model from 'transformers'
        try:
            pretrained_model_names = [
                "facebook/dinov2-small",
                "facebook/dinov2-base",
                "facebook/dinov2-large",
                "facebook/dinov2-giant",
                "facebook/dinov3-vits16plus-pretrain-lvd1689m"
                "facebook/dinov3-vits16-pretrain-lvd1689m",
                "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            ]
            # Use the default model for now
            pretrained_model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
            
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.model = AutoModel.from_pretrained(
                pretrained_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa",
            ).to(self.device).eval()
            self.patch_size = self.model.config.patch_size
        except Exception as e:
            print(f"Error loading DINOv3 model from transformers: {e}")
            raise

        # 4. Initialize state
        self.original_image_size = None # (W, H)
        self.grid_size = None # (h, w)
        self.similarity_matrix = None # This will be our [N, N] matrix

        print(f"DINOv3 ViT-S/16 model loaded. Patch size: {self.patch_size}")

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        """
        Sets and pre-processes the image at the specified resolution.
        
        This runs the DINOv3 model ONCE and pre-computes the
        full N x N similarity matrix.
        
        Args:
            image (np.ndarray): The input image as an RGB numpy array [H, W, 3].
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")

        print(f"Setting new image, max resolution {self.max_resolution}px...")
        img_pil = Image.fromarray(image).convert("RGB")
        self.original_image_size = img_pil.size # (W, H)
        
        # --- 1. Get original image size ---
        W, H = img_pil.size
        
        # --- 2. NEW: Scale down if larger than max_resolution ---
        scale_factor = 1.0
        if max(W, H) > self.max_resolution:
            scale_factor = self.max_resolution / max(W, H)
        
        target_H = int(H * scale_factor)
        target_W = int(W * scale_factor)

        # --- 3. Calculate new size that is a multiple of patch_size ---
        new_H = (target_H // self.patch_size) * self.patch_size
        new_W = (target_W // self.patch_size) * self.patch_size
        
        if new_H == 0 or new_W == 0:
            raise ValueError(
                f"Scaled image size ({new_W}x{new_H}) is smaller than patch size "
                f"({self.patch_size}). Try a larger max_resolution."
            )

        # --- 4. Create the manual transform ---
        mean = self.processor.image_mean
        std = self.processor.image_std

        high_res_transforms = T.Compose([
            T.ToImage(), # Convert PIL to tensor
            T.ToDtype(torch.float32, scale=True),
            # Resize and crop to the *scaled* multiple of patch_size
            T.Resize((new_H, new_W), antialias=True), 
            T.Normalize(mean=mean, std=std),
        ])

        # --- 5. Apply the transforms ---
        img_tensor = high_res_transforms(img_pil).to(self.device, torch.float16).unsqueeze(0)
        
        B, C, H_proc, W_proc = img_tensor.shape
        
        h_grid = H_proc // self.patch_size
        w_grid = W_proc // self.patch_size
        self.grid_size = (h_grid, w_grid)
        n_patches = h_grid * w_grid

        # --- 6. Run the model on the high-res tensor ---
        out = self.model(pixel_values=img_tensor)
        hs = out.last_hidden_state.squeeze(0)
        patch_features = hs[-n_patches:, :] # [N, C]
        
        # --- 7. Pre-compute the N x N similarity matrix ---
        print(f"Got {n_patches} patches ({h_grid}x{w_grid} grid). Computing {n_patches}x{n_patches} matrix...")
        
        patch_features_norm = F.normalize(patch_features, p=2, dim=1)
        self.similarity_matrix = torch.matmul(patch_features_norm, patch_features_norm.T)
        
        print("Similarity matrix computed and stored.")

    def _transform_coords(self, coords: tuple) -> int:
        """Transforms original (y, x) coords to a flat patch_index."""
        y, x = coords
        W_orig, H_orig = self.original_image_size
        h_grid, w_grid = self.grid_size
        
        # Map click coordinates to original patch grid
        y_grid = int((y / H_orig) * h_grid)
        x_grid = int((x / W_orig) * w_grid)
        
        # Clamp to valid range
        y_clamped = max(0, min(y_grid, h_grid - 1))
        x_clamped = max(0, min(x_grid, w_grid - 1))
            
        patch_index = y_clamped * w_grid + x_clamped
        return patch_index

    @torch.no_grad()
    def get_similarity_map(self, original_coords: tuple) -> np.ndarray:
        """
        Gets a similarity map by looking up a pre-computed row.
        This is now an extremely fast operation.
        """
        if self.similarity_matrix is None:
            raise RuntimeError("An image must be set with set_image() before predicting.")
            
        # 1. Get query patch index
        patch_index = self._transform_coords(original_coords)
        
        # 2. Look up the pre-computed row of scores
        scores = self.similarity_matrix[patch_index] # [1, N]
        
        # 3. Dynamically normalize THIS ROW to [0, 1]
        min_val = torch.min(scores)
        max_val = torch.max(scores)
        
        if min_val == max_val:
            normalized_scores = torch.ones_like(scores)
        else:
            normalized_scores = (scores - min_val) / (max_val - min_val + 1e-8)
            
        # 4. Reshape to grid
        h, w = self.grid_size
        return normalized_scores.reshape(h, w).cpu().numpy()
        

if __name__ == "__main__":
    
    import time
    import matplotlib.pyplot as plt

    # 1. Load a sample image
    print("Starting to load sample image...")
    start_time = time.time()
    path = "test_images/veggies.jpg"
    try:
        image = Image.open(path).convert("RGB")
        image_np = np.array(image)
        
        print(f"Loaded image with shape: {image_np.shape}")
        download_time = time.time() - start_time
        print(f"Image download and loading completed in {download_time:.2f} seconds.")

        # 2. Initialize segmenter
        print("Starting to initialize DinoSegmenter...")
        start_time = time.time()
        # This is slow the first time as it downloads models
        segmenter = DinoSegmenter()
        init_time = time.time() - start_time
        print(f"DinoSegmenter initialization completed in {init_time:.2f} seconds.")

        # 3. Set the image (this is the main computation step)
        print("Starting to set the image and process features...")
        start_time = time.time()
        segmenter.set_image(image_np)
        set_image_time = time.time() - start_time
        print(f"Image setting and feature processing completed in {set_image_time:.2f} seconds.")

        # 4. Add prompts (coordinates are for the original image)
        print("Starting to add prompts...")
        start_time = time.time()
        # Positive point on the green broccoli
        segmenter.add_prompt(prompt_type='point', coords=[450, 400], is_positive=True)
        # Positive point on the orange carrot
        segmenter.add_prompt(prompt_type='point', coords=[600, 700], is_positive=True)
        
        # Negative point on the dark background
        segmenter.add_prompt(prompt_type='point', coords=[50, 50], is_positive=False)
        # Negative box on the bright white background
        segmenter.add_prompt(prompt_type='box', coords=[0, 1000, 200, 1280], is_positive=False)
        add_prompts_time = time.time() - start_time
        print(f"Adding prompts completed in {add_prompts_time:.2f} seconds.")

        # 5. Predict the mask
        print("Starting mask prediction...")
        start_time = time.time()
        # We use a threshold of 0.6 (since 0.5 is the new "neutral" point)
        mask = segmenter.predict_mask(threshold=0.6)
        predict_time = time.time() - start_time
        print(f"Mask prediction completed in {predict_time:.2f} seconds.")

        # 6. Plot results
        print("Starting to plot results...")
        start_time = time.time()
        fig, axs = plt.subplots(1, 3, figsize=(20, 7))
        
        axs[0].imshow(image_np)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Predicted Mask")
        axs[1].axis("off")

        # Create an overlay
        overlay = image_np.copy()
        overlay[mask] = [255, 0, 0] # Highlight mask in red
        
        axs[2].imshow(image_np)
        axs[2].imshow(overlay, alpha=0.5)
        axs[2].set_title("Mask Overlay")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()
        input("Press Enter to continue to the undo test...")
        plot_time = time.time() - start_time
        print(f"Plotting results completed in {plot_time:.2f} seconds.")
        
        # --- 7. Test undo ---
        print("Starting undo test...")
        start_time = time.time()
        segmenter.undo_prompt()  # Removes the negative box
        mask_undone = segmenter.predict_mask(threshold=0.6)
        
        plt.figure(figsize=(7, 7))
        plt.imshow(mask_undone, cmap='gray')
        plt.title("Mask after 'undo' (negative box removed)")
        plt.axis("off")
        plt.show()
        input("Press Enter to finish...")
        undo_time = time.time() - start_time
        print(f"Undo test completed in {undo_time:.2f} seconds.")
        
    except Exception as e:
        print(f"An error occurred during the demo: {e}")
        import traceback
        traceback.print_exc()
