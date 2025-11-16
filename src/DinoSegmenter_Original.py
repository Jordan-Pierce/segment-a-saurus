import time

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T 

from transformers import AutoImageProcessor, AutoModel

import faiss


class DinoSegmenter:
    """
    A class for high-resolution, interactive segmentation.
    
    - Supports 'anyup' (high-quality) or 'bilinear' (fast) upsampling.
    - Also stores the low-res patch similarity matrix for
      fast, interactive "hover" previews.
    - Only supports positive (additive) prompts.
    """
    
    def __init__(self, 
                 dino_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m", 
                 device: str = None,
                 upsampling_method: str = "bilinear",
                 segmenter_method: str = "torch",
                 resize_method: str = "pad"): 
        
        print(f"Initializing DinoSegmenter with {dino_model_name}...")
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.upsampling_method = upsampling_method
        if self.upsampling_method not in ["anyup", "bilinear"]:
            raise ValueError("upsampling_method must be 'anyup' or 'bilinear'")
            
        self.segmenter_method = segmenter_method
        if self.segmenter_method not in ["faiss", "torch"]:
            raise ValueError("segmenter_method must be 'faiss' or 'torch'")
        print(f"Using segmentation method: {self.segmenter_method}")
        
        self.resize_method = resize_method
        if self.resize_method not in ["pad", "crop"]:
            raise ValueError("resize_method must be 'pad' or 'crop'")
        print(f"Using resize method: {self.resize_method}")

        self.is_vit = "vit" in dino_model_name.lower()
        attn_impl = "sdpa" if self.is_vit else "eager"

        # 1. Load DINOv3 Model
        try:
            print(f"Loading {dino_model_name}...")
            self.processor = AutoImageProcessor.from_pretrained(dino_model_name, use_fast=True)
            self.model = AutoModel.from_pretrained(
                dino_model_name,
                dtype=torch.float16,
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
        
        # --- Scale factor for medium-res bilinear ---
        self.bilinear_med_res_scale = self.patch_size // 4 
        # (This means 4x the patch grid, e.g., 28x28 -> 112x112)
        # ---

        # 2. Load AnyUp Model (Conditionally)
        self.upsampler = None
        if self.upsampling_method == "anyup":
            try:
                print("Loading AnyUp model from torch.hub...")
                self.upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(self.device).eval()
            except Exception as e:
                print(f"Error loading AnyUp model: {e}")
                raise
        else:
            print("Using 'bilinear' upsampling. AnyUp model not loaded.")
        
        # --- Initialize state ---
        self.hr_features = None 
        self.prompts = [] 
        self.transform_info = {} 
        self.grid_size = None 
        self.low_res_similarity_matrix = None 
            
        print("DinoSegmenter initialized successfully.")

    @torch.no_grad()
    def set_image(self, image: np.ndarray, target_res: int = 512):
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")

        action_desc = "padding" if self.resize_method == "pad" else "center cropping"
        print(f"Setting image: resizing smallest edge to {target_res} and {action_desc}...")
        img_pil = Image.fromarray(image).convert("RGB")
        W_orig, H_orig = img_pil.size
        
        # --- 1. Create transforms ---
        mean = self.processor.image_mean
        std = self.processor.image_std
        
        if self.resize_method == "pad":
            # Original padding logic
            transforms_pre = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(size=target_res, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            ])
            
            img_tensor_unpadded = transforms_pre(img_pil)
            _, H_proc_unpadded, W_proc_unpadded = img_tensor_unpadded.shape
            
            H_proc = (H_proc_unpadded + self.patch_size - 1) // self.patch_size * self.patch_size
            W_proc = (W_proc_unpadded + self.patch_size - 1) // self.patch_size * self.patch_size
            
            padding_right = W_proc - W_proc_unpadded
            padding_bottom = H_proc - H_proc_unpadded
            
            transforms_post = T.Compose([
                T.Pad(padding=(0, 0, padding_right, padding_bottom), fill=0),
                T.Normalize(mean=mean, std=std),
            ])
            
            hr_image_tensor_f32 = transforms_post(img_tensor_unpadded).to(self.device).unsqueeze(0)
            
        else:  # crop method
            # Simplified approach: make a square crop at target_res x target_res
            # Round down to nearest patch_size multiple
            patches_per_side = target_res // self.patch_size
            crop_size_processed = patches_per_side * self.patch_size
            
            # This will be our processed size (square)
            H_proc = W_proc = crop_size_processed
            
            # Ensure we have at least one patch
            if H_proc == 0:
                H_proc = W_proc = self.patch_size
            
            # For center crop, we need to determine what size to crop from the original image
            # We want the crop to fill the entire processed area after resize
            # So we crop a square from the original image with size = min(W_orig, H_orig)
            crop_size_orig = min(W_orig, H_orig)
            
            # But we need to ensure this doesn't go below our target
            # If the original image is too small, we'll have to crop less
            crop_h_orig = crop_w_orig = crop_size_orig
            
            transforms = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.CenterCrop(size=(crop_h_orig, crop_w_orig)),
                T.Resize(size=(H_proc, W_proc), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.Normalize(mean=mean, std=std),
            ])
            
            hr_image_tensor_f32 = transforms(img_pil).to(self.device).unsqueeze(0)
            
            # Calculate the scale factor from original crop to processed
            scale = H_proc / crop_h_orig  # Should be same for both dimensions since both are square
            
            # For crop method, store the crop information for coordinate transformations
            H_proc_unpadded, W_proc_unpadded = H_proc, W_proc

        # --- 2. Store processed tensor and calculate grid ---
        
        h_grid = H_proc // self.patch_size
        w_grid = W_proc // self.patch_size
        self.grid_size = (h_grid, w_grid) 
        n_patches = h_grid * w_grid

        # --- 3. DINOv3 Pass ---
        print("Running DINOv3 pass...")
        out = self.model(pixel_values=hr_image_tensor_f32.to(torch.float16))
        hs = out.last_hidden_state 
        patch_features = hs.squeeze(0)[-n_patches:, :] 
        lr_features_f16 = patch_features.permute(1, 0).reshape(1, -1, h_grid, w_grid).contiguous()

        # --- 4. Compute Low-Res Similarity Matrix (for hover) ---
        print(f"Computing {n_patches}x{n_patches} low-res similarity matrix...")
        patch_features_norm = F.normalize(patch_features, p=2, dim=1)
        self.low_res_similarity_matrix = torch.matmul(patch_features_norm, patch_features_norm.T)
        print("Low-res matrix computed.")
        
        # --- 5. Upsampling Pass (MODIFIED) ---
        if self.upsampling_method == "anyup":
            print("Running AnyUp pass...")
            if self.upsampler is None:
                raise RuntimeError("Upsampling method is 'anyup' but AnyUp model was not loaded.")
            hr_features = self.upsampler(
                hr_image_tensor_f32,
                lr_features_f16.to(torch.float32),
                q_chunk_size=1024
            )
        elif self.upsampling_method == "bilinear":
            # --- NEW: Upsample to MEDIUM resolution, not full ---
            med_res_size = (h_grid * self.bilinear_med_res_scale, w_grid * self.bilinear_med_res_scale)
            print(f"Running Bilinear upsampling to medium-res: {med_res_size}...")
            hr_features = F.interpolate(
                lr_features_f16.to(torch.float32),
                size=med_res_size,
                mode='bilinear',
                align_corners=False
            )
        
        # --- 6. Store state ---
        self.hr_features = hr_features.squeeze(0).permute(1, 2, 0).to(torch.float32) 
        self.prompts = []
        
        if self.resize_method == "pad":
            scale_factor = target_res / min(W_orig, H_orig)
            self.transform_info = {
                "original_size": (W_orig, H_orig),
                "processed_size": (H_proc, W_proc), 
                "unpadded_size": (H_proc_unpadded, W_proc_unpadded),
                "resize_scale": scale_factor,
                "resize_method": self.resize_method,
            }
        else:  # crop method
            self.transform_info = {
                "original_size": (W_orig, H_orig),
                "processed_size": (H_proc, W_proc),
                "crop_size_orig": (crop_h_orig, crop_w_orig),
                "final_scale": scale,
                "resize_method": self.resize_method,
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
        """
        Transforms (y, x) from original image to processed feature map.
        - 'anyup': returns (y, x) in PIXEL space.
        - 'bilinear': returns (y_med, x_med) in MEDIUM-RES space.
        
        Handles both 'pad' and 'crop' resize methods.
        """
        y_click, x_click = original_coords
        info = self.transform_info
        
        if info["resize_method"] == "pad":
            # Original padding logic
            # 1. Simulate the resize to get pixel coordinates
            y_proc = int(y_click * info["resize_scale"])
            x_proc = int(x_click * info["resize_scale"])
            
            # 2. Check if click is inside the un-padded area
            H_unpadded, W_unpadded = info["unpadded_size"]
            if not (0 <= y_proc < H_unpadded and 0 <= x_proc < W_unpadded):
                return None  # Click was outside the processed area
        else:
            # Crop method logic - new implementation
            W_orig, H_orig = info["original_size"]
            H_proc, W_proc = info["processed_size"]
            crop_h_orig, crop_w_orig = info["crop_size_orig"]
            scale = info["final_scale"]
            
            # Calculate center crop offsets in original coordinates
            crop_y_orig = (H_orig - crop_h_orig) // 2
            crop_x_orig = (W_orig - crop_w_orig) // 2
            
            # Check if click is within the cropped area in original coordinates
            if not (crop_x_orig <= x_click < crop_x_orig + crop_w_orig and
                    crop_y_orig <= y_click < crop_y_orig + crop_h_orig):
                return None  # Click was outside the cropped area
            
            # Transform to crop-relative coordinates then scale to processed coordinates
            y_crop_rel = y_click - crop_y_orig
            x_crop_rel = x_click - crop_x_orig
            
            y_proc = int(y_crop_rel * scale)
            x_proc = int(x_crop_rel * scale)
            
            # Clamp to processed dimensions
            y_proc = max(0, min(y_proc, H_proc - 1))
            x_proc = max(0, min(x_proc, W_proc - 1))
            
        # 3. Return coordinates based on upsampling method
        if self.upsampling_method == 'anyup':
            # AnyUp uses high-res pixel coordinates
            return (y_proc, x_proc)
        else:
            # Bilinear uses medium-res coordinates
            # We map the pixel coord (y_proc) to the medium-res grid
            # (e.g., 448px -> 112px grid by dividing by 4 (patch_size/med_res_scale))
            
            # This is the scaling factor from pixel space to medium-res space
            med_res_scale_factor = self.bilinear_med_res_scale / self.patch_size
            
            y_med = int(y_proc * med_res_scale_factor)
            x_med = int(x_proc * med_res_scale_factor)
            
            return (y_med, x_med)

    def _transform_coords_to_patch(self, original_coords: tuple) -> int | None:
        """
        Transforms (y, x) from original image to a flat *patch* index.
        Handles both 'pad' and 'crop' resize methods.
        """
        y_click, x_click = original_coords
        info = self.transform_info
        
        if info["resize_method"] == "pad":
            # Original padding logic
            # 1. Simulate the resize to get pixel coordinates
            y_proc = int(y_click * info["resize_scale"])
            x_proc = int(x_click * info["resize_scale"])
            
            # 2. Check if click is inside the un-padded area
            H_unpadded, W_unpadded = info["unpadded_size"]
            if not (0 <= y_proc < H_unpadded and 0 <= x_proc < W_unpadded):
                return None  # Click was outside the processed area
        else:
            # Crop method logic - new implementation
            W_orig, H_orig = info["original_size"]
            H_proc, W_proc = info["processed_size"]
            crop_h_orig, crop_w_orig = info["crop_size_orig"]
            scale = info["final_scale"]
            
            # Calculate center crop offsets in original coordinates
            crop_y_orig = (H_orig - crop_h_orig) // 2
            crop_x_orig = (W_orig - crop_w_orig) // 2
            
            # Check if click is within the cropped area in original coordinates
            if not (crop_x_orig <= x_click < crop_x_orig + crop_w_orig and
                    crop_y_orig <= y_click < crop_y_orig + crop_h_orig):
                return None  # Click was outside the cropped area
            
            # Transform to crop-relative coordinates then scale to processed coordinates
            y_crop_rel = y_click - crop_y_orig
            x_crop_rel = x_click - crop_x_orig
            
            y_proc = int(y_crop_rel * scale)
            x_proc = int(x_crop_rel * scale)
            
            # Clamp to processed dimensions
            y_proc = max(0, min(y_proc, H_proc - 1))
            x_proc = max(0, min(x_proc, W_proc - 1))
            
        # 3. Convert pixel coordinates directly to grid coordinates
        h_grid, w_grid = self.grid_size
        
        y_grid = y_proc // self.patch_size
        x_grid = x_proc // self.patch_size
        
        patch_index = y_grid * w_grid + x_grid
        n_patches = h_grid * w_grid
        
        if 0 <= patch_index < n_patches:
            return patch_index
        
        return None

    # --- INTERACTIVE METHODS (SIMPLIFIED) ---

    def add_prompt(self, original_coords: tuple):
        """Adds a single positive point prompt to the list."""
        processed_coords = self._transform_coords(original_coords)
        
        if processed_coords:
            self.prompts.append({"coords": processed_coords})
        else:
            pass

    def clear_prompts(self):
        self.prompts = []
        print("All prompts cleared.")

    def undo_prompt(self):
        if self.prompts:
            removed = self.prompts.pop()
            print(f"Removed last prompt at {removed['coords']}.")
        else:
            print("No prompts to undo.")

    @torch.no_grad()
    def predict_scores(self) -> np.ndarray:
        """
        Calls the correct segmentation method based on the
        segmenter_method set during initialization.
        """
        if self.segmenter_method == "faiss":
            return self.predict_scores_faiss()
        else: # "torch"
            return self.predict_scores_torch()

    @torch.no_grad()
    def predict_scores_torch(self) -> np.ndarray:
        """
        Calculates the HIGH-RES (pixel) similarity map using torch.matmul.
        Speed is PROPORTIONAL to the number of prompts.
        """
        t0 = time.time()
        if self.hr_features is None:
            raise RuntimeError("An image must be set with set_image() before predicting.")

        H_proc, W_proc, C = self.hr_features.shape
        all_features = self.hr_features.reshape(-1, C)

        if len(self.prompts) == 0:
            return np.zeros((H_proc, W_proc), dtype=np.float32)

        pos_features = []
        for prompt in self.prompts:
            y, x = prompt['coords']
            pos_features.append(self.hr_features[y, x])

        if pos_features:
            pos_map_flat = self._calculate_similarity(all_features, pos_features)
            pos_map_flat = (pos_map_flat + 1.0) / 2.0
        else:
            pos_map_flat = torch.zeros(H_proc * W_proc, device=self.device)
        
        pos_map = pos_map_flat.reshape(H_proc, W_proc).cpu().numpy()
        
        print(f"[torch] _calculate_similarity took {time.time() - t0:.4f}s")
        return pos_map

    @torch.no_grad()
    def predict_scores_faiss(self) -> np.ndarray:
        """
        Calculates the HIGH-RES (pixel) similarity map using Faiss.
        Speed is CONSTANT regardless of the number of prompts.
        """
        t0 = time.time()
        if self.hr_features is None:
            raise RuntimeError("An image must be set with set_image() before predicting.")

        H_proc, W_proc, C = self.hr_features.shape
        
        if len(self.prompts) == 0:
            return np.zeros((H_proc, W_proc), dtype=np.float32)

        pos_features = []
        for prompt in self.prompts:
            y, x = prompt['coords']
            pos_features.append(self.hr_features[y, x])
            
        # pos_features = []
        # if self.prompts:  # Make sure list isn't empty
        #     # Only get the very last point the user added
        #     prompt = self.prompts[-1] 
        #     y, x = prompt['coords']
        #     pos_features.append(self.hr_features[y, x])
            
        if not pos_features:
            return np.zeros((H_proc, W_proc), dtype=np.float32)

        all_features_flat = self.hr_features.reshape(-1, C)
        prompt_features = torch.stack(pos_features)
        
        all_features_norm = F.normalize(all_features_flat, p=2, dim=1)
        prompt_features_norm = F.normalize(prompt_features, p=2, dim=1)
        
        all_features_np = all_features_norm.cpu().numpy().astype('float32')
        prompt_features_np = prompt_features_norm.cpu().numpy().astype('float32')
        
        index = faiss.IndexFlatIP(C)
        index.add(prompt_features_np) 
        
        D, I = index.search(all_features_np, k=1)
        
        pos_map_flat = (D.squeeze() + 1.0) / 2.0
        
        print(f"[faiss] index build + search took {time.time() - t0:.4f}s")
        
        return pos_map_flat.reshape(H_proc, W_proc).astype(np.float32)

    @torch.no_grad()
    def get_low_res_similarity_map(self, original_coords: tuple) -> np.ndarray | None:
        """
        Gets a LOW-RES (patch) similarity map for hover.
        This is an extremely fast lookup.
        """
        if self.low_res_similarity_matrix is None:
            raise RuntimeError("An image must be set with set_image() before predicting.")
            
        patch_index = self._transform_coords_to_patch(original_coords)
        
        if patch_index is None:
            return None
        
        scores = self.low_res_similarity_matrix[patch_index] 
        
        min_val, max_val = torch.min(scores), torch.max(scores)
        if min_val == max_val:
            normalized_scores = torch.ones_like(scores)
        else:
            normalized_scores = (scores - min_val) / (max_val - min_val + 1e-8)
            
        h, w = self.grid_size
        return normalized_scores.reshape(h, w).cpu().numpy()

    def post_process_mask(self, confidence_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Aligns the processed confidence map with the original image.
        - For 'pad' resize method: Un-pads the mask and resizes to original.
        - For 'crop' resize method: Resizes mask and places it back in original image bounds.
        
        Both use BICUBIC upscaling.
        """
        if self.transform_info is None:
            raise RuntimeError("set_image() must be called before post-processing.")
            
        binary_mask = (confidence_map > threshold)
        
        info = self.transform_info
        W_orig, H_orig = info["original_size"]
        
        if info["resize_method"] == "pad":
            # Original padding logic
            H_unpadded, W_unpadded = info["unpadded_size"]
            
            if self.upsampling_method == 'anyup':
                # Un-pad the HIGH-RES pixel mask
                uncropped_mask = binary_mask[0:H_unpadded, 0:W_unpadded]
            else:
                # Un-pad the MEDIUM-RES mask
                # We must calculate the unpadded medium-res size
                med_res_scale_factor = self.bilinear_med_res_scale / self.patch_size
                H_unpadded_med = int(H_unpadded * med_res_scale_factor)
                W_unpadded_med = int(W_unpadded * med_res_scale_factor)
                
                # Crop the medium-res mask
                uncropped_mask = binary_mask[0:H_unpadded_med, 0:W_unpadded_med]
            
            # Resize the uncropped mask to original size
            mask_img = Image.fromarray((uncropped_mask * 255).astype(np.uint8))
            mask_resized = np.array(mask_img.resize((W_orig, H_orig), Image.BICUBIC))
            
        else:
            # Crop method logic - simplified
            crop_h_orig, crop_w_orig = info["crop_size_orig"]
            
            # The binary_mask is at the processed size (H_proc, W_proc)
            # We need to resize it to the crop size and place it in the original image
            
            # Resize the processed mask to the original crop area size
            mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_crop_resized = np.array(mask_img.resize((crop_w_orig, crop_h_orig), Image.BICUBIC))
            
            # Create full-size mask and place the resized crop in the center
            mask_resized = np.zeros((H_orig, W_orig), dtype=np.uint8)
            
            # Calculate center crop offsets in original coordinates
            crop_y_orig = (H_orig - crop_h_orig) // 2
            crop_x_orig = (W_orig - crop_w_orig) // 2
            
            # Place the crop mask in the center of the original image
            y_end = crop_y_orig + crop_h_orig
            x_end = crop_x_orig + crop_w_orig
            
            # Ensure we don't go out of bounds
            y_end = min(y_end, H_orig)
            x_end = min(x_end, W_orig)
            
            mask_resized[crop_y_orig:y_end, crop_x_orig:x_end] = \
                mask_crop_resized[0:y_end - crop_y_orig, 0:x_end - crop_x_orig]
        
        return (mask_resized > 128)


# (Main test function is disabled)
if __name__ == "__main__":
    print("This file is intended to be used as a module.") 
