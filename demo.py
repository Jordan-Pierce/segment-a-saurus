import sys

import argparse
import cv2
import numpy as np
import os
import glob

from dino_segmenter import DinoSegmenter


class InteractiveVisualizerApp:
    """
    Wraps the NEW DinoSegmenter to visualize DINOv3 feature similarity,
    replicating the original web demo.
    """
    
    def __init__(self, folder_path: str):
        self.window_name = "DINOv3 Similarity"
        
        # 1. Get all images in the folder
        self.images = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.png"))
        )
        if not self.images:
            raise ValueError(f"No .jpg or .png images found in {folder_path}")
        
        self.current_index = 0
        self.load_image(self.current_index)
        
        # 2. Initialize the segmenter
        try:
            self.segmenter = DinoSegmenter()
        except Exception as e:
            print(f"Fatal error: Could not initialize DinoSegmenter.")
            print(f"Error details: {e}")
            sys.exit(1)
            
        # 3. Run the "slow" setup step
        print("Setting image... (This may take a moment on first run)")
        self.segmenter.set_image(self.image_rgb)
        print("Image set. Ready for interaction.")

    def load_image(self, index):
        """Loads the image at the given index."""
        self.image_path = self.images[index]
        
        # Load the image
        self.image_bgr = cv2.imread(self.image_path)
        if self.image_bgr is None:
            raise FileNotFoundError(f"Could not load image from {self.image_path}")
            
        # Store original size
        self.original_h, self.original_w = self.image_bgr.shape[:2]
            
        # Create a dimmed version for the overlay
        self.image_dimmed = (self.image_bgr * 0.5).astype(np.uint8)
        
        # Convert to RGB for the segmenter
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        
        # Initialize state
        self.display_image = self.image_bgr.copy()
        
        # Update window title
        title = (
            f"DINOv3 Similarity - {os.path.basename(self.image_path)} "
            "(Move mouse to explore, arrows to cycle, 'q' to quit)"
        )
        cv2.setWindowTitle(self.window_name, title)

    def _create_overlay(self, low_res_map: np.ndarray):
        """
        Creates an overlay mimicking the web demo:
        A dimmed image with bright highlights.
        """
        
        # Resize the low-res (e.g., 14x14) map to the full image size
        # Using INTER_NEAREST creates the blocky patch look
        confidence_map = cv2.resize(
            low_res_map,
            (self.original_w, self.original_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Create a yellow highlight color (or any color you prefer)
        highlight_color = np.array([0, 255, 255], dtype=np.uint8) # BGR
        
        # Scale confidence map (0.0-1.0) to a 3-channel BGR mask
        highlight_mask = confidence_map[..., None] * highlight_color
        
        # Add the highlights to the dimmed image
        self.display_image = cv2.add(
            self.image_dimmed, 
            highlight_mask.astype(np.uint8)
        )

    def _refresh(self, low_res_map: np.ndarray = None):
        """Updates the display with a new confidence map or resets it."""
        if low_res_map is not None:
            self._create_overlay(low_res_map)
        else:
            # Reset to the original image
            self.display_image = self.image_bgr.copy()
            
        cv2.imshow(self.window_name, self.display_image)

    def handle_mouse(self, event, x, y, flags, param):
        """OpenCV mouse callback function."""
        
        # On mouse move, get similarity and refresh
        if event == cv2.EVENT_MOUSEMOVE:
            # Clamp coordinates to be within image bounds
            y = max(0, min(y, self.original_h - 1))
            x = max(0, min(x, self.original_w - 1))
            
            try:
                # This is now just a fast lookup
                low_res_similarity_map = self.segmenter.get_similarity_map(
                    original_coords=(y, x)
                )
                self._refresh(low_res_map=low_res_similarity_map)
            except Exception as e:
                print(f" Error during similarity lookup: {e}")

    def run(self):
        """Starts the main application loop."""
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.handle_mouse)
        
        # Show initial image
        cv2.imshow(self.window_name, self.display_image)
        
        print("\n--- Controls ---")
        print("Move mouse:   Show similarity")
        print("Left arrow:   Previous image")
        print("Right arrow:  Next image")
        print("q:            Quit")
        print("----------------\n")
        
        while True:
            key = cv2.waitKey(1)
            
            # Quit
            if key == ord('q'):
                break
                
            # Left arrow
            elif key == 2424832:
                self.current_index = (self.current_index - 1) % len(self.images)
                self.load_image(self.current_index)
                self.segmenter.set_image(self.image_rgb)
                self._refresh()
                
            # Right arrow
            elif key == 2555904:
                self.current_index = (self.current_index + 1) % len(self.images)
                self.load_image(self.current_index)
                self.segmenter.set_image(self.image_rgb)
                self._refresh()
                
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Interactive DINOv2 Segmentation Tool"
    )
    parser.add_argument(
        "-f", "--folder", 
        default="data/",
        help="Path to the folder containing images."
    )
    args = parser.parse_args()
    
    try:
        app = InteractiveVisualizerApp(args.folder)
        app.run()
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
