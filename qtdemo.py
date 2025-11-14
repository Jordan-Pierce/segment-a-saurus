import os
import sys
import glob
import argparse

import cv2
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from dino_segmenter import DinoSegmenter


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        if hasattr(self.parent(), 'handle_mouse_move'):
            self.parent().handle_mouse_move(event)
        super().mouseMoveEvent(event)


class InteractiveVisualizerApp(QWidget):
    """
    Wraps the DinoSegmenter to visualize DINOv3 feature similarity using PyQt5 and QPixmap.
    """

    def __init__(self, folder_path: str, max_res: int = 1024):
        super().__init__()
        self.setWindowTitle("DINOv3 Similarity - PyQt5 Demo")

        # 1. Get all images in the folder
        self.images = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
        )
        if not self.images:
            raise ValueError(f"No .jpg or .png images found in {folder_path}")

        self.current_index = 0
        self.load_image(self.current_index)

        # 2. Initialize the segmenter
        try:
            self.segmenter = DinoSegmenter(max_resolution=max_res)
        except Exception as e:
            print(f"Fatal error: Could not initialize DinoSegmenter. Error: {e}")
            sys.exit(1)

        # 3. Run the "slow" setup step
        print("Setting image... (This may take a moment on first run)")
        self.segmenter.set_image(self.image_rgb)
        print("Image set. Ready for interaction.")

        # Set up UI
        self.init_ui()

    def init_ui(self):
        self.image_label = ImageLabel(self)
        self.update_display()

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        prev_button = QPushButton("Previous")
        prev_button.clicked.connect(self.prev_image)
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_image)
        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(self.close)

        button_layout.addWidget(prev_button)
        button_layout.addWidget(next_button)
        button_layout.addWidget(quit_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_image(self, index):
        """Loads the image at the given index."""
        self.image_path = self.images[index]

        # Load the image using PIL
        pil_image = Image.open(self.image_path).convert("RGB")
        self.image_rgb = np.array(pil_image)

        # Store original size
        self.original_h, self.original_w = self.image_rgb.shape[:2]

        # Create a dimmed version for the overlay
        self.image_dimmed = (self.image_rgb * 0.5).astype(np.uint8)

        # Initialize display image
        self.display_image = self.image_rgb.copy()

        # Update window title
        title = (
            f"DINOv3 Similarity - {os.path.basename(self.image_path)} "
            "(Move mouse to explore, use buttons or arrows to cycle, 'q' to quit)"
        )
        self.setWindowTitle(title)

    def update_display(self, low_res_map: np.ndarray = None):
        """Updates the display with a new confidence map or resets it."""
        if low_res_map is not None:
            self._create_overlay(low_res_map)
        else:
            self.display_image = self.image_rgb.copy()

        # Convert numpy array to QImage
        height, width, channel = self.display_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def _create_overlay(self, low_res_map: np.ndarray):
        """
        Creates an overlay mimicking the web demo:
        A dimmed image with bright highlights.
        """
        # Resize the low-res map to full image size
        confidence_map = cv2.resize(
            low_res_map,
            (self.original_w, self.original_h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create highlight mask
        highlight_color = np.array([255, 255, 255], dtype=np.uint8)  # RGB
        highlight_mask = confidence_map[..., None] * highlight_color
        
        # Add highlights to dimmed image
        self.display_image = cv2.add(
            self.image_dimmed,
            highlight_mask.astype(np.uint8)
        ).astype(np.uint8)

    def handle_mouse_move(self, event):
        """Handle mouse move event."""
        x = event.x()
        y = event.y()

        # Clamp coordinates
        y = max(0, min(y, self.original_h - 1))
        x = max(0, min(x, self.original_w - 1))

        try:
            low_res_similarity_map = self.segmenter.get_similarity_map(
                original_coords=(y, x)
            ).astype(np.float32)
            self.update_display(low_res_map=low_res_similarity_map)
        except Exception as e:
            print(f"Error during similarity lookup: {e}")

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.images)
        self.load_image(self.current_index)
        self.segmenter.set_image(self.image_rgb)
        self.update_display()

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.images)
        self.load_image(self.current_index)
        self.segmenter.set_image(self.image_rgb)
        self.update_display()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Q:
            self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive DINOv3 Segmentation Tool with PyQt5"
    )
    parser.add_argument(
        "-f", "--folder",
        default="data/",
        help="Path to the folder containing images."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=448,
        help="Max resolution (in pixels) to process the image at. Smaller is faster."
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    try:
        window = InteractiveVisualizerApp(args.folder, max_res=args.resolution)
        window.show()
        sys.exit(app.exec_())
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
