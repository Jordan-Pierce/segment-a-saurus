import os
import sys
import glob
import argparse

import cv2
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider
)
from PyQt5.QtWidgets import QLabel as QLabelWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from src import DinoSegmenter


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if hasattr(self.parent(), 'handle_mouse_press'):
            self.parent().handle_mouse_press(event)
        super().mousePressEvent(event)


class InteractiveSegmenterApp(QWidget):
    """
    Wraps the DinoSegmenter to perform interactive segmentation using PyQt5 and QPixmap.
    """

    def __init__(self, folder_path: str, max_res: int = 1024):
        super().__init__()
        self.setWindowTitle("DINOv3 Segmentation - PyQt5 Demo")

        self.max_res = max_res

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
            self.segmenter = DinoSegmenter()
        except Exception as e:
            print(f"Fatal error: Could not initialize DinoSegmenter. Error: {e}")
            sys.exit(1)

        # 3. Run the "slow" setup step
        print("Setting image... (This may take a moment on first run)")
        self.segmenter.set_image(self.image_rgb, target_res=max_res)
        print("Image set. Ready for interaction.")

        # Initialize prompts and threshold
        self.threshold = 0.5
        self.app_prompts = []

        # Set up UI
        self.init_ui()

    def init_ui(self):
        self.image_label = ImageLabel(self)
        self.update_display()

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_label = QLabelWidget("Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.threshold_value_label = QLabelWidget(f"{self.threshold:.2f}")
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)

        layout.addLayout(threshold_layout)

        button_layout = QHBoxLayout()
        prev_button = QPushButton("Previous")
        prev_button.clicked.connect(self.prev_image)
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_image)
        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self.undo_prompt)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_prompts)
        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(self.close)

        button_layout.addWidget(prev_button)
        button_layout.addWidget(next_button)
        button_layout.addWidget(undo_button)
        button_layout.addWidget(clear_button)
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
            f"DINOv3 Segmentation - {os.path.basename(self.image_path)} "
            "(Left click: positive, Right click: negative, buttons for controls)"
        )
        self.setWindowTitle(title)

    def update_display(self):
        """Updates the display with prompts and mask overlay if available."""
        self.display_image = self.image_rgb.copy()

        # Draw prompts
        for prompt in self.app_prompts:
            y, x = prompt['coords']
            color = (0, 255, 0) if prompt['label'] == 1 else (255, 0, 0)  # Green for positive, red for negative
            cv2.circle(self.display_image, (x, y), 5, color, -1)

        # If there are prompts, predict and overlay mask
        if self.segmenter.prompts:
            try:
                confidence_map = self.segmenter.predict_mask()
                binary_mask = (confidence_map > self.threshold).astype(np.uint8)
                # Resize mask to original size
                mask_resized = cv2.resize(
                    binary_mask, (self.original_w, self.original_h),
                    interpolation=cv2.INTER_NEAREST
                )
                # Overlay
                highlight_color = np.array([0, 255, 0], dtype=np.uint8)  # Green overlay
                highlight_mask = mask_resized[..., None] * highlight_color
                self.display_image = cv2.add(self.display_image, highlight_mask.astype(np.uint8))
            except Exception as e:
                print(f"Error during mask prediction: {e}")

        # Convert numpy array to QImage
        height, width, channel = self.display_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def handle_mouse_press(self, event):
        """Handle mouse press event for adding prompts."""
        x = event.x()
        y = event.y()

        # Clamp coordinates
        y = max(0, min(y, self.original_h - 1))
        x = max(0, min(x, self.original_w - 1))

        if event.button() == Qt.LeftButton:
            self.app_prompts.append({'coords': (y, x), 'label': 1})
            self.segmenter.add_prompt((y, x), is_positive=True)
        elif event.button() == Qt.RightButton:
            self.app_prompts.append({'coords': (y, x), 'label': 0})
            self.segmenter.add_prompt((y, x), is_positive=False)

        self.update_display()

    def update_threshold(self, value):
        self.threshold = value / 100.0
        self.threshold_value_label.setText(f"{self.threshold:.2f}")
        self.update_display()

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.images)
        self.load_image(self.current_index)
        self.segmenter.set_image(self.image_rgb, target_res=self.max_res)
        self.app_prompts = []
        self.segmenter.clear_prompts()
        self.update_display()

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.images)
        self.load_image(self.current_index)
        self.segmenter.set_image(self.image_rgb, target_res=self.max_res)
        self.app_prompts = []
        self.segmenter.clear_prompts()
        self.update_display()

    def undo_prompt(self):
        if self.app_prompts:
            self.app_prompts.pop()
        self.segmenter.undo_prompt()
        self.update_display()

    def clear_prompts(self):
        self.app_prompts = []
        self.segmenter.clear_prompts()
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
        window = InteractiveSegmenterApp(args.folder, max_res=args.resolution)
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
