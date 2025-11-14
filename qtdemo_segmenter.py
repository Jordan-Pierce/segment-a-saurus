import os
import sys
import glob
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider
)
from PyQt5.QtWidgets import QLabel as QLabelWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt

from src import DinoSegmenter


class ImageLabel(QLabel):
    """
    A custom QLabel that captures mouse press, move, and release events
    for drawing interactions.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if hasattr(self.parent(), 'handle_mouse_down'):
            self.parent().handle_mouse_down(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if hasattr(self.parent(), 'handle_mouse_move'):
            self.parent().handle_mouse_move(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if hasattr(self.parent(), 'handle_mouse_release'):
            self.parent().handle_mouse_release(event)
        super().mouseReleaseEvent(event)


class InteractiveSegmenterApp(QWidget):
    """
    Wraps the DinoSegmenter to perform interactive segmentation using PyQt5.
    Supports drawing prompts (mouse drag) and updates mask on release.
    """

    def __init__(self, folder_path: str, max_res: int = 1024):
        super().__init__()
        self.setWindowTitle("DINOv3 Segmentation - PyQt5 Demo")

        self.max_res = max_res
        self.images = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
        )
        if not self.images:
            raise ValueError(f"No .jpg or .png images found in {folder_path}")

        self.current_index = 0
        self.load_image(self.current_index)

        try:
            self.segmenter = DinoSegmenter()
        except Exception as e:
            print(f"Fatal error: Could not initialize DinoSegmenter. Error: {e}")
            sys.exit(1)

        print("Setting image... (This may take a moment on first run)")
        self.segmenter.set_image(self.image_rgb, target_res=max_res)
        print("Image set. Ready for interaction.")

        # --- State for drawing ---
        self.is_drawing = False
        self.draw_label = 1  # 1 for positive, 0 for negative
        # ---
        
        self.threshold = 0.5
        self.contrast = 10.0
        self.app_prompts = []

        self.init_ui()

    def init_ui(self):
        self.image_label = ImageLabel(self)
        self.draw_prompts_overlay() # Start with just the image

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        # --- Threshold Slider ---
        threshold_layout = QHBoxLayout()
        threshold_label = QLabelWidget("Threshold (Cutoff):")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.threshold_slider.valueChanged.connect(self.update_sliders)
        self.threshold_value_label = QLabelWidget(f"{self.threshold:.2f}")
        threshold_layout.addLayout(self.create_slider_layout(
            "Threshold (Cutoff):", self.threshold_slider, self.threshold_value_label
        ))
        layout.addLayout(threshold_layout)

        # --- Contrast Slider ---
        contrast_layout = QHBoxLayout()
        contrast_label = QLabelWidget("Contrast (Steepness):")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(1)
        self.contrast_slider.setMaximum(50)
        self.contrast_slider.setValue(int(self.contrast))
        self.contrast_slider.valueChanged.connect(self.update_sliders)
        self.contrast_value_label = QLabelWidget(f"{self.contrast:.1f}")
        contrast_layout.addLayout(self.create_slider_layout(
            "Contrast (Steepness):", self.contrast_slider, self.contrast_value_label
        ))
        layout.addLayout(contrast_layout)

        # Button layout
        button_layout = QHBoxLayout()
        # ... (buttons are the same)
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

    def create_slider_layout(self, label_text, slider, value_label):
        """Helper to create a consistent slider layout."""
        layout = QHBoxLayout()
        label = QLabelWidget(label_text)
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_label)
        return layout

    def load_image(self, index):
        """Loads the image at the given index."""
        self.image_path = self.images[index]
        pil_image = Image.open(self.image_path).convert("RGBA")
        self.image_rgb = np.array(pil_image)
        self.original_h, self.original_w = self.image_rgb.shape[:2]
        self.display_image = self.image_rgb.copy()
        
        title = (
            f"DINOv3 Segmentation - {os.path.basename(self.image_path)} "
            "(Drag Left: positive, Drag Right: negative)"
        )
        self.setWindowTitle(title)
        
        # We'll update the pixmap in draw_prompts_overlay
        
    def draw_prompts_overlay(self):
        """
        FAST update. Only draws the base image and the prompt circles.
        """
        self.display_image = self.image_rgb.copy()
        temp_rgb = self.display_image[:, :, :3].copy()
        
        for prompt in self.app_prompts:
            y, x = prompt['coords']
            color = (0, 255, 0) if prompt['label'] == 1 else (255, 0, 0)
            cv2.circle(temp_rgb, (x, y), 5, color, -1)
            
        self.display_image[:, :, :3] = temp_rgb
        
        height, width, channel = self.display_image.shape
        bytes_per_line = 4 * width
        q_img = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        current_pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(current_pixmap)

    def update_mask(self):
        """
        SLOW update. Runs segmentation and overlays the mask.
        Assumes draw_prompts_overlay() has already been called.
        """
        if self.segmenter.prompts:
            try:
                # A. Get raw 0-1 similarity maps
                pos_map, neg_map = self.segmenter.predict_scores()

                # B. Convert to torch tensors
                device = self.segmenter.device
                pos_map_torch = torch.from_numpy(pos_map).to(device)
                neg_map_torch = torch.from_numpy(neg_map).to(device)

                # C. Apply sigmoid
                pos_conf = 1.0 / (1.0 + torch.exp(-self.contrast * (pos_map_torch - self.threshold)))
                neg_conf = 1.0 / (1.0 + torch.exp(-self.contrast * (neg_map_torch - self.threshold)))

                # D. Subtract negative from positive
                confidence_map_torch = torch.clamp(pos_conf - neg_conf, min=0.0, max=1.0)
                
                # E. Post-process to align to original image
                confidence_map = confidence_map_torch.cpu().numpy()
                final_mask = self.segmenter.post_process_mask(
                    confidence_map, threshold=0.5
                )

                # F. Overlay the mask
                mask_uint8 = (final_mask * 255).astype(np.uint8)
                H, W = final_mask.shape
                q_image_mask = QImage(mask_uint8.data, W, H, W, QImage.Format_Grayscale8)
                
                widget_size = self.image_label.size()
                scaled_mask = q_image_mask.scaled(
                    widget_size, 
                    Qt.IgnoreAspectRatio,
                    Qt.FastTransformation
                )
                
                overlay_pixmap = QPixmap.fromImage(scaled_mask)
                
                # We need to get the *current* pixmap (with prompts) and paint on it
                current_pixmap = self.image_label.pixmap()
                painter = QPainter(current_pixmap)
                painter.setOpacity(0.5)
                painter.drawPixmap(0, 0, overlay_pixmap)
                painter.end()
                
                self.image_label.setPixmap(current_pixmap) # Set the final combined pixmap

            except Exception as e:
                print(f"Error during mask overlay: {e}")

    def add_point_at_event(self, event):
        """Helper to add a prompt and draw the visual feedback."""
        x = event.x()
        y = event.y()
        
        # Clamp coordinates
        y = max(0, min(y, self.original_h - 1))
        x = max(0, min(x, self.original_w - 1))

        self.app_prompts.append({'coords': (y, x), 'label': self.draw_label})
        self.segmenter.add_prompt((y, x), is_positive=(self.draw_label == 1))
        
        # Fast update: just draw the new circle
        self.draw_prompts_overlay()

    def handle_mouse_down(self, event):
        """Starts a drawing session."""
        self.is_drawing = True
        if event.button() == Qt.LeftButton:
            self.draw_label = 1
        elif event.button() == Qt.RightButton:
            self.draw_label = 0
        else:
            self.is_drawing = False # Don't draw for middle mouse, etc.
            return
            
        self.add_point_at_event(event)

    def handle_mouse_move(self, event):
        """Adds points during a drawing session."""
        if self.is_drawing:
            self.add_point_at_event(event)
            
    def handle_mouse_release(self, event):
        """Ends a drawing session and triggers segmentation."""
        if self.is_drawing:
            self.is_drawing = False
            print("Drawing finished. Updating mask...")
            self.update_mask() # Run the slow segmentation

    def update_sliders(self, value):
        """Called when either slider moves. Triggers a full mask update."""
        self.threshold = self.threshold_slider.value() / 100.0
        self.threshold_value_label.setText(f"{self.threshold:.2f}")

        self.contrast = self.contrast_slider.value()
        self.contrast_value_label.setText(f"{self.contrast:.1f}")

        # Update the display with new values
        self.draw_prompts_overlay()
        self.update_mask()

    def update_display_full(self):
        """Helper to do a full-refresh (prompts + mask)."""
        self.draw_prompts_overlay()
        self.update_mask()

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.images)
        self.load_image(self.current_index)
        self.segmenter.set_image(self.image_rgb, target_res=self.max_res)
        self.app_prompts = []
        self.segmenter.clear_prompts()
        self.draw_prompts_overlay() # Just show image

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.images)
        self.load_image(self.current_index)
        self.segmenter.set_image(self.image_rgb, target_res=self.max_res)
        self.app_prompts = []
        self.segmenter.clear_prompts()
        self.draw_prompts_overlay() # Just show image

    def undo_prompt(self):
        if self.app_prompts:
            self.app_prompts.pop()
        self.segmenter.undo_prompt()
        self.update_display_full()

    def clear_prompts(self):
        self.app_prompts = []
        self.segmenter.clear_prompts()
        self.update_display_full()

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
        help="Target resolution for the smallest edge. Smaller is faster."
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