import os
import sys
import glob
import time
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
    A custom QLabel that captures mouse press, move, release,
    and leave events.
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
        
    def leaveEvent(self, event):
        if hasattr(self.parent(), 'handle_mouse_leave'):
            self.parent().handle_mouse_leave(event)
        super().leaveEvent(event)


class InteractiveSegmenterApp(QWidget):
    """
    Wraps the DinoSegmenter for interactive segmentation.
    - HOVER (not drawing): Shows low-res patch similarity.
    - DRAG (drawing): Shows throttled high-res pixel segmentation (positive only).
    - Uses a single "Threshold" slider for cutoff.
    """

    def __init__(self, 
                 folder_path: str, 
                 max_res: int = 1024, 
                 upsampler: str = "anyup",
                 segmenter: str = "faiss",
                 resize_method: str = "pad"):
        super().__init__()
        
        self.max_res = max_res
        self.images = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
        )
        if not self.images:
            raise ValueError(f"No .jpg or .png images found in {folder_path}")

        self.current_index = 0
        self.load_image(self.current_index)

        try:
            self.segmenter = DinoSegmenter(upsampling_method=upsampler, 
                                           segmenter_method=segmenter,
                                           resize_method=resize_method)
        except Exception as e:
            print(f"Fatal error: Could not initialize DinoSegmenter. Error: {e}")
            sys.exit(1)
        
        title = f"DINOv3 Segmentation ({upsampler.upper()}, {resize_method.upper()}) - PyQt5 Demo"
        self.setWindowTitle(title)

        print("Setting image... (This may take a moment on first run)")
        self.segmenter.set_image(self.image_rgb, target_res=max_res)
        print("Image set. Ready for interaction.")

        # --- State for drawing ---
        self.is_drawing = False
        self.move_event_counter = 0
        self.MOVE_EVENT_THRESHOLD = 3
        
        self.last_hover_pos = None
        self.threshold = 0.95
        # --- Contrast is now a hard-coded constant ---
        self.HARD_CONTRAST_VALUE = 50.0 
        self.app_prompts = [] 

        self.init_ui()

    def init_ui(self):
        self.image_label = ImageLabel(self)
        self.draw_prompts_overlay() 

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        # --- Simplified Sliders (Threshold only) ---
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0); self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.threshold_slider.valueChanged.connect(self.update_sliders)
        self.threshold_value_label = QLabelWidget(f"{self.threshold:.2f}")
        threshold_layout.addLayout(self.create_slider_layout(
            "Threshold:", self.threshold_slider, self.threshold_value_label
        ))
        
        layout.addLayout(threshold_layout)
        # --- Contrast Slider Removed ---

        # Buttons
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
        button_layout.addWidget(prev_button); button_layout.addWidget(next_button)
        button_layout.addWidget(undo_button); button_layout.addWidget(clear_button)
        button_layout.addWidget(quit_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def create_slider_layout(self, label_text, slider, value_label):
        layout = QHBoxLayout()
        layout.addWidget(QLabelWidget(label_text))
        layout.addWidget(slider)
        layout.addWidget(value_label)
        return layout

    def load_image(self, index):
        self.image_path = self.images[index]
        pil_image = Image.open(self.image_path).convert("RGBA")
        self.image_rgb = np.array(pil_image)
        self.original_h, self.original_w = self.image_rgb.shape[:2]
        self.display_image = self.image_rgb.copy()
        
        base_title = self.windowTitle().split(" - ")[0] 
        self.setWindowTitle(
            f"{base_title} - {os.path.basename(self.image_path)} "
            "(Hover: low-res, Drag Left: high-res)"
        )
        
    def draw_prompts_overlay(self):
        self.display_image = self.image_rgb.copy()
        temp_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_RGBA2RGB)
        
        color = (0, 255, 0) 
        for prompt in self.app_prompts:
            y, x = prompt['coords']
            cv2.circle(temp_rgb, (x, y), 5, color, -1)
        
        self.display_image = cv2.cvtColor(temp_rgb, cv2.COLOR_RGB2RGBA)
        
        height, width, channel = self.display_image.shape
        bytes_per_line = 4 * width
        q_img = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        current_pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(current_pixmap)

    def update_mask(self):
        """
        SLOW update. Runs high-res segmentation and overlays the mask.
        (Simplified: no negative map, fixed contrast)
        """
        if self.segmenter.prompts:
            try:
                
                t0 = time.time()
                pos_map = self.segmenter.predict_scores()
                t1 = time.time()
                print(f"[{self.segmenter.segmenter_method}] "
                      f"predict_scores took: {t1-t0:.4f}s for {len(self.segmenter.prompts)} prompts")

                device = self.segmenter.device
                pos_map_torch = torch.from_numpy(pos_map).to(device)

                # --- SIMPLIFIED: Use hard-coded contrast value ---
                pos_conf = 1.0 / (1.0 + torch.exp(-self.HARD_CONTRAST_VALUE * (pos_map_torch - self.threshold)))
                
                confidence_map_torch = torch.clamp(pos_conf, min=0.0, max=1.0)
                
                confidence_map = confidence_map_torch.cpu().numpy()
                final_mask = self.segmenter.post_process_mask(
                    confidence_map, threshold=0.5
                )

                # Debug output for crop method
                if hasattr(self.segmenter, 'transform_info') and self.segmenter.transform_info.get("resize_method") == "crop":
                    nonzero_count = np.count_nonzero(final_mask)
                    if nonzero_count > 0:
                        nonzero_indices = np.nonzero(final_mask)
                        min_y, max_y = nonzero_indices[0].min(), nonzero_indices[0].max()
                        min_x, max_x = nonzero_indices[1].min(), nonzero_indices[1].max()
                        print(f"Crop mode: Mask has {nonzero_count} pixels, bounds Y:{min_y}-{max_y}, X:{min_x}-{max_x}")
                        
                        # Check if mask is in expected crop area
                        info = self.segmenter.transform_info
                        crop_h, crop_w = info["crop_size_orig"]
                        orig_h, orig_w = info["original_size"]
                        expected_y_start = (orig_h - crop_h) // 2
                        expected_x_start = (orig_w - crop_w) // 2
                        expected_y_end = expected_y_start + crop_h
                        expected_x_end = expected_x_start + crop_w
                        print(f"Expected crop area: Y:{expected_y_start}-{expected_y_end}, X:{expected_x_start}-{expected_x_end}")

                # --- Overlay the mask ---
                mask_uint8 = (final_mask * 255).astype(np.uint8)
                H, W = final_mask.shape
                q_image_mask = QImage(mask_uint8.data, W, H, W, QImage.Format_Grayscale8)
                
                current_pixmap = self.image_label.pixmap()
                painter = QPainter(current_pixmap)
                painter.setOpacity(0.5)
                
                # The final_mask from post_process_mask is already at original image size
                # and properly positioned for both crop and pad modes, so we can treat them the same
                widget_size = self.image_label.size()
                scaled_mask = q_image_mask.scaled(
                    widget_size, 
                    Qt.IgnoreAspectRatio,
                    Qt.FastTransformation
                )
                
                overlay_pixmap = QPixmap.fromImage(scaled_mask)
                painter.drawPixmap(0, 0, overlay_pixmap)
                
                painter.end()
                self.image_label.setPixmap(current_pixmap)

            except Exception as e:
                print(f"Error during mask overlay: {e}")

    def draw_hover_overlay(self, low_res_map: np.ndarray):
        try:
            self.draw_prompts_overlay()
            
            # Handle crop vs pad differently
            transform_info = getattr(self.segmenter, 'transform_info', {})
            if transform_info.get("resize_method") == "crop":
                
                # For crop mode: only display hover overlay in the center cropped area
                info = transform_info
                crop_h, crop_w = info["crop_size_orig"]  # (height, width)
                orig_w, orig_h = info["original_size"]   # (width, height)
                
                print(f"Original size: {orig_w}x{orig_h}, Crop size: {crop_w}x{crop_h}")
                
                # Resize low_res_map to the crop area size
                low_res_map_resized = cv2.resize(
                    low_res_map,
                    (crop_w, crop_h),
                    interpolation=cv2.INTER_NEAREST
                )
                
                low_res_map_uint8 = (low_res_map_resized * 255).astype(np.uint8)
                colormap = cv2.applyColorMap(low_res_map_uint8, cv2.COLORMAP_JET)
                colormap_rgba = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGBA)
                
                H, W, C = colormap_rgba.shape
                q_image_mask = QImage(colormap_rgba.data, W, H, 4 * W, QImage.Format_RGBA8888)
                
                # Calculate crop area position in original coordinates
                crop_y_start = (orig_h - crop_h) // 2
                crop_x_start = (orig_w - crop_w) // 2
                
                # Since widget coordinates map directly to image pixels (no scaling in QLabel),
                # the crop area coordinates are the same as in the original image
                widget_crop_x = crop_x_start
                widget_crop_y = crop_y_start
                widget_crop_w = crop_w
                widget_crop_h = crop_h
                
                print(f"Crop area: ({widget_crop_x}, {widget_crop_y}) size {widget_crop_w}x{widget_crop_h}")
                
                # Scale the colormap to the widget crop area size
                scaled_mask = q_image_mask.scaled(
                    widget_crop_w, widget_crop_h,
                    Qt.IgnoreAspectRatio,
                    Qt.FastTransformation
                )
                
                overlay_pixmap = QPixmap.fromImage(scaled_mask)
                
                current_pixmap = self.image_label.pixmap()
                painter = QPainter(current_pixmap)
                painter.setOpacity(0.4)
                # Draw overlay only in the crop area
                painter.drawPixmap(widget_crop_x, widget_crop_y, overlay_pixmap)
                painter.end()
                
            else:
                # For pad mode: use original behavior (scale to entire image)
                low_res_map_resized = cv2.resize(
                    low_res_map,
                    (self.original_w, self.original_h),
                    interpolation=cv2.INTER_NEAREST
                )
                
                low_res_map_uint8 = (low_res_map_resized * 255).astype(np.uint8)
                colormap = cv2.applyColorMap(low_res_map_uint8, cv2.COLORMAP_JET)
                colormap_rgba = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGBA)
                
                H, W, C = colormap_rgba.shape
                q_image_mask = QImage(colormap_rgba.data, W, H, 4 * W, QImage.Format_RGBA8888)
                overlay_pixmap = QPixmap.fromImage(q_image_mask)
                
                current_pixmap = self.image_label.pixmap()
                painter = QPainter(current_pixmap)
                painter.setOpacity(0.4) 
                painter.drawPixmap(0, 0, overlay_pixmap)
                painter.end()
            
            self.image_label.setPixmap(current_pixmap)

        except Exception as e:
            print(f"Error during hover overlay: {e}")

    def add_point_at_event(self, event):
        x = event.x()
        y = event.y()
        
        y = max(0, min(y, self.original_h - 1))
        x = max(0, min(x, self.original_w - 1))

        self.app_prompts.append({'coords': (y, x)})
        self.segmenter.add_prompt((y, x))
        
        self.draw_prompts_overlay()

    def handle_mouse_down(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.last_hover_pos = None 
            
            self.add_point_at_event(event)
            self.update_mask() 
        else:
            self.is_drawing = False

    def handle_mouse_move(self, event):
        if self.is_drawing:
            self.add_point_at_event(event) 
            self.move_event_counter += 1
            if self.move_event_counter % self.MOVE_EVENT_THRESHOLD == 0:
                self.update_mask()
        
        else:
            pos = (event.x(), event.y())
            if pos == self.last_hover_pos:
                return 
            self.last_hover_pos = pos
            
            low_res_map = self.segmenter.get_low_res_similarity_map(
                original_coords=(event.y(), event.x())
            )
            if low_res_map is not None:
                self.draw_hover_overlay(low_res_map)
            
    def handle_mouse_release(self, event):
        if self.is_drawing:
            self.is_drawing = False
            self.move_event_counter = 0 
            print("Drawing finished. Updating final mask...")
            self.update_mask() 
            
    def handle_mouse_leave(self, event):
        self.last_hover_pos = None
        if not self.is_drawing:
            self.draw_prompts_overlay() 

    def update_sliders(self, value):
        """Called when the threshold slider moves."""
        self.threshold = self.threshold_slider.value() / 100.0
        self.threshold_value_label.setText(f"{self.threshold:.2f}")

        # --- Contrast update removed ---

        self.draw_prompts_overlay() 
        self.update_mask()          

    def update_display_full(self):
        self.draw_prompts_overlay()
        self.update_mask()

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.images)
        self.load_image(self.current_index)
        self.segmenter.set_image(self.image_rgb, target_res=self.max_res)
        self.app_prompts = []
        self.segmenter.clear_prompts()
        self.draw_prompts_overlay() 

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.images)
        self.load_image(self.current_index)
        self.segmenter.set_image(self.image_rgb, target_res=self.max_res)
        self.app_prompts = []
        self.segmenter.clear_prompts()
        self.draw_prompts_overlay() 

    def undo_prompt(self):
        if self.app_prompts:
            self.app_prompts.pop()
        self.segmenter.undo_prompt()
        self.update_display_full()

    def clear_prompts(self):
        self.app_prompts = []
        self.segmenter.clear_prompts()
        self.draw_prompts_overlay() 

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
    parser.add_argument(
        "--upsampler",
        type=str,
        default="bilinear",
        choices=["anyup", "bilinear"],
        help="The method to use for upsampling DINO features. 'anyup' is high quality, 'bilinear' is fast."
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="faiss",
        choices=["faiss", "torch"],
        help="Segmentation method. 'faiss' is constant-time, 'torch' is brute-force."
    )
    parser.add_argument(
        "--resize-method",
        type=str,
        default="pad",
        choices=["pad", "crop"],
        help="Resize method. 'pad' adds padding to make image divisible by patch size, 'crop' center crops."
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    try:
        window = InteractiveSegmenterApp(
            args.folder, 
            max_res=args.resolution, 
            upsampler=args.upsampler,
            segmenter=args.segmenter,
            resize_method=args.resize_method
        )
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
