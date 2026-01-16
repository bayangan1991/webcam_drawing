from functools import partial

import cv2
import numpy as np

from camera import VideoCamera
from utils import BBox, Zone


class Canvas:
    def __init__(self, camera: VideoCamera):
        self.camera = camera
        self.camera_flip = False
        self.canvas = None
        self.prev_green_pos = None  # Track previous green position for drawing lines
        self.active_pen_index = 2
        self.pen_colours = [
            ("White", (255, 255, 255)),
            ("Black", (2, 2, 2)),
            ("Red", (26, 17, 236)),
            ("Orange", (48, 99, 251)),
            ("Yellow", (47, 212, 255)),
            ("Green", (84, 172, 19)),
            ("Blue", (214, 157, 0)),
            ("Purple", (184, 73, 120)),
            ("Pink", (158, 96, 242)),
        ]
        self.pen_colour = self.pen_colours[self.active_pen_index][1]
        self.pen_size = 15
        self.zones = {
            "Clear": Zone(
                bbox=BBox(x1=0, y1=0, x2=150, y2=150),
                action=self.clear_canvas,
            ),
            "Eraser": Zone(
                bbox=BBox(x1=5, y1=-100, x2=95, y2=-5),
                action=partial(self.set_pen_colour, colour=(0, 0, 0)),
                colour=(200, 200, 200),
            ),
            "Flip": Zone(
                bbox=BBox(x1=-150, y1=0, x2=-5, y2=150),
                action=self.toggle_flip,
            ),
            "+": Zone(
                bbox=BBox(x1=-100, y1=-100, x2=-5, y2=-5),
                action=self.increase_pen_size,
            ),
            "-": Zone(
                bbox=BBox(x1=-200, y1=-100, x2=-105, y2=-5),
                action=self.decrease_pen_size,
            ),
            **{
                colour_name: Zone(
                    bbox=BBox(x1=(105, 100 * i), y1=-100, x2=(195, 100 * i), y2=-5),
                    action=partial(self.set_pen_colour_by_name, name=colour_name),
                    colour=colour,
                )
                for i, (colour_name, colour) in enumerate(self.pen_colours)
            },
        }
        self.ui = None

    def toggle_flip(self):
        self.camera_flip = not self.camera_flip

    def set_pen_size(self, size: int):
        self.pen_size = size

    def increase_pen_size(self):
        self.pen_size = min(50, self.pen_size + 5)

    def decrease_pen_size(self):
        self.pen_size = max(5, self.pen_size - 5)

    def set_pen_colour(self, colour: tuple):
        self.pen_colour = colour

    def set_pen_colour_by_name(self, name: str):
        for i, (colour_name, colour) in enumerate(self.pen_colours):
            if colour_name.lower() == name.lower():
                self.active_pen_index = i
                self.pen_colour = colour
                return

    def initialize_ui(self, frame):
        if frame is None or self.ui is not None:
            return
        size_x, size_y = frame.shape[1], frame.shape[0]
        self.ui = np.zeros_like(frame)
        for name, zone in self.zones.items():
            (x1, y1), (x2, y2) = zone.bbox.normalised(size_x, size_y).as_tuple()
            cv2.rectangle(self.ui, (x1, y1), (x2, y2), zone.colour, 3)

            # Get text dimensions and centre it in the rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                name, font, font_scale, thickness
            )

            # Calculate centre position
            center_x = x1 + (x2 - x1 - text_width) // 2
            center_y = y1 + (y2 - y1 + text_height) // 2

            cv2.putText(
                self.ui,
                name,
                (center_x, center_y),
                font,
                font_scale,
                zone.colour,
                thickness,
            )

    def initialize_canvas(self, frame):
        if frame is None or self.canvas is not None:
            return
        self.canvas = np.zeros_like(frame)

    def clear_canvas(self):
        """Clear the drawing canvas."""
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
        self.prev_green_pos = None

    def draw_circle(
        self,
        frame,
        x,
        y,
        radius=10,
        color=(0, 255, 0),
        thickness=2,
        fill=None,
    ) -> bytes:
        if frame is None:
            return None

        # Draw circle on the frame
        if fill is not None:
            cv2.circle(frame, (x, y), radius, fill, -1)
        cv2.circle(frame, (x, y), radius, color, thickness)

        return frame

    def get_zone(self, x, y):
        size_x, size_y = self.canvas.shape[1], self.canvas.shape[0]
        for zone in self.zones.values():
            if zone.bbox.inside(x, y, size_x, size_y):
                return zone
        return None

    def blit(self, base_frame, layer, alpha=1.0):
        """
        Apply a layer on top of base frame, replacing pixels where layer is non-zero.
        """
        mask = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        mask_3channel = mask[:, :, np.newaxis] > 0

        if alpha >= 1.0:
            # Full opacity - direct replacement
            return np.where(mask_3channel, layer, base_frame)
        # Blend with transparency
        blended = cv2.addWeighted(base_frame, 1 - alpha, layer, alpha, 0)
        return np.where(mask_3channel, blended, base_frame)

    def get_frame(self):
        frame, colour_coords = self.camera.get_frame(flip=self.camera_flip)
        if frame is None:
            return None

        if (debug_mask := colour_coords.get("debug_mask")) is not None:
            return debug_mask

        overlay = np.zeros_like(frame)

        # Initialise canvas and ui with same dimensions as frame
        self.initialize_canvas(frame)
        self.initialize_ui(frame)

        cursor = None

        # Draw on canvas when green is detected
        if colour_coords["green"]:
            x, y = colour_coords["green"]
            cursor = ("green", x, y, (0, 255, 0))
        else:
            # Reset previous position when green is not detected
            self.prev_green_pos = None
            # Draw red circle if red is detected
            if colour_coords["red"]:
                x, y = colour_coords["red"]
                cursor = ("red", x, y, (0, 0, 255))

        if cursor:
            mode, x, y, colour = cursor
            zone = self.get_zone(x, y) if cursor else None
            overlay = self.draw_circle(
                overlay,
                x,
                y,
                radius=self.pen_size,
                color=colour,
                thickness=3,
                fill=self.pen_colour,
            )

            if not zone and mode == "green":
                # Draw on canvas
                if self.prev_green_pos is not None:
                    # Draw line from previous position to current position

                    cv2.line(
                        self.canvas,
                        self.prev_green_pos,
                        (x, y),
                        self.pen_colour,
                        2 * self.pen_size,
                    )
                else:
                    # Draw initial point
                    cv2.circle(
                        self.canvas,
                        (x, y),
                        self.pen_size,
                        self.pen_colour,
                        -1,
                    )

                self.prev_green_pos = (x, y)
            elif zone and mode == "green":
                if self.prev_green_pos is None:
                    zone.action()
                    self.prev_green_pos = (x, y)

        # Draw an arrow over the active pen colour in the UI
        active_i = 0 if self.pen_colour == (0, 0, 0) else self.active_pen_index + 1

        cv2.arrowedLine(
            overlay,
            (50 + (100 * active_i), overlay.shape[0] - 145),
            (50 + (100 * active_i), overlay.shape[0] - 110),
            (255, 255, 255),
            5,
            tipLength=0.5,
        )

        cv2.putText(
            overlay,
            f"Pen Size: {self.pen_size}",
            (overlay.shape[1] - 200, overlay.shape[0] - 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        frame = self.blit(frame, self.canvas, alpha=0.7)
        frame = self.blit(frame, self.ui)
        frame = self.blit(frame, overlay)

        if frame is None:
            return None

        return frame

    def gen(self):
        while True:
            frame = self.get_frame()
            if frame is None:
                continue
            encoded_frame = self.encode_frame(frame)

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + encoded_frame
                + b"\r\n\r\n"
            )

    def encode_frame(self, frame) -> bytes:
        if frame is None:
            return None
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            return None
        return jpeg.tobytes()
