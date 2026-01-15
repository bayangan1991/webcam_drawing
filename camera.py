import cv2
import numpy as np


class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_frame(self, *, flip: bool = False):
        success, frame = self.cap.read()
        if not success:
            return None
        # Mirror the frame horizontally
        if flip:
            frame = cv2.flip(frame, 1)
        return frame, self.detect_coloured_areas(frame)

    def detect_coloured_areas(self, frame) -> dict[str, tuple[int, int] | None]:
        results = {"red": None, "green": None}
        if frame is None:
            return results

        # Convert BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Higher saturation and value for vibrant colours only
        min_saturation = 90

        min_green_value = 150

        # Detect GREEN
        lower_green = np.array([40, min_saturation, min_green_value])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        contours_green, _ = cv2.findContours(
            mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours_green:
            largest_contour = max(contours_green, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                results["green"] = (cx, cy)

        if not results["green"]:
            min_red_value = 220
            # Detect RED
            lower_red1 = np.array([0, min_saturation, min_red_value])
            upper_red1 = np.array([30, 255, 255])
            lower_red2 = np.array([170, min_saturation, min_red_value])
            upper_red2 = np.array([180, 255, 255])

            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            contours_red, _ = cv2.findContours(
                mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours_red:
                largest_contour = max(contours_red, key=cv2.contourArea)
                moments = cv2.moments(largest_contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    results["red"] = (cx, cy)

        return results
