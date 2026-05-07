import cv2
import numpy as np
from std_msgs.msg import MultiArrayDimension


NUB_DETECTION_FIELD_COUNT = 7


def make_nub_detection_layout(detection_count):
    return [
        MultiArrayDimension(
            label="detection",
            size=detection_count,
            stride=detection_count * NUB_DETECTION_FIELD_COUNT,
        ),
        MultiArrayDimension(
            label="field",
            size=NUB_DETECTION_FIELD_COUNT,
            stride=NUB_DETECTION_FIELD_COUNT,
        ),
    ]


def pack_nub_detection(point, u, v, area, circularity):
    return [
        float(point.x),
        float(point.y),
        float(point.z),
        float(u),
        float(v),
        float(area),
        float(circularity),
    ]


def iter_nub_detection_rows(values):
    for index in range(0, len(values), NUB_DETECTION_FIELD_COUNT):
        fields = values[index : index + NUB_DETECTION_FIELD_COUNT]
        if len(fields) == NUB_DETECTION_FIELD_COUNT:
            yield fields


class GreenNubDetector:
    def __init__(self, settings):
        self.settings = settings

    def detect(self, cv_image):
        return self.detect_from_mask(self.find_mask(cv_image))

    def detect_from_mask(self, green_mask):
        green_contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []

        min_area = float(self.settings["green_min_area"])
        max_area = float(self.settings["green_max_area"])
        min_circularity = float(self.settings["green_min_circularity"])

        for contour in green_contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < min_circularity:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            candidates.append(
                {
                    "u": int(moments["m10"] / moments["m00"]),
                    "v": int(moments["m01"] / moments["m00"]),
                    "area": float(area),
                    "circularity": float(circularity),
                    "contour": contour,
                }
            )

        return candidates

    def find_mask(self, cv_image):
        blur_kernel_size = self._odd_kernel_size(
            int(self.settings["green_blur_kernel_size"])
        )
        morph_kernel_size = self._odd_kernel_size(
            int(self.settings["green_morph_kernel_size"])
        )

        working_image = cv_image
        if blur_kernel_size > 1:
            working_image = cv2.GaussianBlur(
                working_image, (blur_kernel_size, blur_kernel_size), 0
            )

        hsv_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array(
            [
                int(self.settings["green_h_min"]),
                int(self.settings["green_s_min"]),
                int(self.settings["green_v_min"]),
            ],
            dtype=np.uint8,
        )
        upper_green = np.array(
            [
                int(self.settings["green_h_max"]),
                int(self.settings["green_s_max"]),
                int(self.settings["green_v_max"]),
            ],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv_image, lower_green, upper_green)

        if morph_kernel_size > 1:
            kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _odd_kernel_size(self, kernel_size):
        if kernel_size < 1:
            return 1
        if kernel_size % 2 == 0:
            return kernel_size + 1
        return kernel_size
