# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np
import matplotlib.pyplot as plt

def translate(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    h, w = img.shape
    translated = np.zeros((h, w))
    translated[shift_y:, shift_x:] = img[:h-shift_y, :w-shift_x]
    return translated

def rotate_90_clockwise(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=-1)

def stretch(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape
    new_w = int(w * factor)
    stretched = np.zeros((h, new_w))

    for i in range(new_w):
        stretched[:, i] = img[:, int(i / factor)]

    return stretched

def mirror(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1]

def barrel_distortion(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    distorted = np.zeros_like(img)

    center_x, center_y = w // 2, h // 2
    for y in range(h):
        for x in range(w):
            dx, dy = x - center_x, y - center_y
            r = np.sqrt(dx**2 + dy**2)
            factor = 1 + 0.0005 * (r**2)
            new_x = int(center_x + dx * factor)
            new_y = int(center_y + dy * factor)

            if 0 <= new_x < w and 0 <= new_y < h:
                distorted[y, x] = img[new_y, new_x]

    return distorted

def apply_geometric_transformations(img: np.ndarray) -> dict:
     return {
        "translated": translate(img, shift_x=10, shift_y=10),
        "rotated": rotate_90_clockwise(img),
        "stretched": stretch(img, factor=1.5),
        "mirrored": mirror(img),
        "distorted": barrel_distortion(img)
    }
