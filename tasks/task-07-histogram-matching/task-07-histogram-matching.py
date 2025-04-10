# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    matched_img = match_histograms(source_img, reference_img, channel_axis=-1)
    return matched_img.astype(np.uint8)

# def plot_all_histograms(src, ref, matched):
#     titles = ["Original", "Referência", "Ajustada"]
#     images = [src, ref, matched]
#     colors = ['r', 'g', 'b']

#     plt.figure(figsize=(15, 4))

#     for idx, img in enumerate(images):
#         plt.subplot(1, 3, idx + 1)
#         for i, color in enumerate(colors):
#             hist, bins = np.histogram(img[:, :, i], bins=256, range=(0, 256))
#             plt.plot(bins[:-1], hist, color=color, label=color.upper())
#         plt.title(f"Histograma - {titles[idx]}")
#         plt.xlabel("Intensidade")
#         plt.ylabel("Frequência")
#         plt.grid(True)
#         plt.legend()

#     plt.tight_layout()
#     plt.show()

# source_bgr = cv2.imread('source.jpg')
# reference_bgr = cv2.imread('reference.jpg')

# source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
# reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)

# matched_rgb = match_histograms_rgb(source_rgb, reference_rgb)

# cv2.imwrite('output.jpg', matched_rgb)

# plot_all_histograms(source_rgb, reference_rgb, matched_rgb)
