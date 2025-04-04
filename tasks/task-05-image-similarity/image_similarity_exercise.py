# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def mse(i1: np.ndarray, i2: np.ndarray) -> float:
    return np.mean((i1 - i2) ** 2)

def psnr(i1: np.ndarray, i2: np.ndarray) -> float:
    mse_value = mse(i1, i2)
    if mse_value == 0:
        return float('inf')
    max_pixel = 1.0
    return 10 * np.log10((max_pixel ** 2)/mse_value)

def ssim(i1: np.ndarray, i2: np.ndarray) -> float:
    mu1, mu2 = np.mean(i1), np.mean(i2)
    sigma1, sigma2 = np.var(i1), np.var(i2)
    sigma12 = np.mean((i1 - mu1) * (i2 - mu2))
    
    C1, C2 = 1e-8, 1e-8
    
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    div = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    
    return num/div

def npcc(i1: np.ndarray, i2: np.ndarray) -> float:
    i1_mean, i2_mean = np.mean(i1), np.mean(i2)
    num = np.sum((i1 - i1_mean) * (i2 - i2_mean))
    div = np.sqrt(np.sum((i1 - i1_mean) ** 2) * np.sum((i2 - i2_mean) ** 2))
    
    return num/div if div != 0 else 0

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    return {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }