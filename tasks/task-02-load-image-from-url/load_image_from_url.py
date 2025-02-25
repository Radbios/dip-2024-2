import argparse
import numpy as np
import cv2 as cv
import requests

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        
        flags = kwargs.get("flags", cv.IMREAD_COLOR)
        image = cv.imdecode(image_array, flags)
        
        if image is None:
            raise ValueError("Falha ao carregar a imagem.")

        return image

    except Exception as e:
        print(f"Erro ao carregar imagem: {e}")
        return None
    ### END CODE HERE ###
    
    return image

# parser = argparse.ArgumentParser(description="Carrega uma imagem a partir de uma URL.")
# parser.add_argument("--url", type=str, )
# parser.add_argument("--kwargs", type=int, default=cv.IMREAD_COLOR)

# args = parser.parse_args()

# load_image_from_url(args.url, flags=args.kwargs)

load_image_from_url()