# background_removal.py
import os
import io
import ssl
from PIL import Image
import numpy as np
import cv2

# SSL sertifika sorununu çöz
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    from rembg import remove, new_session
except ImportError as e:
    print(f"rembg import error: {e}")
    raise

class BackgroundRemover:
    def __init__(self, model_name='u2net'):
        """
        Initialize background remover
        model_name options: 'u2net', 'u2net_human_seg', 'u2netp', 'silueta'
        u2net_human_seg is specifically for people/clothing
        """
        try:
            self.session = new_session(model_name)
            print(f"rembg model '{model_name}' loaded successfully")
        except Exception as e:
            print(f"rembg model loading error: {e}")
            print("Falling back to basic model...")
            try:
                self.session = new_session('u2net')
                print("Fallback model 'u2net' loaded")
            except Exception as e2:
                print(f"Fallback model also failed: {e2}")
                self.session = None
        
    def remove_background(self, image_path, output_path=None):
        """Remove background from image"""
        try:
            # Load image
            with open(image_path, 'rb') as f:
                input_image = f.read()
            
            # Remove background
            output_image = remove(input_image, session=self.session)
            
            # Convert to PIL Image
            result = Image.open(io.BytesIO(output_image))
            
            if output_path:
                result.save(output_path)
                return output_path
            else:
                return result
                
        except Exception as e:
            print(f"Background removal error: {e}")
            # Return original image if background removal fails
            return Image.open(image_path)
    
    def remove_background_pil(self, pil_image):
        """Remove background from PIL Image"""
        if self.session is None:
            print("No rembg session available, returning original image")
            return pil_image
            
        try:
            # Convert PIL to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Remove background
            output_image = remove(img_byte_arr, session=self.session)
            
            # Convert back to PIL
            result = Image.open(io.BytesIO(output_image))
            return result
            
        except Exception as e:
            print(f"Background removal error: {e}")
            return pil_image

def create_white_background(image, background_color=(255, 255, 255)):
    """Create white background for transparent images"""
    if image.mode in ('RGBA', 'LA'):
        # Create white background
        background = Image.new('RGB', image.size, background_color)
        if image.mode == 'RGBA':
         background.paste(image, mask=image.split()[-1])  # Alpha channel var
        else:
          background.paste(image)  # Alpha channel yok, direkt yapıştır # Use alpha channel as mask
        return background
    return image.convert('RGB')

# Global model instance - bir kez yüklenir
_global_bg_remover = None

def preprocess_image_for_clothing(image_path, use_background_removal=True):
    """
    Preprocess image to focus on clothing
    """
    global _global_bg_remover
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        if use_background_removal:
            # Model'i bir kez yükle
            if _global_bg_remover is None:
                from config import BACKGROUND_REMOVAL
                model_name = BACKGROUND_REMOVAL.get('model', 'u2net')
                print(f"Loading background removal model: {model_name}")
                _global_bg_remover = BackgroundRemover(model_name)
            
            # Remove background
            no_bg_image = _global_bg_remover.remove_background_pil(image)
            
            # Add white background
            final_image = create_white_background(no_bg_image)
            
            return final_image
        
        return image
        
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return Image.open(image_path).convert("RGB")

def apply_clothing_mask(image):
    """
    Apply simple clothing detection mask using color segmentation
    This is a basic approach - for better results, use a trained segmentation model
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define ranges for skin color (to exclude from clothing)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Invert to get clothing mask
        clothing_mask = cv2.bitwise_not(skin_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to image
        masked_image = img_array.copy()
        masked_image[clothing_mask == 0] = [255, 255, 255]  # White background
        
        return Image.fromarray(masked_image)
        
    except Exception as e:
        print(f"Clothing mask error: {e}")
        return image
