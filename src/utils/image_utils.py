from PIL import Image
import base64
import io
from typing import Tuple, Optional

class ImageProcessor:
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """Validate if file is a valid image."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def prepare_image(image_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> Optional[str]:
        """Prepare image for LLM processing."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if needed
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size)
                
                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            print(f"Error preparing image: {str(e)}")
            return None