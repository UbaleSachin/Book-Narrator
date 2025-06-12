from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from typing import Dict, Optional, List
import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
import warnings
import logging
from src.utils.audio_utils import AudioNarrator

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

load_dotenv()

class ImageDescriber:
    """Class to handle image description using Hugging Face models with model fallback."""
    
    def __init__(self, audio_folder=None):
        """Initialize the Hugging Face models and tracking."""
        # Available Hugging Face vision models
        self.models = [
            "Salesforce/blip-image-captioning-large",
            "Salesforce/blip-image-captioning-base",
            "microsoft/git-base-coco",
            "microsoft/git-large-coco",
            "nlpconnect/vit-gpt2-image-captioning"
        ]
        
        # Filter models based on environment variable if specified
        env_model = os.getenv('MODEL_NAME')
        if env_model:
            self.models = [env_model] + [m for m in self.models if m != env_model]
        
        self.hf_api_key = os.getenv('HuggingFace_API_KEY')
        self.current_model_index = 0
        self.current_model = None
        self.current_processor = None
        self.audio_folder = audio_folder
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Token usage tracking file
        self.usage_file = 'model_usage_tracking.json'
        self.daily_limit = 1000  # Daily inference limit per model
        
        # Load usage tracking
        self.usage_data = self._load_usage_data()
        
        # Try to load the first available model
        self._load_current_model()
        
        # Initialize audio narrator
        self.narrator = AudioNarrator()

    def _load_current_model(self) -> bool:
        """Load the current model and processor."""
        try:
            model_name = self.models[self.current_model_index]
            print(f"Loading model: {model_name}")
            
            # Load model based on type
            if "blip" in model_name.lower():
                self.current_processor = BlipProcessor.from_pretrained(model_name)
                self.current_model = BlipForConditionalGeneration.from_pretrained(model_name)
            elif "git" in model_name.lower():
                # For GIT models, use pipeline
                self.current_pipeline = pipeline(
                    "image-to-text", 
                    model=model_name,
                    device=0 if self.device == "cuda" else -1
                )
                self.current_processor = None
                self.current_model = None
            elif "vit-gpt2" in model_name.lower():
                # For ViT-GPT2 models, use pipeline
                self.current_pipeline = pipeline(
                    "image-to-text", 
                    model=model_name,
                    device=0 if self.device == "cuda" else -1
                )
                self.current_processor = None
                self.current_model = None
            else:
                # Generic approach using pipeline
                self.current_pipeline = pipeline(
                    "image-to-text", 
                    model=model_name,
                    device=0 if self.device == "cuda" else -1
                )
                self.current_processor = None
                self.current_model = None
            
            # Move model to device if loaded directly
            if self.current_model:
                self.current_model.to(self.device)
            
            print(f"Successfully loaded: {model_name}")
            return True
            
        except Exception as e:
            print(f"Failed to load model {self.models[self.current_model_index]}: {str(e)}")
            return False

    def _load_usage_data(self) -> Dict:
        """Load inference usage data from file."""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error loading usage data: {e}")
            return {}

    def _save_usage_data(self):
        """Save inference usage data to file."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            print(f"Error saving usage data: {e}")

    def _get_today_key(self) -> str:
        """Get today's date as a string key."""
        return datetime.now().strftime('%Y-%m-%d')

    def _update_inference_usage(self, model_name: str, count: int = 1):
        """Update inference usage for a model."""
        today = self._get_today_key()
        
        if model_name not in self.usage_data:
            self.usage_data[model_name] = {}
        
        if today not in self.usage_data[model_name]:
            self.usage_data[model_name][today] = 0
        
        self.usage_data[model_name][today] += count
        self._save_usage_data()

    def _get_today_usage(self, model_name: str) -> int:
        """Get today's inference usage for a model."""
        today = self._get_today_key()
        return self.usage_data.get(model_name, {}).get(today, 0)

    def _can_use_model(self, model_name: str, count: int = 1) -> bool:
        """Check if a model can be used without exceeding daily limit."""
        current_usage = self._get_today_usage(model_name)
        return (current_usage + count) <= self.daily_limit

    def _switch_to_next_model(self) -> bool:
        """Switch to the next available model. Returns True if successful, False if no models available."""
        original_index = self.current_model_index
        
        while True:
            self.current_model_index = (self.current_model_index + 1) % len(self.models)
            
            # If we've cycled through all models, return False
            if self.current_model_index == original_index:
                return False
            
            current_model = self.models[self.current_model_index]
            
            # Check if this model has capacity
            if self._can_use_model(current_model, 1):
                if self._load_current_model():
                    print(f"Switched to model: {current_model}")
                    return True
                else:
                    continue  # Try next model if loading failed
        
        return False

    def _handle_model_error(self, error_message: str) -> bool:
        """Handle model errors by switching models."""
        print(f"Error with model {self.models[self.current_model_index]}: {error_message}")
        
        # Try to switch to next model
        return self._switch_to_next_model()

    def get_current_model_info(self) -> Dict:
        """Get information about current model usage."""
        current_model = self.models[self.current_model_index]
        today_usage = self._get_today_usage(current_model)
        remaining = self.daily_limit - today_usage
        
        return {
            'current_model': current_model,
            'today_usage': today_usage,
            'remaining_inferences': remaining,
            'usage_percentage': (today_usage / self.daily_limit) * 100,
            'device': self.device
        }

    def _generate_caption_with_blip(self, image: Image.Image) -> str:
        """Generate caption using BLIP model."""
        inputs = self.current_processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.current_model.generate(**inputs, max_length=150, num_beams=5)
        
        caption = self.current_processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption

    def _generate_caption_with_pipeline(self, image: Image.Image) -> str:
        """Generate caption using pipeline."""
        result = self.current_pipeline(image)
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', 'No description generated')
        return 'No description generated'

    def _format_description(self, caption: str, image_path: str) -> str:
        """Format the basic caption into a more detailed description."""
        # Get image metadata
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                format_type = img.format
        except:
            width = height = mode = format_type = "Unknown"
        
        # Create formatted description
        formatted_desc = f"""🖼️ Scene Overview:
{caption}

👥 Subjects:
{self._extract_subjects(caption)}

🎨 Visual Details:
- Colors: {self._analyze_colors(caption)}
- Lighting: {self._analyze_lighting(caption)}
- Style: Photographic image

📝 Context & Setting:
{self._extract_context(caption)}

🔍 Notable Details:
- Image dimensions: {width}x{height} pixels
- Color mode: {mode}
- Format: {format_type}
- AI-generated description based on visual analysis

📊 Technical Information:
- Model used: {self.models[self.current_model_index]}
- Processing device: {self.device}
- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return formatted_desc

    def _extract_subjects(self, caption: str) -> str:
        """Extract subject information from caption."""
        subjects = []
        common_subjects = ['person', 'man', 'woman', 'child', 'people', 'dog', 'cat', 'car', 'building', 'tree', 'flower']
        
        for subject in common_subjects:
            if subject in caption.lower():
                subjects.append(subject)
        
        if subjects:
            return f"Detected subjects: {', '.join(subjects)}"
        else:
            return "Various objects and elements as described in the scene overview"

    def _analyze_colors(self, caption: str) -> str:
        """Analyze color information from caption."""
        colors = []
        color_words = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'grey']
        
        for color in color_words:
            if color in caption.lower():
                colors.append(color)
        
        if colors:
            return f"Mentioned colors: {', '.join(colors)}"
        else:
            return "Color information not explicitly detected in description"

    def _analyze_lighting(self, caption: str) -> str:
        """Analyze lighting information from caption."""
        lighting_words = ['bright', 'dark', 'sunny', 'cloudy', 'shadow', 'light', 'illuminated', 'dim']
        
        for word in lighting_words:
            if word in caption.lower():
                return f"Lighting appears to be {word}"
        
        return "Natural lighting conditions"

    def _extract_context(self, caption: str) -> str:
        """Extract context information from caption."""
        context_words = ['indoor', 'outdoor', 'street', 'park', 'building', 'room', 'kitchen', 'garden', 'beach', 'mountain']
        
        for context in context_words:
            if context in caption.lower():
                return f"Setting appears to be {context}-related"
        
        return "Context inferred from the scene elements described above"

    def describe_image(self, image_path: str, narrate: bool = True) -> dict:
        """Generate description and audio narration for an image."""
        try:
            # Generate description
            description = self._generate_description(image_path)
            
            # Extract only the scene overview for narration
            scene_overview = ''
            if '🖼️ Scene Overview:' in description:
                scene_overview = description.split('👥')[0].strip()
                scene_overview = scene_overview.replace('🖼️ Scene Overview:', '').strip()
            
            result = {
                'description': description,  # Keep full description for display
                'scene_overview': scene_overview,  # Add scene overview for narration
                'audio_path': None
            }
            
            if narrate and scene_overview:
                # Create unique filename for audio
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"narration_{timestamp}.mp3"
                audio_path = os.path.join(self.audio_folder, filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                
                # Generate audio only for scene overview
                self.narrator.narrate(scene_overview, audio_path)
                result['audio_path'] = audio_path
                
            return result
            
        except Exception as e:
            logger.error(f"Error in describe_image: {str(e)}")
            return None

    def _generate_description(self, image_path: str, max_retries: int = 3) -> Optional[str]:
        """
        Generate description for an image using available models with retries.
        
        Args:
            image_path: Path to the image file or URL
            max_retries: Maximum number of retry attempts across models
            
        Returns:
            Formatted string containing structured description of the image
        """
        try:
            # Load image
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    return None
                image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            for attempt in range(max_retries):
                try:
                    current_model_name = self.models[self.current_model_index]
                    
                    # Check if current model has enough quota
                    if not self._can_use_model(current_model_name, 1):
                        print(f"Model {current_model_name} would exceed daily limit. Switching...")
                        if not self._switch_to_next_model():
                            print("All models have reached their daily limits!")
                            return None
                        current_model_name = self.models[self.current_model_index]
                    
                    print(f"Using model: {current_model_name} (Attempt {attempt + 1})")
                    
                    # Generate caption based on model type
                    if self.current_model and self.current_processor:
                        caption = self._generate_caption_with_blip(image)
                    else:
                        caption = self._generate_caption_with_pipeline(image)
                    
                    # Update inference usage
                    self._update_inference_usage(current_model_name, 1)
                    
                    # Format and return description
                    formatted_description = self._format_description(caption, image_path)
                    return formatted_description
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"Error with model {current_model_name}: {error_message}")
                    
                    # Try next model on error
                    if attempt < max_retries - 1:
                        if self._switch_to_next_model():
                            continue
                        else:
                            print("No more models available to try!")
                            return None
                    else:
                        print(f"Failed after {max_retries} attempts")
                        return None
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

    def get_usage_summary(self) -> Dict:
        """Get usage summary for all models."""
        today = self._get_today_key()
        summary = {}
        
        for model in self.models:
            usage = self._get_today_usage(model)
            remaining = self.daily_limit - usage
            summary[model] = {
                'today_usage': usage,
                'remaining_inferences': remaining,
                'usage_percentage': (usage / self.daily_limit) * 100,
                'status': 'Available' if remaining > 0 else 'Exhausted'
            }
        
        return summary

    def reset_daily_usage(self, model_name: Optional[str] = None):
        """Reset daily usage for a specific model or all models."""
        today = self._get_today_key()
        
        if model_name:
            if model_name in self.usage_data and today in self.usage_data[model_name]:
                del self.usage_data[model_name][today]
                print(f"Reset usage for {model_name}")
        else:
            for model in self.models:
                if model in self.usage_data and today in self.usage_data[model]:
                    del self.usage_data[model][today]
            print("Reset usage for all models")
        
        self._save_usage_data()

    def list_available_models(self) -> List[str]:
        """List all available models."""
        return self.models.copy()

    def switch_model(self, model_name: str) -> bool:
        """Manually switch to a specific model."""
        if model_name in self.models:
            self.current_model_index = self.models.index(model_name)
            return self._load_current_model()
        else:
            print(f"Model {model_name} not found in available models")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced image describer
    test_image_path = "path_to_your_image_file.jpg"  # Replace with actual image path
    
    describer = ImageDescriber()
    
    # Show available models
    print("Available Models:")
    for i, model in enumerate(describer.list_available_models()):
        print(f"{i+1}. {model}")
    print("\n" + "="*50 + "\n")
    
    # Show current model info
    print("Current Model Info:")
    model_info = describer.get_current_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")
    print("\n" + "="*50 + "\n")
    
    # Test image description
    if os.path.exists(test_image_path):
        print("Describing image...")
        result = describer.describe_image(test_image_path)
        
        if result:
            print("Image Description:")
            print(result['description'])
            
            if result['audio_path']:
                print(f"Audio Narration: {result['audio_path']}")
            
            print("\n" + "="*50 + "\n")
            
            # Show usage summary after processing
            print("Usage Summary:")
            usage_summary = describer.get_usage_summary()
            for model, stats in usage_summary.items():
                print(f"{model}:")
                print(f"  Usage: {stats['today_usage']} inferences ({stats['usage_percentage']:.1f}%)")
                print(f"  Remaining: {stats['remaining_inferences']} inferences")
                print(f"  Status: {stats['status']}")
                print()
        else:
            print("Failed to describe the image.")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please update the test_image_path variable with a valid image file path.")