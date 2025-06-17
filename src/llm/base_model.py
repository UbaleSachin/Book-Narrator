import os
import json
import time
import base64
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO
from typing import Dict, Optional, List
import logging
from dotenv import load_dotenv
from src.utils.audio_utils import AudioNarrator

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

class ImageDescriber:
    """Class to handle image description using API calls (Groq, OpenAI, Gemini, etc.)"""
    
    def __init__(self, audio_folder=None):
        """Initialize the API-based image describer."""
        self.audio_folder = audio_folder
        
        # API configurations
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Available API providers and models
        self.api_providers = {
            'groq': {
                'base_url': 'https://api.groq.com/openai/v1',
                'models': [
                    'meta-llama/llama-4-scout-17b-16e-instruct'
                ],
                'api_key': self.groq_api_key
            },
            'openai': {
                'base_url': 'https://api.openai.com/v1',
                'models': [
                    'gpt-4o',           # Best vision model
                    'gpt-4o-mini',      # Faster and cheaper
                    'gpt-4-vision-preview'  # Fallback
                ],
                'api_key': self.openai_api_key
            },
            'gemini': {
                'base_url': 'https://generativelanguage.googleapis.com/v1beta',
                'models': [
                    'gemini-2.0-flash',     # Latest multimodal model
                    'gemini-1.5-pro',      # High-quality option
                    'gemini-1.5-flash'     # Faster option
                ],
                'api_key': self.gemini_api_key
            }
        }
        
        # Current provider and model
        self.current_provider = 'gemini'  # Default to Gemini (good balance of cost/quality)
        self.current_model_index = 0
        
        # Determine which provider to use based on available API keys
        if not self.gemini_api_key and self.groq_api_key:
            self.current_provider = 'groq'
        elif not self.gemini_api_key and not self.groq_api_key and self.openai_api_key:
            self.current_provider = 'openai'
        elif not any([self.gemini_api_key, self.groq_api_key, self.openai_api_key]):
            raise ValueError("At least one API key must be set: GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY")
        
        # Usage tracking
        self.usage_file = 'api_usage_tracking.json'
        self.daily_limits = {
            'groq': 14400,   # Groq free tier: 14,400 requests/day
            'openai': 1000,  # Conservative limit for OpenAI
            'gemini': 1500   # Gemini free tier: 1,500 requests/day
        }
        self.usage_data = self._load_usage_data()
        
        # Initialize audio narrator
        self.narrator = AudioNarrator()
        
        print(f"Initialized with provider: {self.current_provider}")
        print(f"Current model: {self.get_current_model()}")

    def _load_usage_data(self) -> Dict:
        """Load API usage data from file."""
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
        """Save API usage data to file."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            print(f"Error saving usage data: {e}")

    def _get_today_key(self) -> str:
        """Get today's date as a string key."""
        return datetime.now().strftime('%Y-%m-%d')

    def _update_api_usage(self, provider: str, model: str, count: int = 1):
        """Update API usage for a provider and model."""
        today = self._get_today_key()
        
        if provider not in self.usage_data:
            self.usage_data[provider] = {}
        
        if model not in self.usage_data[provider]:
            self.usage_data[provider][model] = {}
        
        if today not in self.usage_data[provider][model]:
            self.usage_data[provider][model][today] = 0
        
        self.usage_data[provider][model][today] += count
        self._save_usage_data()

    def _get_today_usage(self, provider: str, model: str = None) -> int:
        """Get today's API usage for a provider (and optionally specific model)."""
        today = self._get_today_key()
        
        if model:
            return self.usage_data.get(provider, {}).get(model, {}).get(today, 0)
        else:
            # Get total usage for provider across all models
            total = 0
            provider_data = self.usage_data.get(provider, {})
            for model_data in provider_data.values():
                total += model_data.get(today, 0)
            return total

    def _can_use_provider(self, provider: str, count: int = 1) -> bool:
        """Check if a provider can be used without exceeding daily limit."""
        current_usage = self._get_today_usage(provider)
        daily_limit = self.daily_limits.get(provider, 1000)
        return (current_usage + count) <= daily_limit

    def get_current_model(self) -> str:
        """Get the current model being used."""
        provider_config = self.api_providers[self.current_provider]
        return provider_config['models'][self.current_model_index]

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string."""
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image_data = response.content
            else:
                with open(image_path, 'rb') as image_file:
                    image_data = image_file.read()
            
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return None

    def _get_image_format(self, image_path: str) -> str:
        """Get image format (jpeg, png, etc.)."""
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            return image.format.lower() if image.format else 'jpeg'
        except:
            return 'jpeg'  # Default fallback

    def _get_mime_type(self, image_format: str) -> str:
        """Get MIME type from image format."""
        mime_types = {
            'jpeg': 'image/jpeg',
            'jpg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'bmp': 'image/bmp'
        }
        return mime_types.get(image_format.lower(), 'image/jpeg')

    def _make_gemini_api_call(self, image_path: str, model: str) -> Optional[str]:
        """Make API call to Gemini for image description."""
        try:
            # Encode image
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                return None
            
            image_format = self._get_image_format(image_path)
            mime_type = self._get_mime_type(image_format)
            
            # Prepare the prompt
            prompt = """Please provide a detailed description of this image. Structure your response as follows:

🖼️ Scene Overview:
[Provide a clear, concise description of what you see in the image]

Please be descriptive but concise, focusing on the most important visual elements."""

            # Prepare request payload for Gemini
            url = f"{self.api_providers['gemini']['base_url']}/models/{model}:generateContent"
            params = {'key': self.gemini_api_key}
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Make API request
            response = requests.post(url, params=params, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']
                    if 'parts' in content and len(content['parts']) > 0:
                        description = content['parts'][0]['text']
                        
                        # Update usage tracking
                        self._update_api_usage('gemini', model, 1)
                        
                        return description
                else:
                    logger.error(f"Unexpected Gemini response format: {result}")
                    return None
            else:
                logger.error(f"Gemini API call failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error making Gemini API call: {str(e)}")
            return None

    def _make_api_call(self, image_path: str, provider: str, model: str) -> Optional[str]:
        """Make API call to describe image."""
        if provider == 'gemini':
            return self._make_gemini_api_call(image_path, model)
        
        try:
            provider_config = self.api_providers[provider]
            
            # Encode image
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                return None
            
            image_format = self._get_image_format(image_path)
            
            # Prepare the prompt
            prompt = """Please provide a detailed description of this image. Structure your response as follows:

🖼️ Scene Overview:
[Provide a clear, concise description of what you see in the image]

Please be descriptive but concise, focusing on the most important visual elements."""

            # Prepare request payload
            if provider == 'groq':
                headers = {
                    'Authorization': f'Bearer {provider_config["api_key"]}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'model': model,
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'text',
                                    'text': prompt
                                },
                                {
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': f'data:image/{image_format};base64,{image_base64}'
                                    }
                                }
                            ]
                        }
                    ],
                    'max_tokens': 1000,
                    'temperature': 0.7
                }
                
            elif provider == 'openai':
                headers = {
                    'Authorization': f'Bearer {provider_config["api_key"]}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'model': model,
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'text',
                                    'text': prompt
                                },
                                {
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': f'data:image/{image_format};base64,{image_base64}'
                                    }
                                }
                            ]
                        }
                    ],
                    'max_tokens': 1000
                }
            
            # Make API request
            url = f"{provider_config['base_url']}/chat/completions"
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                description = result['choices'][0]['message']['content']
                
                # Update usage tracking
                self._update_api_usage(provider, model, 1)
                
                return description
            else:
                logger.error(f"API call failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error making API call to {provider}: {str(e)}")
            return None

    def _switch_to_next_model(self) -> bool:
        """Switch to next available model or provider."""
        # Try next model in current provider
        provider_config = self.api_providers[self.current_provider]
        if self.current_model_index < len(provider_config['models']) - 1:
            self.current_model_index += 1
            return True
        
        # Try switching provider based on priority: gemini -> groq -> openai
        available_providers = []
        if self.gemini_api_key and self.current_provider != 'gemini':
            available_providers.append('gemini')
        if self.groq_api_key and self.current_provider != 'groq':
            available_providers.append('groq')
        if self.openai_api_key and self.current_provider != 'openai':
            available_providers.append('openai')
        
        if available_providers:
            self.current_provider = available_providers[0]
            self.current_model_index = 0
            return True
        
        return False

    def describe_image(self, image_path: str, narrate: bool = True, max_retries: int = 3) -> dict:
        """Generate description and audio narration for an image using API calls."""
        try:
            for attempt in range(max_retries):
                # Check if current provider has capacity
                if not self._can_use_provider(self.current_provider, 1):
                    print(f"Provider {self.current_provider} would exceed daily limit. Switching...")
                    if not self._switch_to_next_model():
                        return {
                            'description': "All API providers have reached their daily limits.",
                            'scene_overview': '',
                            'audio_path': None,
                            'error': 'Daily limit exceeded'
                        }
                
                current_model = self.get_current_model()
                print(f"Using {self.current_provider} - {current_model} (Attempt {attempt + 1})")
                
                # Make API call
                description = self._make_api_call(image_path, self.current_provider, current_model)
                
                if description:
                    # Extract scene overview for narration
                    scene_overview = ''
                    
                    # Try different patterns for scene overview
                    scene_patterns = [
                        '🖼️ Scene Overview:',
                        '🖼️ **Scene Overview:**',
                        'Scene Overview:',
                        '**Scene Overview:**'
                    ]
                    
                    found_pattern = None
                    for pattern in scene_patterns:
                        if pattern in description:
                            found_pattern = pattern
                            break
                    
                    if found_pattern:
                        lines = description.split('\n')
                        in_overview = False
                        overview_lines = []
                        
                        for line in lines:
                            line_stripped = line.strip()
                            
                            # Check if this line contains the pattern
                            if found_pattern in line:
                                in_overview = True
                                # Extract text after the pattern on the same line
                                after_pattern = line.split(found_pattern, 1)
                                if len(after_pattern) > 1 and after_pattern[1].strip():
                                    overview_lines.append(after_pattern[1].strip())
                                continue
                            
                            # Check if we've hit the next section
                            elif (line_stripped.startswith(('👥', '🎨', '📝', '🔍', '**👥', '**🎨', '**📝', '**🔍')) 
                                  and in_overview):
                                break
                            
                            # Add line if we're in the overview section
                            elif in_overview and line_stripped:
                                overview_lines.append(line_stripped)
                        
                        scene_overview = ' '.join(overview_lines)
                    
                    # Fallback: if no structured format found, use first few sentences
                    if not scene_overview and description:
                        sentences = description.replace('\n', ' ').split('.')
                        if len(sentences) >= 2:
                            scene_overview = '. '.join(sentences[:2]) + '.'
                        elif sentences[0]:
                            scene_overview = sentences[0] + '.'
                    
                    result = {
                        'description': description,
                        'scene_overview': scene_overview,
                        'audio_path': None,
                        'provider': self.current_provider,
                        'model': current_model
                    }
                    
                    # Generate audio narration if requested
                    if narrate:
                        if scene_overview:
                            print(f"Scene overview extracted: {scene_overview[:100]}...")
                            
                            # Add microseconds to timestamp for better uniqueness
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                            filename = f"narration_{timestamp}.mp3"
                            audio_path = os.path.join(self.audio_folder, filename)
                            
                            # Ensure directory exists
                            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                            
                            # Add retry logic for audio generation
                            max_audio_retries = 3
                            audio_success = False
                            
                            for audio_attempt in range(max_audio_retries):
                                try:
                                    print(f"Generating audio narration (attempt {audio_attempt + 1}/{max_audio_retries})...")
                                    
                                    # Add small delay to prevent resource conflicts
                                    if audio_attempt > 0:
                                        time.sleep(1)
                                    
                                    self.narrator.narrate(scene_overview, audio_path)
                                    
                                    # Verify file was created and has content
                                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                        result['audio_path'] = audio_path
                                        print(f"Audio narration saved to: {audio_path}")
                                        audio_success = True
                                        break
                                    else:
                                        print(f"Audio file not created or empty on attempt {audio_attempt + 1}")
                                        
                                except Exception as e:
                                    print(f"Audio generation attempt {audio_attempt + 1} failed: {e}")
                                    
                                    # Clean up failed file if it exists
                                    if os.path.exists(audio_path):
                                        try:
                                            os.remove(audio_path)
                                        except:
                                            pass
                            
                            if not audio_success:
                                print("Failed to generate audio narration after multiple attempts")
                                result['audio_error'] = 'Audio generation failed after retries'
                        else:
                            print("No scene overview found for audio narration")
                            print(f"Description preview: {description[:200]}...")
                            result['audio_error'] = 'No scene overview extracted'
                    
                    return result
                else:
                    # Try next model/provider
                    if attempt < max_retries - 1:
                        if not self._switch_to_next_model():
                            break
                    
            return {
                'description': "Failed to generate description after multiple attempts.",
                'scene_overview': '',
                'audio_path': None,
                'error': 'API calls failed'
            }
            
        except Exception as e:
            logger.error(f"Error in describe_image: {str(e)}")
            return {
                'description': f"Error: {str(e)}",
                'scene_overview': '',
                'audio_path': None,
                'error': str(e)
            }

    def get_usage_summary(self) -> Dict:
        """Get usage summary for all providers and models."""
        today = self._get_today_key()
        summary = {}
        
        for provider, config in self.api_providers.items():
            if not config['api_key']:
                continue
                
            provider_summary = {}
            total_usage = self._get_today_usage(provider)
            daily_limit = self.daily_limits.get(provider, 1000)
            
            provider_summary['total_usage'] = total_usage
            provider_summary['daily_limit'] = daily_limit
            provider_summary['remaining'] = daily_limit - total_usage
            provider_summary['usage_percentage'] = (total_usage / daily_limit) * 100
            provider_summary['status'] = 'Available' if total_usage < daily_limit else 'Exhausted'
            provider_summary['models'] = {}
            
            # Model-specific usage
            for model in config['models']:
                model_usage = self._get_today_usage(provider, model)
                provider_summary['models'][model] = {
                    'usage': model_usage,
                    'last_used': 'Today' if model_usage > 0 else 'Not used today'
                }
            
            summary[provider] = provider_summary
        
        return summary

    def switch_provider(self, provider: str, model_index: int = 0) -> bool:
        """Manually switch to a specific provider and model."""
        if provider in self.api_providers and self.api_providers[provider]['api_key']:
            self.current_provider = provider
            self.current_model_index = min(model_index, len(self.api_providers[provider]['models']) - 1)
            print(f"Switched to {provider} - {self.get_current_model()}")
            return True
        else:
            print(f"Provider {provider} not available or API key not set")
            return False

    def list_available_providers(self) -> Dict:
        """List all available providers and their models."""
        available = {}
        for provider, config in self.api_providers.items():
            if config['api_key']:
                available[provider] = {
                    'models': config['models'],
                    'status': 'Available' if self._can_use_provider(provider) else 'Daily limit reached'
                }
        return available


# Example usage
if __name__ == "__main__":
    # Test the API-based image describer
    test_image_path = "path_to_your_image_file.jpg"  # Replace with actual image path
    
    try:
        describer = ImageDescriber(audio_folder="./audio_output")
        
        # Show available providers
        print("Available Providers:")
        providers = describer.list_available_providers()
        for provider, info in providers.items():
            print(f"\n{provider.upper()}:")
            print(f"  Status: {info['status']}")
            print(f"  Models: {', '.join(info['models'])}")
        
        print("\n" + "="*50 + "\n")
        
        # Show usage summary
        print("Usage Summary:")
        usage = describer.get_usage_summary()
        for provider, stats in usage.items():
            print(f"\n{provider.upper()}:")
            print(f"  Total Usage: {stats['total_usage']}/{stats['daily_limit']} ({stats['usage_percentage']:.1f}%)")
            print(f"  Status: {stats['status']}")
        
        print("\n" + "="*50 + "\n")
        
        # Test image description
        if os.path.exists(test_image_path):
            print("Describing image...")
            result = describer.describe_image(test_image_path, narrate=True)
            
            if 'description' in result and not result.get('error'):
                print(f"Provider used: {result.get('provider', 'Unknown')}")
                print(f"Model used: {result.get('model', 'Unknown')}")
                print("\nImage Description:")
                print(result['description'])
                
                if result['audio_path']:
                    print(f"\nAudio Narration saved: {result['audio_path']}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"Test image not found: {test_image_path}")
            print("Please update the test_image_path variable with a valid image file path.")
            
    except Exception as e:
        print(f"Error initializing describer: {str(e)}")
        print("Make sure to set at least one API key in your .env file:")
        print("- GEMINI_API_KEY")
        print("- GROQ_API_KEY")
        print("- OPENAI_API_KEY")