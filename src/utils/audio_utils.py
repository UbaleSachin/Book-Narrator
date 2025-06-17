import os
import logging
from pathlib import Path
from gtts import gTTS
from tempfile import NamedTemporaryFile
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class AudioNarrator:
    def __init__(self, output_dir: str = "output/audio"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure retries for HTTP requests
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def narrate(self, text: str, filepath: str = None, max_retries: int = 3) -> str:
        """Convert text to speech and save as audio file."""
        if not text:
            logger.error("No text provided for narration")
            return False

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Try with temporary file first
        with NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
            
            for attempt in range(max_retries):
                try:
                    tts = gTTS(text=text, lang='en', slow=False)
                    tts.save(temp_path)
                    
                    # Verify the file was created and is valid
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        # Move temp file to final destination
                        try:
                            os.replace(temp_path, filepath)
                            logger.info(f"Successfully created audio file: {filepath}")
                            return True
                        except OSError as e:
                            logger.error(f"Failed to move audio file: {e}")
                            return False
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error("All narration attempts failed")
                        return False
                    continue
                
            return False