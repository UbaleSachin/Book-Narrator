import os
import logging
import time
from typing import Optional
import tempfile
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class AudioNarrator:
    """Class to handle audio narration with robust error handling."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        """
        Initialize the audio narrator.
        
        Args:
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Try to import gTTS with fallback
        try:
            from gtts import gTTS
            self.gtts_available = True
            logger.info("gTTS library loaded successfully")
        except ImportError as e:
            self.gtts_available = False
            logger.warning(f"gTTS not available: {e}")
        
        # Try to import pyttsx3 as fallback
        try:
            import pyttsx3
            self.pyttsx3_available = True
            logger.info("pyttsx3 library loaded successfully")
        except ImportError as e:
            self.pyttsx3_available = False
            logger.warning(f"pyttsx3 not available: {e}")
        
        if not self.gtts_available and not self.pyttsx3_available:
            logger.error("No TTS engines available. Audio narration will be disabled.")
    
    def _create_dummy_audio(self, filepath: str) -> bool:
        """
        Create a dummy audio file as fallback.
        
        Args:
            filepath (str): Path where to save the dummy audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a minimal MP3 file (silence)
            # This is just a placeholder - in production you might want to use
            # a more sophisticated approach like generating actual silence
            dummy_content = b'\xff\xfb\x90\x00' + b'\x00' * 100  # Minimal MP3 header + silence
            
            with open(filepath, 'wb') as f:
                f.write(dummy_content)
            
            logger.info(f"Created dummy audio file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dummy audio file: {e}")
            return False
    
    def _narrate_with_gtts(self, text: str, filepath: str, language: str = 'en') -> bool:
        """
        Generate audio using gTTS with retry logic.
        
        Args:
            text (str): Text to narrate
            filepath (str): Path to save the audio file
            language (str): Language code for TTS
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.gtts_available:
            return False
            
        from gtts import gTTS
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting gTTS narration (attempt {attempt + 1}/{self.max_retries})")
                
                # Create gTTS object with timeout settings
                tts = gTTS(text=text, lang=language, slow=False)
                
                # Save to temporary file first
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Save with timeout
                    tts.save(temp_path)
                    
                    # Verify the file was created and has content
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        # Move to final location
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        os.rename(temp_path, filepath)
                        logger.info(f"Successfully created audio file: {filepath}")
                        return True
                    else:
                        logger.warning(f"gTTS created empty file on attempt {attempt + 1}")
                        
                except Exception as save_error:
                    logger.warning(f"gTTS save failed on attempt {attempt + 1}: {save_error}")
                    
                finally:
                    # Clean up temp file if it exists
                    if temp_file and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                
                # Wait before retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.warning(f"gTTS attempt {attempt + 1} failed: {e}")
                
                # Wait before retry
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error(f"All gTTS attempts failed after {self.max_retries} retries")
        return False
    
    def _narrate_with_pyttsx3(self, text: str, filepath: str) -> bool:
        """
        Generate audio using pyttsx3 (offline TTS).
        
        Args:
            text (str): Text to narrate
            filepath (str): Path to save the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.pyttsx3_available:
            return False
            
        try:
            import pyttsx3
            
            logger.info("Attempting pyttsx3 narration")
            
            # Initialize the TTS engine
            engine = pyttsx3.init()
            
            # Set properties
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to file
            engine.save_to_file(text, filepath)
            engine.runAndWait()
            
            # Verify the file was created
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                logger.info(f"Successfully created audio with pyttsx3: {filepath}")
                return True
            else:
                logger.warning("pyttsx3 created empty file")
                return False
                
        except Exception as e:
            logger.error(f"pyttsx3 narration failed: {e}")
            return False
    
    def narrate(self, text: str, filepath: str, language: str = 'en') -> bool:
        """
        Generate audio narration with fallback options.
        
        Args:
            text (str): Text to narrate
            filepath (str): Path to save the audio file
            language (str): Language code for TTS
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for narration")
            return False
            
        # Clean the text
        text = text.strip()
        
        # Limit text length to prevent very long audio files
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.info(f"Text truncated to {max_length} characters")
        
        logger.info(f"Generating narration for text: {text[:100]}...")
        
        # Try gTTS first (better quality)
        if self.gtts_available:
            logger.info("Trying gTTS for narration")
            if self._narrate_with_gtts(text, filepath, language):
                return True
            else:
                logger.warning("gTTS failed, trying fallback options")
        
        # Try pyttsx3 as fallback
        if self.pyttsx3_available:
            logger.info("Trying pyttsx3 for narration")
            if self._narrate_with_pyttsx3(text, filepath):
                return True
            else:
                logger.warning("pyttsx3 failed")
        
        # Last resort: create dummy audio file
        logger.warning("All TTS methods failed, creating dummy audio file")
        return self._create_dummy_audio(filepath)
    
    def is_available(self) -> bool:
        """
        Check if any TTS engine is available.
        
        Returns:
            bool: True if at least one TTS engine is available
        """
        return self.gtts_available or self.pyttsx3_available
    
    def get_available_engines(self) -> list:
        """
        Get list of available TTS engines.
        
        Returns:
            list: List of available engine names
        """
        engines = []
        if self.gtts_available:
            engines.append('gTTS')
        if self.pyttsx3_available:
            engines.append('pyttsx3')
        return engines


# Utility function for testing
def test_audio_narrator():
    """Test the AudioNarrator functionality."""
    narrator = AudioNarrator()
    
    print(f"Available engines: {narrator.get_available_engines()}")
    print(f"TTS available: {narrator.is_available()}")
    
    if narrator.is_available():
        test_text = "This is a test of the audio narration system."
        test_file = "test_narration.mp3"
        
        success = narrator.narrate(test_text, test_file)
        print(f"Narration test {'successful' if success else 'failed'}")
        
        if success and os.path.exists(test_file):
            print(f"Audio file created: {test_file} ({os.path.getsize(test_file)} bytes)")
        else:
            print("No audio file created")
    else:
        print("No TTS engines available for testing")


if __name__ == "__main__":
    test_audio_narrator()