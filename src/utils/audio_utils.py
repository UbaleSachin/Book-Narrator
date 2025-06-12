from gtts import gTTS
import os
from pathlib import Path

class AudioNarrator:
    def __init__(self, output_dir: str = "output/audio"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def narrate(self, text: str, filename: str = None) -> str:
        """Convert text to speech and save as audio file."""
        try:
            if not filename:
                # Create filename from first few words of text
                filename = "_".join(text.split()[:5]).lower()
                filename = "".join(c for c in filename if c.isalnum() or c == "_")
                filename = f"{filename}.mp3"
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Generate speech
            tts = gTTS(text=text, lang='en')
            tts.save(filepath)
            
            return filepath
            
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return None