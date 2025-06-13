#!/usr/bin/env python3
"""
Book Narrator - AI-powered image description and narration
Main Flask application file
"""

import unittest
import os
import argparse
import logging
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Import your custom modules
try:
    from src.llm.base_model import ImageDescriber
    from src.utils.image_utils import ImageProcessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have the src/llm/base_model.py and src/utils/image_utils.py files")
    print("Run the debug script first to check your file structure")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Flask app configuration
app = Flask(__name__,
           template_folder=os.path.join(BASE_DIR, 'templates'),
           static_folder=os.path.join(BASE_DIR, 'static'))

# Ensure static directories exist
os.makedirs(os.path.join(app.static_folder, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(app.static_folder, 'audio'), exist_ok=True)

# Update file paths
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['AUDIO_FOLDER'] = os.path.join(BASE_DIR, 'static', 'audio')

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'output', 'uploads')
app.config['AUDIO_FOLDER'] = os.path.join(BASE_DIR, 'output', 'audio')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['AUDIO_FOLDER']]:
    os.makedirs(folder, exist_ok=True)
    logger.info(f"Ensured directory exists: {folder}")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path: str, audio_folder: str = None):
    """Process a single image with narration."""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Validate image first
        if not ImageProcessor.validate_image(image_path):
            logger.error(f"Invalid image file: {image_path}")
            return None
            
        describer = ImageDescriber(audio_folder=audio_folder)
        result = describer.describe_image(image_path, narrate=True)
        
        return result
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large. Please upload an image smaller than 16MB.'
    }), 413

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    try:
        logger.info(f"Serving static file: {filename}")
        return send_from_directory(app.static_folder, filename)
    except FileNotFoundError:
        logger.error(f"Static file not found: {filename}")
        return "File not found", 404

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            flash('No image uploaded')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            
            # Save to uploads folder within static
            upload_path = os.path.join(app.static_folder, 'uploads')
            os.makedirs(upload_path, exist_ok=True)
            filepath = os.path.join(upload_path, filename)
            file.save(filepath)
            
            # Process image
            result = process_image(filepath, audio_folder=app.config['AUDIO_FOLDER'])
            
            if result:
                # Create URLs for static files
                image_url = url_for('static', filename=f'uploads/{filename}')
                audio_filename = os.path.basename(result.get('audio_path', ''))
                audio_url = url_for('audio_file', filename=audio_filename) if audio_filename else None
                
                # Extract scene overview
                full_description = result.get('description', '')
                scene_overview = ''
                if '🖼️ Scene Overview:' in full_description:
                    scene_overview = full_description.split('👥')[0].strip()
                
                return render_template('index.html',
                                    image_path=image_url,
                                    description=scene_overview,
                                    audio_path=audio_url)
            
        return 'Processing failed', 400
                
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return str(e), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    try:
        upload_path = os.path.abspath(app.config['UPLOAD_FOLDER'])
        logger.info(f"Serving file: {filename} from {upload_path}")
        return send_from_directory(upload_path, filename)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return "File not found", 404

@app.route('/audio/<filename>')
def audio_file(filename):
    """Serve audio files with proper MIME type."""
    try:
        return send_from_directory(
            app.config['AUDIO_FOLDER'],
            filename,
            mimetype='audio/mpeg',
            as_attachment=False
        )
    except FileNotFoundError:
        return "Audio file not found", 404

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'audio_folder': app.config['AUDIO_FOLDER'],
        'base_dir': BASE_DIR
    })

@app.route('/debug')
def debug_info():
    """Debug information endpoint."""
    return jsonify({
        'base_dir': BASE_DIR,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'audio_folder': app.config['AUDIO_FOLDER'],
        'template_folder': app.template_folder,
        'static_folder': app.static_folder,
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'audio_folder_exists': os.path.exists(app.config['AUDIO_FOLDER']),
        'template_folder_exists': os.path.exists(app.template_folder),
        'files_in_upload': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else [],
        'files_in_audio': os.listdir(app.config['AUDIO_FOLDER']) if os.path.exists(app.config['AUDIO_FOLDER']) else []
    })

class TestImageDescriber(unittest.TestCase):
    """Unit tests for the ImageDescriber functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.describer = ImageDescriber()
        self.test_image_path = os.path.join(BASE_DIR, "data", "shared image (3).jpg")
    
    def test_image_exists(self):
        """Test if the test image file exists."""
        if os.path.exists(self.test_image_path):
            self.assertTrue(True)
        else:
            self.skipTest(f"Test image not found at {self.test_image_path}")
    
    def test_image_validation(self):
        """Test image validation functionality."""
        if os.path.exists(self.test_image_path):
            self.assertTrue(ImageProcessor.validate_image(self.test_image_path))
        else:
            self.skipTest("Test image file not found")
    
    def test_image_description(self):
        """Test the image description generation."""
        if not os.path.exists(self.test_image_path):
            self.skipTest("Test image file not found")
            
        try:
            description = self.describer.describe_image(self.test_image_path)
            
            # Basic validation
            self.assertIsNotNone(description)
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 50, "Description seems too short")
            
            # Print description for manual review
            print(f"\n{'='*50}")
            print("Generated Description:")
            print('='*50)
            print(description)
            print('='*50)
            
        except Exception as e:
            self.fail(f"Image description failed with error: {str(e)}")

def run_web(host='127.0.0.1', port=5000, debug=True):
    """
    Run the web interface.
    
    Args:
        host (str): Host address to bind to
        port (int): Port number to bind to
        debug (bool): Enable debug mode
    """
    logger.info(f"Starting web server on {host}:{port}")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Audio folder: {app.config['AUDIO_FOLDER']}")
    logger.info(f"Template folder: {app.template_folder}")
    
    # Check if critical directories exist
    critical_dirs = [
        ('Templates', app.template_folder),
        ('Upload', app.config['UPLOAD_FOLDER']),
        ('Audio', app.config['AUDIO_FOLDER'])
    ]
    
    for name, path in critical_dirs:
        if os.path.exists(path):
            logger.info(f"✓ {name} directory exists: {path}")
        else:
            logger.warning(f"✗ {name} directory missing: {path}")
    
    # Check for index.html
    index_path = os.path.join(app.template_folder, 'index.html')
    if os.path.exists(index_path):
        logger.info("✓ index.html template found")
    else:
        logger.error(f"✗ index.html template missing at: {index_path}")
    
    print(f"\n🌟 Book Narrator starting...")
    print(f"📂 Project directory: {BASE_DIR}")
    print(f"🌐 Access the app at: http://{host}:{port}")
    print(f"🔧 Debug endpoint: http://{host}:{port}/debug")
    print(f"❤️  Health check: http://{host}:{port}/health")
    
    app.run(host=host, port=port, debug=debug)

def run_tests():
    """Run unit tests."""
    logger.info("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

def process_single_image(image_path: str):
    """
    Process a single image from command line.
    
    Args:
        image_path (str): Path to the image file
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return
        
    logger.info(f"Processing image: {image_path}")
    result = process_image(image_path)
    
    if result:
        print(f"\n{'='*50}")
        print("GENERATED DESCRIPTION:")
        print('='*50)
        print(result['description'])
        print('='*50)
        
        if result.get('audio_path'):
            print(f"\nAudio narration saved to: {result['audio_path']}")
            
            # Try to open audio file (cross-platform)
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(result['audio_path'])
                elif os.name == 'posix':  # macOS and Linux
                    import subprocess
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', result['audio_path']], check=True)
                    else:  # Linux
                        subprocess.run(['xdg-open', result['audio_path']], check=True)
            except Exception as e:
                logger.warning(f"Could not automatically open audio file: {e}")
                print("You can manually open the audio file to listen to the narration.")
    else:
        logger.error("Failed to process image")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description='Book Narrator - AI-powered image description and narration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --web                    # Run web interface (default)
  python main.py --test                   # Run unit tests
  python main.py --image path/to/img.jpg  # Process single image
  python main.py --web --port 8080        # Run web on custom port
  python main.py --web --host 0.0.0.0     # Run web accessible from network
        """
    )
    
    parser.add_argument('--test', action='store_true', 
                       help='Run unit tests')
    parser.add_argument('--image', type=str, 
                       help='Path to image file to process')
    parser.add_argument('--web', action='store_true', 
                       help='Run web interface')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address for web server (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port for web server (default: 5000)')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug mode for web server')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            run_tests()
        elif args.image:
            process_single_image(args.image)
        else:
            # Default to web interface
            debug_mode = not args.no_debug
            run_web(host=args.host, port=args.port, debug=debug_mode)
                
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == '__main__':
    import sys
    main()
