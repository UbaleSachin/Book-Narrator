#!/usr/bin/env python3
"""
Book Narrator - AI-powered image description and narration
Main Flask application file with PDF support
"""
import time
import gc
import unittest
import os
import argparse
import logging
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
# Replace the fitz import with:
try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError("PyMuPDF not installed correctly. Install it with `pip install pymupdf`") from e
from PIL import Image
import io
import json
import threading

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

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size for PDFs

# File paths - using static folder for web-accessible files
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['AUDIO_FOLDER'] = os.path.join(app.static_folder, 'audio')
app.config['PDF_IMAGES_FOLDER'] = os.path.join(app.static_folder, 'pdf_images')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'pdf'}
PDF_EXTENSIONS = {'pdf'}

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['AUDIO_FOLDER'], app.config['PDF_IMAGES_FOLDER']]:
    os.makedirs(folder, exist_ok=True)
    logger.info(f"Ensured directory exists: {folder}")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_pdf_file(filename):
    """Check if the uploaded file is a PDF."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in PDF_EXTENSIONS

class PDFProcessor:
    """Class to handle PDF processing and image extraction with cloud deployment optimizations."""
    
    @staticmethod
    def extract_images_from_pdf(pdf_path: str, output_folder: str, base_filename: str, max_pages: int = 20):
        """
        Extract images from PDF file with memory management.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_folder (str): Directory to save extracted images
            base_filename (str): Base name for extracted image files
            max_pages (int): Maximum number of pages to process (default: 20)
            
        Returns:
            list: List of paths to extracted image files
        """
        doc = None
        try:
            logger.info(f"Extracting images from PDF: {pdf_path}")
            
            # Open PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            image_paths = []
            
            # Limit pages for cloud deployment
            pages_to_process = min(total_pages, max_pages)
            if total_pages > max_pages:
                logger.warning(f"PDF has {total_pages} pages, processing only first {max_pages} for cloud deployment")
            
            for page_num in range(pages_to_process):
                try:
                    page = doc.load_page(page_num)
                    
                    # Get page as image with reduced quality for cloud deployment
                    mat = fitz.Matrix(1.5, 1.5)  # Reduced from 2x to 1.5x zoom
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Optimize image size for cloud deployment
                    if img.width > 1200 or img.height > 1200:
                        img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                    
                    # Save image with compression
                    image_filename = f"{base_filename}_page_{page_num + 1:03d}.png"
                    image_path = os.path.join(output_folder, image_filename)
                    img.save(image_path, "PNG", optimize=True)
                    
                    image_paths.append(image_path)
                    logger.info(f"Extracted page {page_num + 1} to {image_path}")
                    
                    # Clean up resources immediately
                    img.close()
                    del img
                    pix = None
                    page = None
                    
                    # Force garbage collection every few pages
                    if (page_num + 1) % 3 == 0:
                        gc.collect()
                        time.sleep(0.1)  # Small delay to prevent overwhelming the system
                    
                except Exception as e:
                    logger.error(f"Error extracting page {page_num + 1}: {str(e)}")
                    continue
            
            logger.info(f"Successfully extracted {len(image_paths)} pages from PDF")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            return []
        finally:
            if doc:
                doc.close()
            gc.collect()
    
    @staticmethod
    def has_meaningful_content(image_path: str, min_colors: int = 10):
        """
        Check if an image has meaningful content (not just blank pages).
        
        Args:
            image_path (str): Path to the image
            min_colors (int): Minimum number of unique colors to consider meaningful
            
        Returns:
            bool: True if image has meaningful content
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get unique colors
                colors = img.getcolors(maxcolors=256*256*256)
                unique_colors = len(colors) if colors else 0
                
                # Check if image has enough color variation
                return unique_colors >= min_colors
                
        except Exception as e:
            logger.error(f"Error checking image content: {str(e)}")
            return True  # Default to processing if we can't determine

def process_image(image_path: str, audio_folder: str = None):
    """Process a single image with narration."""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Validate image first
        if not ImageProcessor.validate_image(image_path):
            logger.error(f"Invalid image file: {image_path}")
            return {'error': 'Invalid image file'}
            
        describer = ImageDescriber(audio_folder=audio_folder)
        result = describer.describe_image(image_path, narrate=True)
        
        if result and 'error' not in result:
            # Verify audio file was created
            if 'audio_path' in result and not os.path.exists(result['audio_path']):
                logger.warning("Audio generation failed, returning description only")
                result['audio_path'] = None
                result['audio_filename'] = None
            return result
        else:
            logger.error("Failed to generate description")
            return {'error': 'Failed to process image'}
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return {'error': str(e)}

def process_pdf_book(pdf_path: str, output_folder: str, audio_folder: str = None, max_pages: int = 20, batch_size: int = 2):
    """
    Process a PDF book and generate narrations for each page with cloud deployment optimizations.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Directory to save extracted images
        audio_folder (str): Directory to save audio files
        max_pages (int): Maximum number of pages to process
        batch_size (int): Number of pages to process at once
        
    Returns:
        dict: Results containing page descriptions and audio files
    """
    try:
        logger.info(f"Processing PDF book: {pdf_path} (max {max_pages} pages)")
        
        # Extract base filename
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Extract images from PDF with page limit
        image_paths = PDFProcessor.extract_images_from_pdf(pdf_path, output_folder, base_filename, max_pages)
        
        if not image_paths:
            logger.error("No images extracted from PDF")
            return None
        
        # Process pages in batches to manage memory
        results = {
            'pdf_name': base_filename,
            'total_pages': len(image_paths),
            'pages': []
        }
        
        # Process images in batches
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_images = image_paths[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: pages {batch_start + 1} to {batch_end}")
            
            # Process batch with timeout and error handling
            batch_results = process_image_batch(batch_images, audio_folder, batch_start)
            results['pages'].extend(batch_results)
            
            # Clean up after each batch
            gc.collect()
            time.sleep(1)  # Brief pause between batches
        
        logger.info(f"PDF processing complete. Processed {len(results['pages'])} pages")
        return results
        
    except Exception as e:
        logger.error(f"Error processing PDF book: {str(e)}")
        return None
    
def process_image_batch(image_paths: list, audio_folder: str = None, batch_offset: int = 0):
    """
    Process a batch of images with error handling and timeouts.
    
    Args:
        image_paths (list): List of image paths to process
        audio_folder (str): Directory to save audio files
        batch_offset (int): Offset for page numbering
        
    Returns:
        list: List of processed page results
    """
    batch_results = []
    
    for i, image_path in enumerate(image_paths):
        page_num = batch_offset + i + 1
        
        try:
            logger.info(f"Processing page {page_num}")
            
            # Check if page has meaningful content
            if not PDFProcessor.has_meaningful_content(image_path):
                logger.info(f"Skipping page {page_num} - appears to be blank")
                continue
            
            # Process the page image with timeout
            result = process_image_with_timeout(image_path, audio_folder, timeout=120)
            
            if result and 'error' not in result:
                page_result = {
                    'page_number': page_num,
                    'image_path': image_path,
                    'image_filename': os.path.basename(image_path),
                    'description': result.get('description', ''),
                    'audio_path': result.get('audio_path', ''),
                    'audio_filename': os.path.basename(result.get('audio_path', '')) if result.get('audio_path') else ''
                }
                batch_results.append(page_result)
                logger.info(f"Successfully processed page {page_num}")
            else:
                logger.warning(f"Failed to process page {page_num}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            continue
    
    return batch_results

def process_image_with_timeout(image_path: str, audio_folder: str = None, timeout: int = 120):
    """
    Process a single image with timeout for cloud deployment.
    
    Args:
        image_path (str): Path to the image file
        audio_folder (str): Directory to save audio files
        timeout (int): Timeout in seconds
        
    Returns:
        dict: Processing result or error
    """
    def process_image_worker():
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Validate image first
            if not ImageProcessor.validate_image(image_path):
                logger.error(f"Invalid image file: {image_path}")
                return {'error': 'Invalid image file'}
            
            # Initialize describer with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    describer = ImageDescriber(audio_folder=audio_folder)
                    result = describer.describe_image(image_path, narrate=True)
                    
                    if result and 'error' not in result:
                        # Verify audio file was created
                        if 'audio_path' in result and result['audio_path'] and not os.path.exists(result['audio_path']):
                            logger.warning("Audio generation failed, returning description only")
                            result['audio_path'] = None
                            result['audio_filename'] = None
                        return result
                    else:
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                            time.sleep(5)  # Wait before retry
                            continue
                        else:
                            logger.error("All attempts failed to generate description")
                            return {'error': 'Failed to process image after multiple attempts'}
                            
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}, retrying...")
                        time.sleep(5)
                        continue
                    else:
                        logger.error(f"Final attempt failed: {str(e)}")
                        return {'error': str(e)}
                        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return {'error': str(e)}
    
    # Use threading for timeout control
    result_container = {'result': None, 'completed': False}
    
    def worker():
        result_container['result'] = process_image_worker()
        result_container['completed'] = True
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if result_container['completed']:
        return result_container['result']
    else:
        logger.error(f"Image processing timed out after {timeout} seconds")
        return {'error': f'Processing timed out after {timeout} seconds'}


@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large. Please upload a file smaller than 50MB.'
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
    """Process uploaded file (image or PDF) with cloud deployment optimizations."""
    try:
        if 'image' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('index'))
        
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))

        # Get selected language
        language = request.form.get('language', 'en-US')
            
        if file and allowed_file(file.filename):
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            
            # Save uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File saved to: {filepath}")
            
            # Check if it's a PDF
            if is_pdf_file(file.filename):
                logger.info("Processing PDF file")
                
                # Check file size for cloud deployment
                file_size = os.path.getsize(filepath)
                max_size = 25 * 1024 * 1024  # 25MB limit for cloud
                
                if file_size > max_size:
                    flash('PDF file too large for cloud processing. Please upload a smaller file (max 25MB).')
                    os.remove(filepath)  # Clean up
                    return redirect(url_for('index'))
                
                # Process PDF with limits for cloud deployment
                results = process_pdf_book(
                    filepath, 
                    app.config['PDF_IMAGES_FOLDER'], 
                    audio_folder=app.config['AUDIO_FOLDER'],
                    max_pages=15,  # Limit pages for cloud
                    batch_size=2   # Process in smaller batches
                )
                
                # Clean up uploaded PDF after processing
                try:
                    os.remove(filepath)
                except:
                    pass
                
                if results and results['pages']:
                    # Create URLs for the results
                    for page in results['pages']:
                        # Create static URLs for images and audio
                        page['image_url'] = url_for('static', filename=f"pdf_images/{page['image_filename']}")
                        if page['audio_filename']:
                            page['audio_url'] = url_for('static', filename=f"audio/{page['audio_filename']}")
                        else:
                            page['audio_url'] = None
                    
                    logger.info(f"Rendering PDF results with {len(results['pages'])} pages")
                    return render_template('index.html', 
                                         pdf_results=results,
                                         file_type='pdf')
                else:
                    flash('Failed to process PDF. Please ensure it contains readable images.')
                    return redirect(url_for('index'))
            else:
                logger.info("Processing single image file")
                
                # Check image size for cloud deployment
                try:
                    with Image.open(filepath) as img:
                        if img.width > 2000 or img.height > 2000:
                            # Resize large images
                            img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
                            img.save(filepath, optimize=True)
                except Exception as e:
                    logger.warning(f"Could not optimize image: {e}")
                
                # Process single image with timeout
                result = process_image_with_timeout(filepath, audio_folder=app.config['AUDIO_FOLDER'], timeout=90)
                
                if result and 'error' not in result:
                    # Create URLs for static files
                    image_url = url_for('static', filename=f'uploads/{filename}')
                    
                    # Handle audio file URL
                    audio_url = None
                    if result.get('audio_path'):
                        audio_filename = os.path.basename(result['audio_path'])
                        audio_url = url_for('static', filename=f'audio/{audio_filename}')
                    
                    # Extract scene overview
                    full_description = result.get('description', '')
                    
                    scene_overview = ''
                    if '🖼️ Scene Overview:' in full_description:
                        scene_overview = full_description.split('👥')[0].strip()
                    else:
                        scene_overview = full_description
                    
                    return render_template('index.html',
                                        image_path=image_url,
                                        description=scene_overview,
                                        audio_path=audio_url,
                                        file_type='image')
                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'Failed to process image'
                    flash(f'Failed to process image: {error_msg}')
                    return redirect(url_for('index'))
            
        flash('Invalid file format. Please upload an image or PDF file.')
        return redirect(url_for('index'))
                
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('index'))

def cleanup_old_files():
    """Clean up old files to prevent disk space issues in cloud deployment."""
    try:
        current_time = time.time()
        max_age = 3600  # 1 hour
        
        folders_to_clean = [
            app.config['UPLOAD_FOLDER'],
            app.config['AUDIO_FOLDER'],
            app.config['PDF_IMAGES_FOLDER']
        ]
        
        for folder in folders_to_clean:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        file_age = current_time - os.path.getctime(filepath)
                        if file_age > max_age:
                            try:
                                os.remove(filepath)
                                logger.info(f"Cleaned up old file: {filepath}")
                            except Exception as e:
                                logger.warning(f"Could not clean up file {filepath}: {e}")
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    try:
        logger.info(f"Serving uploaded file: {filename}")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
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
        logger.error(f"Audio file not found: {filename}")
        return "Audio file not found", 404

@app.route('/pdf_images/<filename>')
def pdf_image_file(filename):
    """Serve PDF extracted images."""
    try:
        return send_from_directory(app.config['PDF_IMAGES_FOLDER'], filename)
    except FileNotFoundError:
        logger.error(f"PDF image file not found: {filename}")
        return "Image file not found", 404

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'audio_folder': app.config['AUDIO_FOLDER'],
        'pdf_images_folder': app.config['PDF_IMAGES_FOLDER'],
        'base_dir': BASE_DIR,
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'audio_folder_exists': os.path.exists(app.config['AUDIO_FOLDER']),
        'pdf_images_folder_exists': os.path.exists(app.config['PDF_IMAGES_FOLDER'])
    })

@app.route('/debug')
def debug_info():
    """Debug information endpoint."""
    return jsonify({
        'base_dir': BASE_DIR,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'audio_folder': app.config['AUDIO_FOLDER'],
        'pdf_images_folder': app.config['PDF_IMAGES_FOLDER'],
        'template_folder': app.template_folder,
        'static_folder': app.static_folder,
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'audio_folder_exists': os.path.exists(app.config['AUDIO_FOLDER']),
        'pdf_images_folder_exists': os.path.exists(app.config['PDF_IMAGES_FOLDER']),
        'template_folder_exists': os.path.exists(app.template_folder),
        'files_in_upload': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else [],
        'files_in_audio': os.listdir(app.config['AUDIO_FOLDER']) if os.path.exists(app.config['AUDIO_FOLDER']) else [],
        'files_in_pdf_images': os.listdir(app.config['PDF_IMAGES_FOLDER']) if os.path.exists(app.config['PDF_IMAGES_FOLDER']) else []
    })

class TestImageDescriber(unittest.TestCase):
    """Unit tests for the ImageDescriber functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.describer = ImageDescriber()
        self.test_image_path = os.path.join(BASE_DIR, "data", "shared image (3).jpg")
        self.test_pdf_path = os.path.join(BASE_DIR, "data", "test_book.pdf")
    
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
    
    def test_pdf_processing(self):
        """Test PDF processing functionality."""
        if not os.path.exists(self.test_pdf_path):
            self.skipTest("Test PDF file not found")
            
        try:
            output_folder = os.path.join(BASE_DIR, "test_output")
            os.makedirs(output_folder, exist_ok=True)
            
            results = process_pdf_book(self.test_pdf_path, output_folder)
            
            self.assertIsNotNone(results)
            self.assertIn('pages', results)
            self.assertGreater(len(results['pages']), 0)
            
            print(f"\n{'='*50}")
            print("PDF Processing Results:")
            print('='*50)
            print(f"Total pages processed: {len(results['pages'])}")
            for page in results['pages']:
                print(f"Page {page['page_number']}: {page['description'][:100]}...")
            print('='*50)
            
        except Exception as e:
            self.fail(f"PDF processing failed with error: {str(e)}")

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
    logger.info(f"PDF images folder: {app.config['PDF_IMAGES_FOLDER']}")
    logger.info(f"Template folder: {app.template_folder}")
    
    # Check if critical directories exist
    critical_dirs = [
        ('Templates', app.template_folder),
        ('Upload', app.config['UPLOAD_FOLDER']),
        ('Audio', app.config['AUDIO_FOLDER']),
        ('PDF Images', app.config['PDF_IMAGES_FOLDER'])
    ]
    
    for name, path in critical_dirs:
        if os.path.exists(path):
            logger.info(f"✓ {name} directory exists: {path}")
        else:
            logger.warning(f"✗ {name} directory missing: {path}")
            os.makedirs(path, exist_ok=True)
            logger.info(f"✓ Created {name} directory: {path}")
    
    # Check for templates
    templates = ['index.html']  # Removed pdf_results.html
    for template in templates:
        template_path = os.path.join(app.template_folder, template)
        if os.path.exists(template_path):
            logger.info(f"✓ {template} template found")
        else:
            logger.error(f"✗ {template} template missing at: {template_path}")
    
    print(f"\n🌟 Book Narrator with PDF support starting...")
    print(f"📂 Project directory: {BASE_DIR}")
    print(f"🌐 Access the app at: http://{host}:{port}")
    print(f"🔧 Debug endpoint: http://{host}:{port}/debug")
    print(f"❤️  Health check: http://{host}:{port}/health")
    print(f"📖 Now supports PDF picture books!")
    
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

def process_single_pdf(pdf_path: str):
    """
    Process a single PDF from command line.
    
    Args:
        pdf_path (str): Path to the PDF file
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return
        
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Create output directory
    output_dir = os.path.join(BASE_DIR, "output", "pdf_processing")
    os.makedirs(output_dir, exist_ok=True)
    
    result = process_pdf_book(pdf_path, output_dir)
    
    if result:
        print(f"\n{'='*50}")
        print("PDF PROCESSING RESULTS:")
        print('='*50)
        print(f"Book: {result['pdf_name']}")
        print(f"Total pages: {result['total_pages']}")
        print(f"Processed pages: {len(result['pages'])}")
        print('='*50)
        
        for page in result['pages']:
            print(f"\nPage {page['page_number']}:")
            print("-" * 30)
            print(page['description'])
            if page['audio_path']:
                print(f"Audio: {page['audio_path']}")
    else:
        logger.error("Failed to process PDF")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description='Book Narrator - AI-powered image description and narration with PDF support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --web                    # Run web interface (default)
  python main.py --test                   # Run unit tests
  python main.py --image path/to/img.jpg  # Process single image
  python main.py --pdf path/to/book.pdf   # Process PDF book
  python main.py --web --port 8080        # Run web on custom port
  python main.py --web --host 0.0.0.0     # Run web accessible from network
        """
    )
    
    parser.add_argument('--test', action='store_true', 
                       help='Run unit tests')
    parser.add_argument('--image', type=str, 
                       help='Path to image file to process')
    parser.add_argument('--pdf', type=str, 
                       help='Path to PDF file to process')
    parser.add_argument('--web', action='store_true', 
                       help='Run web interface')
    parser.add_argument('--host', type=str, default='127.0.0.1',
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
        elif args.pdf:
            process_single_pdf(args.pdf)
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
    import atexit
    atexit.register(cleanup_old_files)
    main()