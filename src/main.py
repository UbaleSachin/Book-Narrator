#!/usr/bin/env python3
"""
Book Narrator - Fixed PDF processing for hosted platforms
Main improvements for deployment compatibility
"""

import unittest
import os
import argparse
import logging
import tempfile
import shutil
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# More robust PyMuPDF import with fallback
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyMuPDF not available: {e}")
    print("PDF processing will be disabled")
    PYMUPDF_AVAILABLE = False
    fitz = None

from PIL import Image
import io
import json
import sys
import traceback

# Import your custom modules with better error handling
try:
    from src.llm.base_model import ImageDescriber
    from src.utils.image_utils import ImageProcessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have the src/llm/base_model.py and src/utils/image_utils.py files")
    sys.exit(1)

# Configure logging with more details for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a') if os.access('.', os.W_OK) else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Flask app configuration
app = Flask(__name__,
           template_folder=os.path.join(BASE_DIR, 'templates'),
           static_folder=os.path.join(BASE_DIR, 'static'))

# Configuration with environment variable support
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))  # 50MB default

# Use temp directory for hosted platforms
if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RENDER'):
    # For hosted platforms, use system temp directory
    temp_base = tempfile.gettempdir()
    app.config['UPLOAD_FOLDER'] = os.path.join(temp_base, 'uploads')
    app.config['AUDIO_FOLDER'] = os.path.join(temp_base, 'audio')
    app.config['PDF_IMAGES_FOLDER'] = os.path.join(temp_base, 'pdf_images')
else:
    # For local development, use static folder
    app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
    app.config['AUDIO_FOLDER'] = os.path.join(app.static_folder, 'audio')
    app.config['PDF_IMAGES_FOLDER'] = os.path.join(app.static_folder, 'pdf_images')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
if PYMUPDF_AVAILABLE:
    ALLOWED_EXTENSIONS.add('pdf')

PDF_EXTENSIONS = {'pdf'}

# Create necessary directories with proper error handling
for folder_name, folder_path in [
    ('Upload', app.config['UPLOAD_FOLDER']),
    ('Audio', app.config['AUDIO_FOLDER']),
    ('PDF Images', app.config['PDF_IMAGES_FOLDER'])
]:
    try:
        os.makedirs(folder_path, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(folder_path, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logger.info(f"✓ {folder_name} directory ready: {folder_path}")
    except Exception as e:
        logger.error(f"✗ Failed to create/access {folder_name} directory {folder_path}: {e}")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_pdf_file(filename):
    """Check if the uploaded file is a PDF."""
    return PYMUPDF_AVAILABLE and '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in PDF_EXTENSIONS

class PDFProcessor:
    """Enhanced PDF processing class with better error handling for hosted platforms."""
    
    @staticmethod
    def extract_images_from_pdf(pdf_path: str, output_folder: str, base_filename: str):
        """
        Extract images from PDF file with enhanced error handling.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_folder (str): Directory to save extracted images
            base_filename (str): Base name for extracted image files
            
        Returns:
            list: List of paths to extracted image files
        """
        if not PYMUPDF_AVAILABLE:
            logger.error("PyMuPDF not available for PDF processing")
            return []
            
        doc = None
        try:
            logger.info(f"Extracting images from PDF: {pdf_path}")
            
            # Verify PDF file exists and is readable
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return []
                
            if os.path.getsize(pdf_path) == 0:
                logger.error(f"PDF file is empty: {pdf_path}")
                return []
            
            # Ensure output directory exists and is writable
            os.makedirs(output_folder, exist_ok=True)
            
            # Test write permissions
            test_file = os.path.join(output_folder, '.write_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                logger.error(f"Output folder not writable: {output_folder} - {e}")
                return []
            
            # Open PDF with error handling
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                logger.error(f"Failed to open PDF: {e}")
                return []
            
            if doc.page_count == 0:
                logger.error("PDF has no pages")
                return []
                
            image_paths = []
            
            for page_num in range(len(doc)):
                try:
                    logger.info(f"Processing page {page_num + 1}/{len(doc)}")
                    page = doc.load_page(page_num)
                    
                    # Get page as image with error handling
                    try:
                        # Use smaller matrix for hosted platforms to reduce memory usage
                        mat = fitz.Matrix(1.5, 1.5)  # Reduced from 2x to 1.5x zoom
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Convert to PIL Image with memory management
                        img_data = pix.tobytes("png")
                        pix = None  # Free memory immediately
                        
                        img = Image.open(io.BytesIO(img_data))
                        img_data = None  # Free memory
                        
                        # Save image with compression for hosted platforms
                        image_filename = f"{base_filename}_page_{page_num + 1:03d}.png"
                        image_path = os.path.join(output_folder, image_filename)
                        
                        # Optimize image for hosted platforms
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Compress image if it's too large
                        max_size = (1200, 1600)  # Reasonable size for processing
                        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                            img.thumbnail(max_size, Image.Resampling.LANCZOS)
                        
                        img.save(image_path, "PNG", optimize=True)
                        img.close()
                        
                        # Verify the file was created successfully
                        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                            image_paths.append(image_path)
                            logger.info(f"✓ Extracted page {page_num + 1} to {image_path}")
                        else:
                            logger.error(f"✗ Failed to save page {page_num + 1}")
                            
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {e}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error loading page {page_num + 1}: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(image_paths)} pages from PDF")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        finally:
            # Ensure document is closed
            if doc:
                try:
                    doc.close()
                except:
                    pass
    
    @staticmethod
    def has_meaningful_content(image_path: str, min_colors: int = 5):
        """
        Check if an image has meaningful content with reduced threshold for hosted platforms.
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Reduce image size for faster processing on hosted platforms
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                
                # Get unique colors with lower limit to reduce memory usage
                colors = img.getcolors(maxcolors=1000)
                unique_colors = len(colors) if colors else 0
                
                # Check if image has enough color variation (reduced threshold)
                return unique_colors >= min_colors
                
        except Exception as e:
            logger.error(f"Error checking image content: {str(e)}")
            return True  # Default to processing if we can't determine

def process_image(image_path: str, audio_folder: str = None):
    """Process a single image with enhanced error handling."""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Validate image exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        # Validate image first
        if not ImageProcessor.validate_image(image_path):
            logger.error(f"Invalid image file: {image_path}")
            return None
            
        describer = ImageDescriber(audio_folder=audio_folder)
        result = describer.describe_image(image_path, narrate=True)
        
        return result
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def process_pdf_book(pdf_path: str, output_folder: str, audio_folder: str = None):
    """
    Enhanced PDF processing with better error handling for hosted platforms.
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PDF processing not available - PyMuPDF not installed")
        return None
        
    try:
        logger.info(f"Processing PDF book: {pdf_path}")
        
        # Validate PDF file
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None
            
        if os.path.getsize(pdf_path) == 0:
            logger.error(f"PDF file is empty: {pdf_path}")
            return None
        
        # Extract base filename
        base_filename = secure_filename(os.path.splitext(os.path.basename(pdf_path))[0])
        
        # Create unique output folder to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_output_folder = os.path.join(output_folder, f"{base_filename}_{timestamp}")
        os.makedirs(unique_output_folder, exist_ok=True)
        
        # Extract images from PDF
        image_paths = PDFProcessor.extract_images_from_pdf(pdf_path, unique_output_folder, base_filename)
        
        if not image_paths:
            logger.error("No images extracted from PDF")
            return None
        
        # Process each page with better error handling
        results = {
            'pdf_name': base_filename,
            'total_pages': len(image_paths),
            'pages': [],
            'processing_timestamp': timestamp
        }
        
        successful_pages = 0
        
        for i, image_path in enumerate(image_paths):
            page_num = i + 1
            logger.info(f"Processing page {page_num}/{len(image_paths)}")
            
            try:
                # Check if page has meaningful content
                if not PDFProcessor.has_meaningful_content(image_path):
                    logger.info(f"Skipping page {page_num} - appears to be blank")
                    continue
                
                # Process the page image
                result = process_image(image_path, audio_folder=audio_folder)
                
                if result and result.get('description'):
                    # Get relative path for web serving
                    relative_image_path = os.path.relpath(image_path, app.static_folder)
                    
                    page_result = {
                        'page_number': page_num,
                        'image_path': image_path,
                        'image_filename': os.path.basename(image_path),
                        'relative_image_path': relative_image_path,
                        'description': result.get('description', ''),
                        'audio_path': result.get('audio_path', ''),
                        'audio_filename': os.path.basename(result.get('audio_path', '')) if result.get('audio_path') else ''
                    }
                    results['pages'].append(page_result)
                    successful_pages += 1
                    logger.info(f"✓ Successfully processed page {page_num}")
                else:
                    logger.warning(f"✗ Failed to process page {page_num}")
                    
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue
        
        if successful_pages > 0:
            logger.info(f"PDF processing complete. Successfully processed {successful_pages}/{len(image_paths)} pages")
            return results
        else:
            logger.error("No pages were successfully processed")
            return None
        
    except Exception as e:
        logger.error(f"Error processing PDF book: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large. Please upload a file smaller than 50MB.'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors with detailed logging."""
    logger.error(f"Internal Server Error: {error}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'Please check the server logs for details.'
    }), 500

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html', pdf_supported=PYMUPDF_AVAILABLE)

@app.route('/process', methods=['POST'])
def process():
    """Enhanced process route with better error handling."""
    try:
        logger.info("Processing upload request")
        
        if 'image' not in request.files:
            logger.warning("No file uploaded")
            flash('No file uploaded')
            return redirect(url_for('index'))
        
        file = request.files['image']
        if file.filename == '':
            logger.warning("No file selected")
            flash('No selected file')
            return redirect(url_for('index'))

        # Get selected language (if implemented)
        language = request.form.get('language', 'en-US')
        logger.info(f"Processing file: {file.filename}, language: {language}")
            
        if file and allowed_file(file.filename):
            # Create unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]  # Include microseconds but limit length
            original_filename = secure_filename(file.filename)
            filename = f"{timestamp}_{original_filename}"
            
            # Save uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                logger.info(f"File saved to: {filepath}")
                
                # Verify file was saved correctly
                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    logger.error("File was not saved correctly")
                    flash('Error saving uploaded file')
                    return redirect(url_for('index'))
                    
            except Exception as e:
                logger.error(f"Error saving file: {e}")
                flash('Error saving uploaded file')
                return redirect(url_for('index'))
            
            # Check if it's a PDF
            if is_pdf_file(file.filename):
                logger.info("Processing PDF file")
                
                if not PYMUPDF_AVAILABLE:
                    flash('PDF processing is not available on this server')
                    return redirect(url_for('index'))
                
                try:
                    # Process PDF
                    results = process_pdf_book(
                        filepath, 
                        app.config['PDF_IMAGES_FOLDER'], 
                        audio_folder=app.config['AUDIO_FOLDER']
                    )
                    
                    if results and results.get('pages'):
                        # Create URLs for the results
                        for page in results['pages']:
                            # For hosted platforms, we need to handle file serving differently
                            if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RENDER'):
                                # Store file path for custom serving
                                page['image_url'] = f"/serve_temp_file/{os.path.basename(page['image_path'])}"
                                if page['audio_filename']:
                                    page['audio_url'] = f"/serve_temp_file/{page['audio_filename']}"
                                else:
                                    page['audio_url'] = None
                            else:
                                # Local development - use static URLs
                                page['image_url'] = url_for('static', filename=f"pdf_images/{page['image_filename']}")
                                if page['audio_filename']:
                                    page['audio_url'] = url_for('static', filename=f"audio/{page['audio_filename']}")
                                else:
                                    page['audio_url'] = None
                        
                        logger.info(f"Rendering PDF results with {len(results['pages'])} pages")
                        return render_template('index.html', 
                                             pdf_results=results,
                                             file_type='pdf',
                                             pdf_supported=PYMUPDF_AVAILABLE)
                    else:
                        logger.error("PDF processing returned no results")
                        flash('Failed to process PDF. Please ensure it contains readable images.')
                        return redirect(url_for('index'))
                        
                except Exception as e:
                    logger.error(f"Error processing PDF: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    flash('Error processing PDF. Please try a different file.')
                    return redirect(url_for('index'))
                    
            else:
                logger.info("Processing single image file")
                try:
                    # Process single image
                    result = process_image(filepath, audio_folder=app.config['AUDIO_FOLDER'])
                    
                    if result and result.get('description'):
                        # Create URLs for static files
                        if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RENDER'):
                            image_url = f"/serve_temp_file/{filename}"
                        else:
                            image_url = url_for('static', filename=f'uploads/{filename}')
                        
                        # Handle audio file URL
                        audio_url = None
                        if result.get('audio_path'):
                            audio_filename = os.path.basename(result['audio_path'])
                            if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RENDER'):
                                audio_url = f"/serve_temp_file/{audio_filename}"
                            else:
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
                                            file_type='image',
                                            pdf_supported=PYMUPDF_AVAILABLE)
                    else:
                        logger.error("Image processing returned no results")
                        flash('Failed to process image. Please try another file.')
                        return redirect(url_for('index'))
                        
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    flash('Error processing image. Please try a different file.')
                    return redirect(url_for('index'))
            
        else:
            logger.warning(f"Invalid file format: {file.filename}")
            flash('Invalid file format. Please upload an image' + (' or PDF' if PYMUPDF_AVAILABLE else '') + ' file.')
            return redirect(url_for('index'))
                
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('index'))

@app.route('/serve_temp_file/<filename>')
def serve_temp_file(filename):
    """Serve temporary files for hosted platforms."""
    try:
        # Try different folders
        for folder in [app.config['UPLOAD_FOLDER'], app.config['AUDIO_FOLDER'], app.config['PDF_IMAGES_FOLDER']]:
            file_path = os.path.join(folder, filename)
            if os.path.exists(file_path):
                return send_from_directory(folder, filename)
        
        # If not found, try searching recursively
        for root, dirs, files in os.walk(app.config['PDF_IMAGES_FOLDER']):
            if filename in files:
                return send_from_directory(root, filename)
        
        logger.error(f"Temp file not found: {filename}")
        return "File not found", 404
        
    except Exception as e:
        logger.error(f"Error serving temp file {filename}: {e}")
        return "Error serving file", 500

# Keep existing routes for backward compatibility
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    try:
        return send_from_directory(app.static_folder, filename)
    except FileNotFoundError:
        logger.error(f"Static file not found: {filename}")
        return "File not found", 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    try:
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
    """Enhanced health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pdf_support': PYMUPDF_AVAILABLE,
        'python_version': sys.version,
        'platform': sys.platform,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'audio_folder': app.config['AUDIO_FOLDER'],
        'pdf_images_folder': app.config['PDF_IMAGES_FOLDER'],
        'base_dir': BASE_DIR,
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'audio_folder_exists': os.path.exists(app.config['AUDIO_FOLDER']),
        'pdf_images_folder_exists': os.path.exists(app.config['PDF_IMAGES_FOLDER']),
        'environment': {
            'RAILWAY_ENVIRONMENT': os.environ.get('RAILWAY_ENVIRONMENT'),
            'RENDER': os.environ.get('RENDER'),
            'PORT': os.environ.get('PORT')
        }
    })

def run_web(host='0.0.0.0', port=None, debug=False):
    """
    Enhanced web server startup for hosted platforms.
    """
    # Use PORT environment variable for hosted platforms
    if port is None:
        port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting web server on {host}:{port}")
    logger.info(f"PDF support: {'✓ Enabled' if PYMUPDF_AVAILABLE else '✗ Disabled'}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    if os.environ.get('RAILWAY_ENVIRONMENT'):
        logger.info("Running on Railway")
    elif os.environ.get('RENDER'):
        logger.info("Running on Render")
    else:
        logger.info("Running locally")
    
    print(f"\n🌟 Book Narrator starting...")
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📖 PDF Support: {'✓ Available' if PYMUPDF_AVAILABLE else '✗ Not Available'}")
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    import sys
    
    # Simple argument parsing for hosted platforms
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run tests
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        # Run web server
        debug_mode = '--debug' in sys.argv or os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        run_web(debug=debug_mode)