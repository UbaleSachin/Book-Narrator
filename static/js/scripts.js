const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const uploadForm = document.getElementById('uploadForm');
const loading = document.getElementById('loading');
const processButton = document.getElementById('processButton');
const uploadButton = document.getElementById('uploadButton');
const processingMessage = document.getElementById('processingMessage');
const imageOption = document.getElementById('imageOption');
const pdfOption = document.getElementById('pdfOption');

let selectedFileType = null;

// Only initialize upload controls if they exist (not on results pages)
if (fileInput) {
    // Handle upload option selection
    imageOption.addEventListener('click', function() {
        selectUploadType('image');
    });

    pdfOption.addEventListener('click', function() {
        selectUploadType('pdf');
    });

    function selectUploadType(type) {
        imageOption.classList.remove('active');
        pdfOption.classList.remove('active');
        
        if (type === 'image') {
            imageOption.classList.add('active');
            fileInput.accept = "image/*";
            uploadButton.innerHTML = "📷 Choose Image";
            selectedFileType = 'image';
        } else if (type === 'pdf') {
            pdfOption.classList.add('active');
            fileInput.accept = ".pdf";
            uploadButton.innerHTML = "📖 Choose PDF Book";
            selectedFileType = 'pdf';
        }
        
        // Clear previous file selection
        fileInput.value = '';
        fileInfo.textContent = '';
        processButton.disabled = true;
    }

    // Default selection
    selectUploadType('image');
    
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const fileSize = (file.size / 1024 / 1024).toFixed(2);
            const fileType = file.type.toLowerCase();
            const fileName = file.name.toLowerCase();
            
            let typeIndicator = '';
            let isValidFile = false;
            
            if (fileType.startsWith('image/') || fileName.match(/\.(jpg|jpeg|png|gif|bmp|webp)$/)) {
                typeIndicator = '<span class="file-type-indicator file-type-image">IMAGE</span>';
                isValidFile = true;
            } else if (fileType === 'application/pdf' || fileName.endsWith('.pdf')) {
                typeIndicator = '<span class="file-type-indicator file-type-pdf">PDF BOOK</span>';
                isValidFile = true;
            }
            
            if (isValidFile) {
                fileInfo.innerHTML = `Selected: <strong>${file.name}</strong> (${fileSize} MB) ${typeIndicator}`;
                processButton.disabled = false;
                
                // Show appropriate file size warning
                if (fileSize > 20) {
                    fileInfo.innerHTML += '<br><small style="color: #ff9800;">⚠️ Large file - processing may take longer</small>';
                }
            } else {
                fileInfo.innerHTML = '<span style="color: #f44336;">❌ Invalid file type. Please select an image or PDF file.</span>';
                processButton.disabled = true;
            }
        } else {
            processButton.disabled = true;
            fileInfo.textContent = '';
        }
    });
    
    uploadForm.addEventListener('submit', function(e) {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
            
            loading.style.display = 'block';
            document.querySelector('.upload-section form').style.display = 'none';
            
            // Update loading message and style based on file type
            if (isPdf) {
                loading.className = 'loading pdf-processing';
                processingMessage.innerHTML = `
                    <strong>📖 Processing PDF Picture Book...</strong><br>
                    <small>Extracting pages, analyzing images, and generating narrations.<br>
                    This may take several minutes for longer books.</small>
                `;
            } else {
                loading.className = 'loading image-processing';
                processingMessage.innerHTML = `
                    <strong>🖼️ Processing Image...</strong><br>
                    <small>Analyzing the image and creating audio narration.</small>
                `;
            }
        }
    });
}

// Add some interactive animations
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('.section');
    sections.forEach((section, index) => {
        section.style.animationDelay = `${index * 0.2}s`;
    });
});

// Add error handling for audio playback
document.addEventListener('DOMContentLoaded', function() {
    const audioElement = document.querySelector('audio');
    if (audioElement) {
        audioElement.onerror = function() {
            console.error('Error loading audio:', audioElement.error);
            alert('Error loading audio. Please try downloading the file instead.');
        };
    }
});

// File drag and drop functionality
const container = document.querySelector('.container');
let dragCounter = 0;

container.addEventListener('dragenter', function(e) {
    e.preventDefault();
    dragCounter++;
    container.style.transform = 'scale(1.02)';
    container.style.boxShadow = '0 25px 50px rgba(0, 0, 0, 0.2)';
});

container.addEventListener('dragleave', function(e) {
    e.preventDefault();
    dragCounter--;
    if (dragCounter === 0) {
        container.style.transform = 'scale(1)';
        container.style.boxShadow = '0 20px 40px rgba(0, 0, 0, 0.1)';
    }
});

container.addEventListener('dragover', function(e) {
    e.preventDefault();
});

container.addEventListener('drop', function(e) {
    e.preventDefault();
    dragCounter = 0;
    container.style.transform = 'scale(1)';
    container.style.boxShadow = '0 20px 40px rgba(0, 0, 0, 0.1)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        fileInput.dispatchEvent(new Event('change'));
    }
});