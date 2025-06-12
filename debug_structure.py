import os
import sys

def debug_file_structure():
    """Debug the file structure and paths for the Flask app."""
    
    print("=== DEBUGGING FILE STRUCTURE ===\n")
    
    # Current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Check main script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    
    # Expected directories
    directories_to_check = [
        'templates',
        'output',
        'output/uploads',
        'output/audio',
        'src',
        'src/llm',
        'src/utils',
        'data'
    ]
    
    print(f"\n=== DIRECTORY STRUCTURE ===")
    for directory in directories_to_check:
        full_path = os.path.join(cwd, directory)
        exists = os.path.exists(full_path)
        print(f"{'✓' if exists else '✗'} {directory:<20} -> {full_path}")
        
        # If it's the templates directory, check for index.html
        if directory == 'templates' and exists:
            index_path = os.path.join(full_path, 'index.html')
            index_exists = os.path.exists(index_path)
            print(f"  {'✓' if index_exists else '✗'} index.html")
    
    # Check for key files
    print(f"\n=== KEY FILES ===")
    key_files = [
        'templates/index.html',
        'src/__init__.py',
        'src/llm/__init__.py',
        'src/llm/base_model.py',
        'src/utils/__init__.py',
        'src/utils/image_utils.py'
    ]
    
    for file_path in key_files:
        full_path = os.path.join(cwd, file_path)
        exists = os.path.exists(full_path)
        print(f"{'✓' if exists else '✗'} {file_path}")
    
    # Look for possible main files
    print(f"\n=== LOOKING FOR MAIN FILES ===")
    possible_main_files = []
    for file in os.listdir(cwd):
        if file.endswith('.py') and ('main' in file.lower() or 'app' in file.lower() or 'run' in file.lower()):
            possible_main_files.append(file)
    
    if possible_main_files:
        print("Found possible main files:")
        for file in possible_main_files:
            print(f"  ✓ {file}")
    else:
        print("No main files found (looking for files containing 'main', 'app', or 'run')")
    
    # List all Python files in current directory
    print(f"\n=== ALL PYTHON FILES IN CURRENT DIRECTORY ===")
    python_files = [f for f in os.listdir(cwd) if f.endswith('.py')]
    if python_files:
        for file in python_files:
            print(f"  {file}")
    else:
        print("No Python files found in current directory")
    
    # Create missing directories
    print(f"\n=== CREATING MISSING DIRECTORIES ===")
    for directory in directories_to_check:
        full_path = os.path.join(cwd, directory)
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path, exist_ok=True)
                print(f"✓ Created: {directory}")
            except Exception as e:
                print(f"✗ Failed to create {directory}: {e}")
    
    # Test file creation in upload directory
    print(f"\n=== TESTING FILE OPERATIONS ===")
    upload_dir = os.path.join(cwd, 'output', 'uploads')
    test_file = os.path.join(upload_dir, 'test_file.txt')
    
    try:
        with open(test_file, 'w') as f:
            f.write('Test file for upload directory')
        
        if os.path.exists(test_file):
            print("✓ Can write to upload directory")
            os.remove(test_file)  # Clean up
            print("✓ Can delete from upload directory")
        else:
            print("✗ File write failed")
            
    except Exception as e:
        print(f"✗ Upload directory test failed: {e}")
    
    # Flask app paths
    print(f"\n=== FLASK APP PATHS ===")
    template_folder = os.path.join(cwd, 'templates')
    static_folder = os.path.join(cwd, 'output')
    upload_folder = os.path.join(cwd, 'output', 'uploads')
    audio_folder = os.path.join(cwd, 'output', 'audio')
    
    print(f"Templates folder: {template_folder}")
    print(f"Static folder: {static_folder}")
    print(f"Upload folder: {upload_folder}")
    print(f"Audio folder: {audio_folder}")
    
    # Absolute paths
    print(f"\n=== ABSOLUTE PATHS ===")
    print(f"Templates (abs): {os.path.abspath(template_folder)}")
    print(f"Static (abs): {os.path.abspath(static_folder)}")
    print(f"Upload (abs): {os.path.abspath(upload_folder)}")
    print(f"Audio (abs): {os.path.abspath(audio_folder)}")

if __name__ == '__main__':
    debug_file_structure()