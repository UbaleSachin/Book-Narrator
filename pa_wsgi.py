import os
import sys

# Add your project directory to Python path
project_home = '/home/yourusername/Book-Narrator'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Import your Flask app
from src.main import app as application

# PythonAnywhere looks for an 'application' object by default