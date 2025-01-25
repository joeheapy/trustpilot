import shutil
from datetime import datetime
import os

# This deletes and recreates the working directories to ensure a clean start
def initialize_directories():
    """Initialize working directories by removing and recreating them"""
    directories = [
        'analyzed-chunks',
        'data-chunks',
        'summarized-reviews',
        'journey-steps',
        'reviews-by-journey-step'
    ]
    
    print(f"\nInitializing directories at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for directory in directories:
        try:
            # Remove directory and contents if exists
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Removed existing directory: {directory}")
            
            # Create new empty directory
            os.makedirs(directory)
            print(f"Created new directory: {directory}")
            
        except Exception as e:
            print(f"Error processing directory {directory}: {str(e)}")
            raise