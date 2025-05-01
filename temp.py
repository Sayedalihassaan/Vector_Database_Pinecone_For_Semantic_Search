import os
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

list_of_files = [
    r"README.md",
    r"requirements.txt",
    r".env",
    r".env.example",
    r"Backend.py",
    r".gitignore",
    r"src/__init__.py",
    r"src/config.py",
    r"src/__pycache__",
    r"src/notebooks",
    r"src/notebooks/data",
    r"src/notebooks/pinecone-semantic.ipynb",
    r"src/services/__init__.py",
    r"src/services/EmbeddingModel.py",
    r"src/services/PineconeClient.py",       
    r"src/services/VectorDBController.py", 
    r"src/services/__pycache__",
]



def create_project_structure():
    for filepath in list_of_files:
        filepath = Path(filepath)
        
        file_dir = filepath.parent
        file_name = filepath.name
        
        if str(file_dir) != ".":
            if os.path.exists(file_dir) and not os.path.isdir(file_dir):
                logging.error(f"Cannot create directory '{file_dir}': A file with the same name exists")
                continue
                
            try:
                os.makedirs(file_dir, exist_ok=True)
                logging.info(f"Ensured directory exists: {file_dir}")
            except Exception as e:
                logging.error(f"Error creating directory '{file_dir}': {str(e)}")
                continue
        
        if file_name:
            if os.path.exists(filepath):
                if os.path.isdir(filepath):
                    logging.info(f"Path exists as directory: {filepath}")
                else:
                    logging.info(f"File already exists: {filepath}")
            else:
                try:
                    with open(filepath, "w") as f:
                        pass 
                    logging.info(f"Created empty file: {filepath}")
                except Exception as e:
                    logging.error(f"Error creating file '{filepath}': {str(e)}")
        else:
            logging.info(f"Directory structure ensured: {filepath}")

if __name__ == "__main__":
    try:
        create_project_structure()
        logging.info("Project structure setup completed")
    except Exception as e:
        logging.error(f"Unhandled error: {str(e)}")
        sys.exit(1)