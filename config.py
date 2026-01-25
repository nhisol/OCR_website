import os 
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

MODEL_PATH = os.path.join(BASE_DIR, 'model')
PROCESSOR_PATH = os.path.join(BASE_DIR, 'processor')

MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

TORCH_DEVICE = 'cuda' if os.getenv('USE_CUDA', 'False') == 'True' else 'cpu'