import os

# Project root directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'disaster_model.h5')

# Hyperparameters for the AI
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# These must match your 4 folder names exactly [cite: 79, 80]
CLASSES = ['cyclone', 'fire', 'flood', 'normal']