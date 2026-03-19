import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, sys

# Path fix to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = 'dataset/train' # Un dataset folder path
MODEL_PATH = 'models/disaster_model.h5'

def build_cnn_model():
    model = models.Sequential([
        # Feature Extraction
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification
        layers.Flatten(),
        layers.Dense(256, activation='relu'), # Dense layer size-ah boost panniruken
        layers.Dropout(0.5), 
        layers.Dense(4, activation='softmax') # flood, fire, cyclone, normal
    ])
    
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def start_training():
    print("[INFO] AI Factory: Loading Data with Augmentation...")

    # Confidence booster: Data Augmentation logic
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,      # Image-ah light-ah rotate pannum
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,         # Zoom panni variations create pannum
        horizontal_flip=True,   # Mirror effect
        fill_mode='nearest'
    )

    # Dataset loading
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model = build_cnn_model()
    model.summary()

    # Accuracy improve aagala na stop panna patience-ah 5-ah mathuna nalla irukkum
    early_stop = callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

    print(f"[INFO] Training starting for {train_generator.samples} images...")
    
    # 50 Epochs train panna high confidence kidaikkum
    model.fit(
        train_generator, 
        epochs=50,
        callbacks=[early_stop]
    ) 
    
    # Save the updated brain
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n[SUCCESS] AI Brain with High Accuracy saved at: {MODEL_PATH}")

if __name__ == "__main__":
    start_training()

    