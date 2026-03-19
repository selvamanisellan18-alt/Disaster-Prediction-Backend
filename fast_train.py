import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- CONFIGURATION ---
DATASET_DIR = "dataset/train"
MODEL_SAVE_PATH = "model/disaster_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Verum 5 rounds thaan, so romba fast-ah mudiyum!

print("🚀 Starting Fast Training Pipeline...")

# 1. Load Data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 2. Build Fast Model (MobileNetV2)
print("⚙️ Building AI Architecture...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Pazhaya layers-ah freeze pandrom for SPEED

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train the Model
print(f"🔥 Training on {train_gen.samples} images... This will be quick!")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# 4. Save Model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"✅ Fast Training Complete! New perfect model saved to {MODEL_SAVE_PATH}")