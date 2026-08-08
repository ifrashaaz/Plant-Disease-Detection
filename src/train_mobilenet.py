import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from preprocessing import load_dataset

# -----------------------------
# Settings
# -----------------------------

TRAIN_CSV = "outputs/train.csv"
VAL_CSV = "outputs/validation.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5

# -----------------------------
# Load datasets
# -----------------------------

train_ds, class_names = load_dataset(
    TRAIN_CSV,
    training=True
)

val_ds, _ = load_dataset(
    VAL_CSV,
    training=False
)

print("Number of classes:", len(class_names))
print("Classes:", class_names)

# -----------------------------
# Build MobileNetV2
# -----------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained layers
base_model.trainable = False

# -----------------------------
# Create our model
# -----------------------------

model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.2),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# -----------------------------
# Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Display model information
# -----------------------------

model.summary()

# -----------------------------
# Train
# -----------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# Save model
# -----------------------------

model.save("models/mobilenetv2.keras")

print("\nMobileNetV2 training completed!")
print("Model saved to models/mobilenetv2.keras")