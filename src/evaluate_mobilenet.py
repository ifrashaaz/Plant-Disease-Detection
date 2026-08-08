import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import load_dataset

# -----------------------------
# Settings
# -----------------------------

TEST_CSV = "outputs/test.csv"
MODEL_PATH = "models/mobilenetv2.keras"

# -----------------------------
# Load test dataset
# -----------------------------

test_ds, class_names = load_dataset(
    TEST_CSV,
    training=False
)

print("Number of classes:", len(class_names))

# -----------------------------
# Load trained MobileNetV2
# -----------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")

# -----------------------------
# Evaluate model
# -----------------------------

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# -----------------------------
# Get predictions
# -----------------------------

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# -----------------------------
# Classification report
# -----------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# -----------------------------
# Confusion matrix
# -----------------------------

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)