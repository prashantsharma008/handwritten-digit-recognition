import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Create folders for saved outputs
os.makedirs("outputs", exist_ok=True)

print("TensorFlow Version:", tf.__version__)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)

# Display sample images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title("Digit: " + str(y_train[i]))
    plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/sample_images.png")
plt.show()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/accuracy.png")
plt.show()

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/loss.png")
plt.show()

# Make predictions
predictions = model.predict(x_test, verbose=0)
predicted_digits = np.argmax(predictions, axis=1)

print("\nFirst 20 predicted digits:")
print(predicted_digits[:20])
print("\nFirst 20 actual digits:")
print(y_test[:20])

# Display predictions
plt.figure(figsize=(10, 6))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    predicted = predicted_digits[i]
    actual = y_test[i]
    plt.title(f"Predicted: {predicted}\nActual: {actual}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/predictions.png")
plt.show()

# Find incorrect predictions
incorrect = np.where(predicted_digits != y_test)[0]
print("\nTotal incorrect predictions:", len(incorrect))

if len(incorrect) > 0:
    plt.figure(figsize=(10, 6))
    for i in range(min(10, len(incorrect))):
        index = incorrect[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[index], cmap="gray")
        plt.title(
            f"Predicted: {predicted_digits[index]}\n"
            f"Actual: {y_test[index]}"
        )
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/incorrect_predictions.png")
    plt.show()

# Save trained model
model.save("handwritten_digit_model.keras")
print("\nModel saved as handwritten_digit_model.keras")

# Test loading the saved model
loaded_model = tf.keras.models.load_model("handwritten_digit_model.keras")
print("Model loaded successfully!")
