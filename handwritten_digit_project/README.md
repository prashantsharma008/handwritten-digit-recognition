# Handwritten Digit Recognition Using Deep Learning

A simple deep learning project using TensorFlow/Keras and the MNIST dataset to recognize handwritten digits from 0 to 9.

## Requirements

- Python 3.9-3.12 recommended
- TensorFlow
- NumPy
- Matplotlib

## Installation

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python handwritten_digit_recognition.py
```

The first run may take some time because TensorFlow downloads the MNIST dataset automatically.

## Project workflow

1. Load the MNIST dataset.
2. Display sample handwritten digits.
3. Normalize pixel values from 0-255 to 0-1.
4. Build a neural network.
5. Train the model for 10 epochs.
6. Evaluate the model on test data.
7. Plot accuracy and loss.
8. Predict handwritten digits.
9. Display incorrect predictions.
10. Save the trained model as `handwritten_digit_model.keras`.

## Model architecture

- Input: 28 x 28 grayscale image
- Flatten
- Dense: 128 neurons, ReLU
- Dropout: 20%
- Dense: 64 neurons, ReLU
- Output: 10 neurons, Softmax

## Output files

After running the program, the following files are created:

- `handwritten_digit_model.keras`
- `outputs/sample_images.png`
- `outputs/accuracy.png`
- `outputs/loss.png`
- `outputs/predictions.png`
- `outputs/incorrect_predictions.png` (when incorrect predictions exist)

## Project conclusion

The neural network learns patterns from the MNIST handwritten digit dataset and can classify unseen digit images with high accuracy. The exact test accuracy can vary slightly between runs.
