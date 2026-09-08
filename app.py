import os
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

MODEL_PATH = "handwritten_digit_model.keras"

def get_or_train_model():
    """Load existing model or automatically train one if missing."""
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading existing model from {MODEL_PATH}...")
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading saved model ({e}). Training a fresh model...")
    
    print("No saved model found. Training model on MNIST dataset...")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
    model.save(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return model

# Initialize TensorFlow model
model = get_or_train_model()

def process_and_predict(image_input):
    """Processes input image (uploaded or drawn) and predicts digit."""
    if image_input is None:
        return (
            "<div class='placeholder-badge'>Upload or draw an image above</div>",
            {},
            None
        )

    # Extract image array from input (handles dict/composite from Sketchpad or direct numpy from Upload)
    if isinstance(image_input, dict):
        img_data = image_input.get("composite", image_input.get("background", None))
        if img_data is None and "layers" in image_input and len(image_input["layers"]) > 0:
            img_data = image_input["layers"][0]
    else:
        img_data = image_input

    if img_data is None:
        return (
            "<div class='placeholder-badge'>No valid image detected</div>",
            {},
            None
        )

    # Convert to PIL Grayscale Image
    pil_img = Image.fromarray(np.uint8(img_data)).convert("L")

    # Resize to 28x28 pixels safely across Pillow versions
    resample_mode = getattr(Image, 'Resampling', Image).LANCZOS
    pil_img_28 = pil_img.resize((28, 28), resample_mode)
    img_array = np.array(pil_img_28, dtype=np.float32)

    # Invert background if canvas is light (MNIST expects white digit on black background)
    if np.mean(img_array) > 127:
        img_array = 255.0 - img_array

    # Preview image as 28x28 for UI visualization
    preview_img = Image.fromarray(np.uint8(img_array)).resize((140, 140), Image.Resampling.NEAREST)

    # Normalize pixel values to [0.0, 1.0]
    normalized_array = img_array / 255.0
    input_tensor = normalized_array.reshape(1, 28, 28)

    # Predict probabilities with TensorFlow model
    predictions = model.predict(input_tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(predictions))
    confidence_pct = float(predictions[predicted_digit] * 100.0)

    # Format result dictionary for Gradio Label component
    confidence_dict = {str(i): float(predictions[i]) for i in range(10)}

    # Create stylish result badge HTML
    badge_html = f"""
    <div class="result-card">
        <div class="result-title">RECOGNIZED DIGIT</div>
        <div class="digit-display">{predicted_digit}</div>
        <div class="confidence-tag">Confidence: <strong>{confidence_pct:.1f}%</strong></div>
    </div>
    """

    return badge_html, confidence_dict, preview_img


# Custom CSS styling for premium look & feel
custom_css = """
.container { max-width: 900px; margin: auto; padding: 10px; }
.header-box { text-align: center; margin-bottom: 20px; }
.header-box h1 { font-size: 2.2rem; font-weight: 700; color: #3B82F6; margin-bottom: 6px; }
.header-box p { font-size: 1rem; color: #6B7280; }

.result-card {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    color: white;
    padding: 24px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
    margin-bottom: 15px;
}
.result-title { font-size: 0.85rem; letter-spacing: 1.5px; opacity: 0.9; font-weight: 600; }
.digit-display { font-size: 4.5rem; font-weight: 800; line-height: 1; margin: 10px 0; }
.confidence-tag { font-size: 1.05rem; background: rgba(255, 255, 255, 0.2); padding: 6px 14px; border-radius: 20px; display: inline-block; }

.placeholder-badge {
    text-align: center;
    padding: 30px;
    background: #F3F4F6;
    border: 2px dashed #D1D5DB;
    border-radius: 16px;
    color: #6B7280;
    font-size: 1rem;
    margin-bottom: 15px;
}

footer { visibility: hidden; }
"""

# Build Gradio UI with Tabs for Upload and Canvas
with gr.Blocks(title="Handwritten Digit Classifier") as demo:
    gr.HTML(
        """
        <div class="header-box">
            <h1>🧠 Handwritten Digit Recognition</h1>
            <p>Upload an image or draw a digit (0 to 9) to classify it using Deep Learning</p>
        </div>
        """
    )
    
    with gr.Row():
        # Left Column: Inputs (Tabs for Upload & Drawing)
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("🖼️ Upload Image"):
                    file_input = gr.Image(
                        label="Upload Digit Image (PNG / JPG / JPEG)",
                        type="numpy",
                        sources=["upload", "clipboard"]
                    )
                    upload_btn = gr.Button("Recognize Uploaded Image 🚀", variant="primary")
                    upload_clear = gr.Button("Clear Image 🧹")

                with gr.TabItem("✍️ Draw Canvas"):
                    sketch_input = gr.Sketchpad(
                        label="Draw Digit (0-9)",
                        type="numpy"
                    )
                    sketch_btn = gr.Button("Recognize Drawn Digit 🚀", variant="primary")
                    sketch_clear = gr.Button("Clear Canvas 🧹")

        # Right Column: Output Predictions & Preprocessed Preview
        with gr.Column(scale=1):
            badge_output = gr.HTML(
                value="<div class='placeholder-badge'>Upload or draw an image on the left to see results</div>"
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    preview_output = gr.Image(
                        label="28x28 Model Input View",
                        type="pil",
                        interactive=False,
                        height=160
                    )
                with gr.Column(scale=2):
                    label_output = gr.Label(
                        num_top_classes=3,
                        label="Top Predictions"
                    )

    # Event Handlers for Upload Tab
    upload_btn.click(
        fn=process_and_predict,
        inputs=file_input,
        outputs=[badge_output, label_output, preview_output]
    )
    file_input.change(
        fn=process_and_predict,
        inputs=file_input,
        outputs=[badge_output, label_output, preview_output]
    )
    upload_clear.click(
        fn=lambda: (None, "<div class='placeholder-badge'>Upload or draw an image on the left to see results</div>", {}, None),
        inputs=None,
        outputs=[file_input, badge_output, label_output, preview_output]
    )

    # Event Handlers for Draw Canvas Tab
    sketch_btn.click(
        fn=process_and_predict,
        inputs=sketch_input,
        outputs=[badge_output, label_output, preview_output]
    )
    sketch_input.change(
        fn=process_and_predict,
        inputs=sketch_input,
        outputs=[badge_output, label_output, preview_output]
    )
    sketch_clear.click(
        fn=lambda: (None, "<div class='placeholder-badge'>Upload or draw an image on the left to see results</div>", {}, None),
        inputs=None,
        outputs=[sketch_input, badge_output, label_output, preview_output]
    )

if __name__ == "__main__":
    demo.launch(css=custom_css)
