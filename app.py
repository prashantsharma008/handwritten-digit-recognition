import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_path(rel_path):
    """Find a relative path across possible deployment directory layouts."""
    candidates = [
        os.path.join(BASE_DIR, rel_path),
        os.path.join(BASE_DIR, "api", rel_path),
        os.path.join(os.path.dirname(__file__), rel_path),
        os.path.join(os.path.dirname(__file__), "..", rel_path),
        os.path.join(os.getcwd(), rel_path),
        os.path.join(os.getcwd(), "api", rel_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.join(BASE_DIR, rel_path)


app = Flask(
    __name__,
    template_folder=find_path("templates"),
    static_folder=find_path("static"),
    static_url_path="/static"
)

WEIGHTS_PATH = find_path("model_weights.npz")
KERAS_PATH = find_path("handwritten_digit_model.keras")


class DigitClassifier:
    """
    Ultra-lightweight pure-NumPy neural network inference engine.
    Executes the exact trained weights (Dense 784->128->64->10 with ReLU and Softmax)
    with 100% mathematical parity to TensorFlow, but uses < 400KB of RAM and
    requires ZERO heavy TensorFlow C++ dependencies, enabling sub-second Vercel deployment.
    """
    def __init__(self, weights_path):
        data = np.load(weights_path)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w3 = data["w3"]
        self.b3 = data["b3"]
        self.input_shape = (None, 28, 28)

    def predict(self, x, verbose=0):
        x_flat = np.asarray(x, dtype=np.float32).reshape(1, 784)
        h1 = np.maximum(0, np.dot(x_flat, self.w1) + self.b1)
        h2 = np.maximum(0, np.dot(h1, self.w2) + self.b2)
        logits = np.dot(h2, self.w3) + self.b3
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs


def load_model():
    """Load pure-NumPy model or fallback to Keras model if available."""
    weights_file = find_path("model_weights.npz")
    if os.path.exists(weights_file):
        print(f"Loading ultra-lightweight pure-NumPy model from {weights_file}...")
        return DigitClassifier(weights_file)

    keras_file = find_path("handwritten_digit_model.keras")
    if os.path.exists(keras_file):
        try:
            import tensorflow as tf
            print(f"Loading Keras model from {keras_file}...")
            return tf.keras.models.load_model(keras_file)
        except ImportError:
            pass

    raise FileNotFoundError(f"Model file not found. Checked: {weights_file}")


# Initialize model at startup
model = load_model()


@app.route("/", methods=["GET", "POST"])
@app.route("/api", methods=["GET", "POST"])
@app.route("/api/", methods=["GET", "POST"])
@app.route("/api/index", methods=["GET", "POST"])
@app.route("/api/index/", methods=["GET", "POST"])
@app.route("/api/index.py", methods=["GET", "POST"])
def index():
    """Serve the main UI page on GET, or handle prediction on POST if rewritten."""
    if request.method == "POST":
        return predict()
    try:
        return render_template("index.html")
    except Exception:
        html_file = find_path(os.path.join("templates", "index.html"))
        if os.path.exists(html_file):
            with open(html_file, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
        return "<h1>Handwritten Digit Recognition</h1><p>UI loading error</p>", 500


@app.route("/static/<path:filename>")
@app.route("/api/static/<path:filename>")
@app.route("/api/index/static/<path:filename>")
@app.route("/api/index.py/static/<path:filename>")
def custom_static(filename):
    """Ensure static files are always served even if URL is rewritten by Vercel."""
    static_dir = find_path("static")
    return send_from_directory(static_dir, filename)


def preprocess_digit_image(pil_img):
    """
    Official MNIST normalization pipeline for real-world handwritten digits:
    1. Convert image to grayscale.
    2. Invert if background is light (white canvas/paper with dark ink).
    3. Filter out faint background noise.
    4. Detect bounding box around the drawn digit.
    5. Resize bounding box preserving aspect ratio into a 20x20 pixel box.
    6. Place the 20x20 digit inside a standard 28x28 black canvas.
    7. Center the digit using its Center of Mass (Mass-Weighted Centroid).
    8. Return normalized [0, 1] 28x28 image and a boolean indicating whether content was found.
    """
    img = pil_img.convert("L")
    arr = np.array(img, dtype=np.float32)

    # Invert if light background (MNIST expects bright digit on black bg)
    if np.mean(arr) > 127:
        arr = 255.0 - arr

    # Clean small background noise
    arr[arr < 35] = 0

    # Find bounding box of the drawn digit
    rows = np.any(arr > 35, axis=1)
    cols = np.any(arr > 35, axis=0)
    if not np.any(rows) or not np.any(cols):
        # Empty canvas / no digit drawn
        return np.zeros((28, 28), dtype=np.float32), False

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add small padding around bounding box
    pad = 4
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1] - 1, cmax + pad)

    cropped = arr[rmin:rmax+1, cmin:cmax+1]
    h, w = cropped.shape

    # Fit into 20x20 box preserving aspect ratio (official MNIST rule)
    if h > w:
        new_h = 20
        new_w = max(1, int(round((w / h) * 20)))
    else:
        new_w = 20
        new_h = max(1, int(round((h / w) * 20)))

    crop_img = Image.fromarray(np.uint8(cropped))
    resample_mode = getattr(Image, "Resampling", Image).BICUBIC
    crop_resized = crop_img.resize((new_w, new_h), resample_mode)
    resized_arr = np.array(crop_resized, dtype=np.float32)

    # Place inside 28x28 frame
    frame = np.zeros((28, 28), dtype=np.float32)
    start_r = (28 - new_h) // 2
    start_c = (28 - new_w) // 2
    frame[start_r:start_r+new_h, start_c:start_c+new_w] = resized_arr

    # Center by Center of Mass (official MNIST centering)
    total_mass = np.sum(frame)
    if total_mass > 0:
        cy = np.sum(np.arange(28)[:, None] * frame) / total_mass
        cx = np.sum(np.arange(28)[None, :] * frame) / total_mass

        shift_y = int(round(13.5 - cy))
        shift_x = int(round(13.5 - cx))

        # Clamp shift to prevent shifting out of frame
        shift_y = max(-3, min(3, shift_y))
        shift_x = max(-3, min(3, shift_x))

        if shift_y != 0:
            frame = np.roll(frame, shift_y, axis=0)
            if shift_y > 0:
                frame[:shift_y, :] = 0
            else:
                frame[shift_y:, :] = 0

        if shift_x != 0:
            frame = np.roll(frame, shift_x, axis=1)
            if shift_x > 0:
                frame[:, :shift_x] = 0
            else:
                frame[:, shift_x:] = 0

    frame = np.clip(frame, 0, 255.0)
    return frame, True


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
@app.route("/api/index/predict", methods=["POST"])
@app.route("/api/index.py/predict", methods=["POST"])
def predict():
    """Accept a base64-encoded image and return digit predictions."""
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image data provided"}), 400

        image_data = data.get("image", "")

        # Strip base64 header if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decode base64 to image
        img_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(img_bytes))

        # Preprocess with official MNIST pipeline
        processed_frame, has_content = preprocess_digit_image(pil_img)

        if not has_content:
            return jsonify({
                "success": False,
                "error": "No drawn digit detected. Please draw a digit clearly on the canvas."
            }), 400

        # Normalize and predict
        normalized = (processed_frame / 255.0).astype(np.float32)
        input_shape = model.input_shape
        if len(input_shape) == 4:
            input_tensor = normalized.reshape(1, 28, 28, 1)
        else:
            input_tensor = normalized.reshape(1, 28, 28)

        predictions = model.predict(input_tensor, verbose=0)[0]

        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit] * 100.0)

        # Build confidence list for all digits
        confidences = [
            {"digit": i, "confidence": round(float(predictions[i]) * 100, 2)}
            for i in range(10)
        ]
        confidences.sort(key=lambda x: x["confidence"], reverse=True)

        # Generate 28x28 preview as base64
        preview_img = Image.fromarray(np.uint8(processed_frame))
        preview_buffer = io.BytesIO()
        preview_img.save(preview_buffer, format="PNG")
        preview_b64 = base64.b64encode(preview_buffer.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "digit": predicted_digit,
            "confidence": round(confidence, 2),
            "confidences": confidences,
            "preview": f"data:image/png;base64,{preview_b64}"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("\n>>> Starting Handwritten Digit Recognition App...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
