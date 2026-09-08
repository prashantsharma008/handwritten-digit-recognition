# Handwritten Digit Recognition Using Deep Learning & Gradio Web App

A deep learning project using TensorFlow/Keras and the MNIST dataset to recognize handwritten digits (0 to 9), featuring an interactive **Gradio drawing web app** that can be deployed online for free.

---

## 📁 Project Files

- `handwritten_digit_recognition.py` - Trains the deep learning model and saves `handwritten_digit_model.keras`.
- `app.py` - Interactive Gradio web interface with a drawing canvas to draw digits and test model predictions in real time.
- `requirements.txt` - Dependencies (`tensorflow`, `numpy`, `matplotlib`, `gradio`, `pillow`).

---

## 🛠️ Local Installation & Setup

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train & Save Model:**
   ```bash
   python handwritten_digit_recognition.py
   ```
   *This trains the neural network on MNIST dataset and generates `handwritten_digit_model.keras`.*

3. **Run the Gradio Web App locally:**
   ```bash
   python app.py
   ```
   *Open the printed local URL (e.g. `http://127.0.0.1:7860`) in your web browser to test drawing digits.*

---

## 🌐 How to Host Online for FREE (Hugging Face Spaces)

You can host this project on **Hugging Face Spaces** for free with zero server maintenance.

### Step 1: Create a Hugging Face Account & Space
1. Go to [huggingface.co](https://huggingface.co/) and sign up or log in.
2. Click on your profile picture at the top right and select **New Space** (or visit [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Fill in the details:
   - **Space name**: `handwritten-digit-recognition` (or any name you prefer)
   - **License**: `mit`
   - **Select Space SDK**: **Gradio**
   - **Space hardware**: **CPU Basic (Free)**
4. Click **Create Space**.

### Step 2: Upload Files to the Space
You can upload files via web browser or Git:

#### Method A: Direct Web Upload (Easiest)
1. Inside your new Space page, click the **Files** tab.
2. Click **Add file** -> **Upload files**.
3. Drag & drop the following 3 files:
   - `app.py`
   - `requirements.txt`
   - `handwritten_digit_model.keras` (generated after running `handwritten_digit_recognition.py`)
4. Click **Commit changes to main**.

#### Method B: Git Clone & Push
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/handwritten-digit-recognition
cd handwritten-digit-recognition
# Copy app.py, requirements.txt, and handwritten_digit_model.keras into this folder
git add .
git commit -m "Deploy Gradio app with TensorFlow model"
git push
```

### Step 3: View Your Live Web App!
Hugging Face will automatically install `requirements.txt`, launch `app.py`, and give you a **public URL** (e.g., `https://huggingface.co/spaces/YOUR_USERNAME/handwritten-digit-recognition`) that anyone can open to draw digits and test predictions!

---

## 🧠 Model Architecture

- Input: 28 x 28 grayscale image
- Flatten Layer
- Dense: 128 neurons, ReLU activation
- Dropout: 20%
- Dense: 64 neurons, ReLU activation
- Output: 10 neurons, Softmax activation
