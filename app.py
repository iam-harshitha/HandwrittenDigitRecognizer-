import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from cnn import CNN  # Import your trained model

# ---------------------
# App Config
# ---------------------
st.set_page_config(page_title="Handwritten Digit Recognizer", layout="centered")
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🎨 Handwritten Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #333;'>Draw a digit (0–9) and let the AI guess!</h5>", unsafe_allow_html=True)
st.divider()

# ---------------------
# Load Model
# ---------------------
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# ---------------------
# Canvas Configuration
# ---------------------
canvas_result = st_canvas(
    fill_color="#000000",  # Black ink
    stroke_width=10,
    stroke_color="#FFFFFF",  # White pen on black canvas
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------------
# Preprocessing Function
# ---------------------
def preprocess_canvas_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)
    image = 255 - image  # Invert colors
    image = image / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    image = (image - 0.1307) / 0.3081  # Normalize as MNIST
    return image

# ---------------------
# Prediction Button
# ---------------------
if st.button("🔍 Predict Digit"):
    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'))
        processed_image = preprocess_canvas_image(image)

        with torch.no_grad():
            output = model(processed_image)
            prediction = torch.argmax(output, dim=1).item()
            confidence = F.softmax(output, dim=1)[0][prediction].item()

        st.markdown(f"<h3 style='text-align: center; color: #1F618D;'>🧠 Predicted Digit: <span style='color:#28B463'>{prediction}</span></h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #7D3C98;'>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)
    else:
        st.warning("Please draw a digit before prediction!")

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.markdown("<small style='text-align: center; display: block;'>Built with ❤️ by Harshitha using PyTorch + Streamlit</small>", unsafe_allow_html=True)
