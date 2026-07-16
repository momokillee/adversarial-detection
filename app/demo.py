"""Gradio demo for adversarial sample detection."""

from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from detector.model import AdversarialDetector
from utils.preprocess import pil_to_tensor

MODEL_PATH = Path("models/detector.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 64


def load_model() -> AdversarialDetector:
    model = AdversarialDetector().to(DEVICE)
    if MODEL_PATH.exists():
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        except TypeError:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    model.eval()
    return model


_model = load_model()


def predict(image: Image.Image) -> str:
    if image is None:
        return "Upload an image to classify."
    tensor = pil_to_tensor(image, IMAGE_SIZE, DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(_model(tensor)).item()
    label = "Adversarial" if prob > 0.5 else "Clean"
    return f"{label} (confidence: {prob:.2%})"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Adversarial Sample Detector",
    description=f"Upload an image to check whether it looks like a clean or adversarial sample. "
                f"Running on: {DEVICE}",
)

if __name__ == "__main__":
    print(f"Starting Gradio demo on device: {DEVICE}")
    demo.launch()
