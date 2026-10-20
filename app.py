from pathlib import Path
import time
import numpy as np
import gradio as gr
import onnxruntime as ort
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mobilenet_cifar10.onnx"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image):
    if image is None:
        return "Lütfen bir görsel yükleyin.", {}

    image = image.convert("RGB")
    input_data = preprocess(image).unsqueeze(0).numpy().astype(np.float32)

    start = time.perf_counter()
    logits = session.run(["logits"], {"image": input_data})[0][0]
    latency_ms = (time.perf_counter() - start) * 1000

    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    top_indices = np.argsort(probabilities)[-3:][::-1]
    best_index = int(top_indices[0])

    summary = (
        f"Tahmin: {CLASS_NAMES[best_index]}\\n"
        f"Güven: %{probabilities[best_index] * 100:.2f}\\n"
        f"Model çıkarım süresi: {latency_ms:.2f} ms\\n"
        "Çalışma ortamı: ONNX Runtime CPU"
    )

    top3 = {
        CLASS_NAMES[int(i)]: float(probabilities[i])
        for i in top_indices
    }

    return summary, top3

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Bir görsel yükle"),
    outputs=[
        gr.Textbox(label="Tahmin sonucu", lines=4),
        gr.Label(label="En olası 3 sınıf", num_top_classes=3)
    ],
    title="Edge AI Image Classifier",
    description=(
        "MobileNetV2 ve ONNX Runtime ile CIFAR-10 sınıflandırması. "
        "Model şu anda CPU üzerinde çalışmaktadır; FPGA üzerinde "
        "derleme veya donanım testi yapılmamıştır."
    )
)

if __name__ == "__main__":
    demo.launch()
