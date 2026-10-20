# Edge AI Image Classifier: MobileNetV2 + ONNX Runtime

An image classification application built with MobileNetV2, ONNX
Runtime and Gradio. The project explores model deployment preparation
for edge AI systems, with AMD Kria KV260 as the intended FPGA target.

## Features

- ONNX-based inference
- Image upload interface built with Gradio
- Top-3 class predictions and confidence scores
- Inference latency measurement
- JSON performance report
- Initial INT8 quantization experiments

## Model

The model classifies images into the 10 CIFAR-10 categories:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Input shape: `1 x 3 x 224 x 224`

Output: 10 class logits.

## Initial Evaluation

The following measurements were obtained on Google Colab CPU:

| Metric | Result |
|---|---:|
| Evaluation samples | 200 |
| Accuracy | 71.5% |
| Mean latency | 29.10 ms |
| Median latency | 14.96 ms |
| P95 latency | 84.94 ms |
| Estimated throughput | 34.36 images/s |

These results are from a 200-image test subset. They are not a
full-dataset evaluation or FPGA performance measurements.
Latency can vary with the runtime environment.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

## Project Status

- [x] Exported model to ONNX
- [x] Created an ONNX Runtime inference application
- [x] Built a Gradio image-upload interface
- [x] Saved an initial performance report
- [ ] Further accuracy and latency evaluation
- [ ] AMD Vitis AI compilation for Kria KV260
- [ ] FPGA hardware validation

**Important:** Vitis AI compilation and physical FPGA testing have
not been completed. All performance numbers above are CPU results.

## Repository Structure

```text
edge-ai-fpga-mobilenetv2/
├── app.py
├── requirements.txt
├── README.md
├── edge_inference_report.json
└── models/
    └── mobilenet_cifar10.onnx
```

## Technologies

Python, PyTorch, MobileNetV2, ONNX, ONNX Runtime, Gradio, NumPy.

## License

Choose a license before public distribution. Confirm the licenses
of model code, dependencies and any third-party assets first.
