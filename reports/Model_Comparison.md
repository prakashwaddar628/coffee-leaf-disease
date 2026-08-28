# Model Architecture Comparison

This document strictly tracks the evaluation metrics of various architectures to determine the optimal trade-off between complexity and performance for Coffee Leaf Disease classification.

| Metric | MobileNetV3 (Frozen) | MobileNetV3 (Fine-Tuned) | ResNet50 (Frozen) | EfficientNet-B0 | DenseNet121 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 67.9% | 74.3% | 74.4% | — | — |
| **Precision (Weighted)** | 67.7% | 72.0% | 73.3% | — | — |
| **Recall (Weighted)** | 67.9% | 74.3% | 74.4% | — | — |
| **F1-Score (Weighted)** | 67.6% | 72.8% | 73.2% | — | — |
| **Total Parameters** | 1,520,931 | 1,520,931 | 23,514,179 | — | — |
| **Trainable Parameters** | 3,075 | 353,619 | 6,147 | — | — |
| **Training Time (sec)** | ~293 | ~252 | ~3167 | — | — |
| **Inference FPS** | 41.4 | 127.8 | 7.3 | — | — |
