# Model Architecture Comparison

This document strictly tracks the evaluation metrics of various architectures to determine the optimal trade-off between complexity and performance for Coffee Leaf Disease classification.

| Metric | MobileNetV3 (Frozen) | MobileNetV3 (Fine-Tuned) | ResNet50 (Frozen) | EfficientNet-B0 | DenseNet121 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 67.9% | 74.3% | — | — | — |
| **Precision (Weighted)** | 67.7% | 72.0% | — | — | — |
| **Recall (Weighted)** | 67.9% | 74.3% | — | — | — |
| **F1-Score (Weighted)** | 67.6% | 72.8% | — | — | — |
| **Total Parameters** | 1,520,931 | 1,520,931 | — | — | — |
| **Trainable Parameters** | 3,075 | 353,619 | — | — | — |
| **Training Time (sec)** | ~293 | ~252 | — | — | — |
| **Inference FPS** | 41.4 | 127.8 | — | — | — |
