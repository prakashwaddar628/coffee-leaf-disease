# Sprint 6: resnet50 Baseline Report

## Objective
Evaluate whether a deeper residual network (ResNet50) can outperform MobileNetV3 while using the exact same experimental setup.

## Configuration
- **Experiment Path**: `..\results\resnet50\experiment_004`
- **Epochs**: 20
- **Optimizer**: AdamW

## Model Complexity
- **Total Parameters**: 23,514,179
- **Trainable Parameters**: 6,147
- **Model Size**: 89.70 MB
- **Inference Speed**: 7.3 FPS

## Results
- **Test Accuracy**: 74.36%

## Comparison with MobileNetV3
- **Performance:** ResNet50 (Frozen) achieved a test accuracy of 74.36%, which matches the performance of the MobileNetV3 (Fine-Tuned) model (74.3%), but significantly outperforms the MobileNetV3 (Frozen) model (67.9%).
- **Computational Cost:** ResNet50 incurs a massive computational penalty. Its model size is ~90MB (vs MobileNet's ~6MB), it has 15x more total parameters, and inference FPS plummeted from 127.8 to 7.3.
- **Training Efficiency:** Training took considerably longer (~3167s vs ~252s) due to the heavier backbone.

## Observations & Limitations
- **Diminishing Returns:** The deeper ResNet50 architecture does not justify its computational cost in a frozen state when compared to a fine-tuned lightweight MobileNetV3. 
- **Limitations:** The model was only trained with its backbone frozen. A fine-tuned ResNet50 might yield higher accuracy, but the inference and training time penalties would remain.

See `results/resnet50/experiment_XXX/` for confusion matrix and misclassifications.
