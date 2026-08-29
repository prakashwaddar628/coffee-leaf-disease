export interface ModelInfo {
  model_name: string;
  accuracy: number;
  parameters: number;
  trainable_parameters: number;
  inference_fps: number;
  model_size_mb?: number;
}
