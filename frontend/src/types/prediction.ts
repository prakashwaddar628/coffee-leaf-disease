export interface PredictionResult {
  id: string;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  image_url?: string;
  gradcam_url?: string;
  model_name: string;
  timestamp: string;
}

export interface HistoryItem {
  id: string;
  predicted_class: string;
  confidence: number;
  timestamp: string;
  image_url?: string;
}
