import { ModelInfo } from "./model";

export interface DatasetInfo {
  total_images: number;
  classes: string[];
  split_ratio: Record<string, number>;
}

export interface ResearchInfo {
  dataset: DatasetInfo;
  models: ModelInfo[];
  methodology: string[];
}
