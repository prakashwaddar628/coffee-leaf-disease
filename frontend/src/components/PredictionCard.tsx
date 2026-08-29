"use client";

import { AlertCircle, CheckCircle2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { PredictionResult } from "@/types/prediction";

interface PredictionCardProps {
  prediction: PredictionResult;
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const isHealthy = prediction.predicted_class.toLowerCase() === "healthy";
  const confidencePercent = (prediction.confidence * 100).toFixed(1);

  return (
    <Card className={isHealthy ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}>
      <CardContent className="pt-6">
        <div className="flex items-start space-x-4">
          <div className={`p-3 rounded-full ${isHealthy ? "bg-green-100" : "bg-red-100"}`}>
            {isHealthy ? (
              <CheckCircle2 className="w-8 h-8 text-green-600" />
            ) : (
              <AlertCircle className="w-8 h-8 text-red-600" />
            )}
          </div>
          
          <div>
            <p className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-1">
              Top Prediction
            </p>
            <h2 className={`text-3xl font-bold ${isHealthy ? "text-green-700" : "text-red-700"}`}>
              {prediction.predicted_class}
            </h2>
            <p className="text-lg text-gray-700 mt-2">
              Confidence: <span className="font-semibold">{confidencePercent}%</span>
            </p>
            
            {!isHealthy && (
              <p className="text-sm text-red-600 mt-4 max-w-md">
                Disease detected. We recommend analyzing the Grad-CAM heatmap below to see which areas of the leaf are affected.
              </p>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
