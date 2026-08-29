"use client";

import { useState } from "react";
import { ImageUploader } from "@/components/ImageUploader";
import { PredictionCard } from "@/components/PredictionCard";
import { ProbabilityChart } from "@/components/ProbabilityChart";
import { GradCAMViewer } from "@/components/GradCAMViewer";
import { api } from "@/lib/api";
import { PredictionResult } from "@/types/prediction";
import { Navbar } from "@/components/Navbar";
import { Alert, AlertDescription } from "@/components/ui/alert";

export default function PredictPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (file: File) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const prediction = await api.predict(file);
      setResult(prediction);
    } catch (err: any) {
      setError(err.message || "Failed to analyze image. Ensure the backend is running.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Navbar />
      
      <main className="flex-1 container mx-auto px-4 py-8 max-w-6xl">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Analyze Coffee Leaf</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Upload an image of a coffee leaf to detect diseases including Rust, Leaf Miner, Phoma, Cercospora, and Red Spider Mite.
          </p>
        </div>

        {error && (
          <Alert variant="destructive" className="mb-8 max-w-2xl mx-auto">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <ImageUploader onAnalyze={handleAnalyze} isLoading={isLoading} />

        {result && (
          <div className="mt-12 space-y-8 animate-in fade-in duration-500">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="md:col-span-2">
                <PredictionCard prediction={result} />
              </div>
              <div className="md:col-span-1">
                <ProbabilityChart probabilities={result.probabilities} />
              </div>
            </div>
            
            <GradCAMViewer 
              originalUrl={result.image_url} 
              gradcamUrl={result.gradcam_url} 
            />
          </div>
        )}
      </main>
    </div>
  );
}
