"use client";

import { useEffect, useState } from "react";
import { Navbar } from "@/components/Navbar";
import { api } from "@/lib/api";
import { ModelInfo } from "@/types/model";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Cpu, Activity, Database, Scale } from "lucide-react";

export default function ModelPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadModel() {
      try {
        const data = await api.getModelInfo();
        setModelInfo(data);
      } catch (err) {
        console.error("Failed to load model info", err);
      } finally {
        setLoading(false);
      }
    }
    loadModel();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Navbar />
      <main className="flex-1 container mx-auto px-4 py-8 max-w-5xl">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Active Model Details</h1>
        
        {loading ? (
          <p className="text-center py-8 text-gray-500">Loading model metrics...</p>
        ) : modelInfo ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card>
              <CardContent className="pt-6 flex flex-col items-center text-center">
                <div className="p-3 bg-blue-100 text-blue-600 rounded-full mb-4">
                  <Cpu className="w-8 h-8" />
                </div>
                <p className="text-sm text-gray-500 font-medium uppercase tracking-wider">Architecture</p>
                <h3 className="text-2xl font-bold mt-1">{modelInfo.model_name}</h3>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6 flex flex-col items-center text-center">
                <div className="p-3 bg-green-100 text-green-600 rounded-full mb-4">
                  <Activity className="w-8 h-8" />
                </div>
                <p className="text-sm text-gray-500 font-medium uppercase tracking-wider">Test Accuracy</p>
                <h3 className="text-2xl font-bold mt-1">{(modelInfo.accuracy * 100).toFixed(1)}%</h3>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6 flex flex-col items-center text-center">
                <div className="p-3 bg-purple-100 text-purple-600 rounded-full mb-4">
                  <Database className="w-8 h-8" />
                </div>
                <p className="text-sm text-gray-500 font-medium uppercase tracking-wider">Parameters</p>
                <h3 className="text-2xl font-bold mt-1">
                  {(modelInfo.parameters / 1000000).toFixed(2)}M
                </h3>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6 flex flex-col items-center text-center">
                <div className="p-3 bg-orange-100 text-orange-600 rounded-full mb-4">
                  <Scale className="w-8 h-8" />
                </div>
                <p className="text-sm text-gray-500 font-medium uppercase tracking-wider">Model Size</p>
                <h3 className="text-2xl font-bold mt-1">
                  {modelInfo.model_size_mb?.toFixed(1) || "?"} MB
                </h3>
              </CardContent>
            </Card>
          </div>
        ) : (
          <p className="text-center py-8 text-red-500">Failed to load metrics.</p>
        )}
      </main>
    </div>
  );
}
