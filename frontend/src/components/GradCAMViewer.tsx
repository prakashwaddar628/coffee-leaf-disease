"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";

interface GradCAMViewerProps {
  originalUrl?: string;
  gradcamUrl?: string;
}

export function GradCAMViewer({ originalUrl, gradcamUrl }: GradCAMViewerProps) {
  if (!originalUrl || !gradcamUrl) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Explainability (Grad-CAM)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-500 text-center">Original Image</h4>
            <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden border">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img 
                src={api.getImageUrl(originalUrl)} 
                alt="Original" 
                className="w-full h-full object-contain"
              />
            </div>
          </div>
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-500 text-center">Attention Heatmap</h4>
            <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden border">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img 
                src={api.getImageUrl(gradcamUrl)} 
                alt="Grad-CAM" 
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        </div>
        <p className="text-sm text-gray-500 mt-4 text-center">
          The heatmap highlights the regions of the leaf that the AI focused on to make its prediction.
        </p>
      </CardContent>
    </Card>
  );
}
