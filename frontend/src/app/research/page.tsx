"use client";

import { useEffect, useState } from "react";
import { Navbar } from "@/components/Navbar";
import { api } from "@/lib/api";
import { ResearchInfo } from "@/types/research";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function ResearchPage() {
  const [researchInfo, setResearchInfo] = useState<ResearchInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadResearch() {
      try {
        const data = await api.getResearchInfo();
        setResearchInfo(data);
      } catch (err) {
        console.error("Failed to load research info", err);
      } finally {
        setLoading(false);
      }
    }
    loadResearch();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Navbar />
      <main className="flex-1 container mx-auto px-4 py-8 max-w-5xl space-y-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Research & Methodology</h1>
        
        {loading ? (
          <p className="text-center py-8 text-gray-500">Loading research details...</p>
        ) : researchInfo ? (
          <>
            <Card>
              <CardHeader>
                <CardTitle>Dataset Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="font-semibold mb-2">Total Images</h3>
                    <p className="text-2xl font-bold text-green-700">{researchInfo.dataset.total_images}</p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Classes</h3>
                    <div className="flex flex-wrap gap-2">
                      {researchInfo.dataset.classes.map(c => (
                        <span key={c} className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm">
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Architecture</TableHead>
                      <TableHead>Accuracy</TableHead>
                      <TableHead>Parameters</TableHead>
                      <TableHead>Size (MB)</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {researchInfo.models.map(m => (
                      <TableRow key={m.model_name}>
                        <TableCell className="font-medium">{m.model_name}</TableCell>
                        <TableCell>{(m.accuracy * 100).toFixed(1)}%</TableCell>
                        <TableCell>{(m.parameters / 1000000).toFixed(2)}M</TableCell>
                        <TableCell>{m.model_size_mb?.toFixed(1) || "?"}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Methodology</CardTitle>
              </CardHeader>
              <CardContent>
                <ol className="list-decimal list-inside space-y-2 text-gray-700">
                  {researchInfo.methodology.map((step, idx) => (
                    <li key={idx} className="leading-relaxed">{step}</li>
                  ))}
                </ol>
              </CardContent>
            </Card>
          </>
        ) : (
          <p className="text-center py-8 text-red-500">Failed to load research data.</p>
        )}
      </main>
    </div>
  );
}
