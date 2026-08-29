import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Leaf, Shield, Zap } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Navbar />
      
      <main className="flex-1">
        {/* Hero Section */}
        <section className="bg-white py-20 border-b">
          <div className="container mx-auto px-4 max-w-5xl text-center">
            <h1 className="text-5xl font-extrabold tracking-tight text-gray-900 sm:text-6xl mb-6">
              Detect Coffee Leaf Diseases <span className="text-green-600">Instantly</span>
            </h1>
            <p className="mt-4 text-xl text-gray-600 max-w-2xl mx-auto mb-10">
              Upload a picture of a coffee leaf and our state-of-the-art AI model will identify diseases like Rust, Leaf Miner, Phoma, and more in milliseconds.
            </p>
            <Link href="/predict">
              <Button size="lg" className="bg-green-600 hover:bg-green-700 text-lg px-8 py-6 rounded-full">
                Analyze a Leaf <ArrowRight className="ml-2 w-5 h-5" />
              </Button>
            </Link>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20 bg-gray-50">
          <div className="container mx-auto px-4 max-w-5xl">
            <div className="grid md:grid-cols-3 gap-10">
              <div className="bg-white p-8 rounded-2xl shadow-sm border text-center">
                <div className="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Zap className="w-7 h-7 text-green-600" />
                </div>
                <h3 className="text-xl font-bold mb-3">Fast Inference</h3>
                <p className="text-gray-600">Powered by a highly optimized MobileNetV3 architecture for real-time predictions.</p>
              </div>
              <div className="bg-white p-8 rounded-2xl shadow-sm border text-center">
                <div className="w-14 h-14 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Shield className="w-7 h-7 text-blue-600" />
                </div>
                <h3 className="text-xl font-bold mb-3">High Accuracy</h3>
                <p className="text-gray-600">Trained on a harmonized dataset of over 3,000 images achieving 84%+ accuracy.</p>
              </div>
              <div className="bg-white p-8 rounded-2xl shadow-sm border text-center">
                <div className="w-14 h-14 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Leaf className="w-7 h-7 text-purple-600" />
                </div>
                <h3 className="text-xl font-bold mb-3">Visual Explanations</h3>
                <p className="text-gray-600">Grad-CAM heatmaps highlight exactly which part of the leaf the AI focused on.</p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
