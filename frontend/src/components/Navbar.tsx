import Link from "next/link";
import { Leaf } from "lucide-react";

export function Navbar() {
  return (
    <nav className="border-b bg-white">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center space-x-2">
          <Leaf className="w-6 h-6 text-green-600" />
          <span className="font-semibold text-lg">Coffee Leaf AI</span>
        </Link>
        
        <div className="hidden md:flex space-x-6 text-sm font-medium">
          <Link href="/" className="text-gray-600 hover:text-green-600 transition-colors">Home</Link>
          <Link href="/predict" className="text-gray-600 hover:text-green-600 transition-colors">Analyze</Link>
          <Link href="/history" className="text-gray-600 hover:text-green-600 transition-colors">History</Link>
          <Link href="/model" className="text-gray-600 hover:text-green-600 transition-colors">Model</Link>
          <Link href="/research" className="text-gray-600 hover:text-green-600 transition-colors">Research</Link>
        </div>
      </div>
    </nav>
  );
}
