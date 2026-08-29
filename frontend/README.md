# Coffee Leaf Disease Detection - Frontend

This is the Next.js frontend application for the Coffee Leaf Disease Detection System.

## Features

- **Instant Analysis:** Upload leaf images via a drag-and-drop interface
- **Explainability:** View Grad-CAM heatmaps overlaying the predicted area
- **History:** Dashboard showing past predictions
- **Metrics:** View real-time model metrics and research methodology

## Tech Stack

- **Framework:** Next.js (App Router)
- **Styling:** Tailwind CSS + shadcn/ui
- **Language:** TypeScript
- **Charts:** Recharts

## Getting Started

First, ensure the FastAPI backend is running on `http://localhost:8000`.

Then, run the development server:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
