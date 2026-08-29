Frontend PRD — Coffee Leaf Disease Research & Detection System

Document Version: 1.0
Project: Coffee Leaf Disease Detection
Frontend: Next.js + React + TypeScript
UI: Tailwind CSS + shadcn/ui
Charts: Recharts
Backend: FastAPI
ML: PyTorch
Status: Product Definition / Ready for Implementation

1. Product Overview

The Coffee Leaf Disease Detection System is an AI-powered web application that allows users to upload an image of a coffee leaf and receive an AI-based disease classification.

The frontend will provide a clean interface between the user and the trained deep-learning model.

The application will eventually support:

Coffee leaf image upload
Disease prediction
Prediction confidence
Class probability distribution
Grad-CAM explainability
Prediction history
Model information
Research/model comparison
Dataset information

The frontend will not directly execute the PyTorch model.

Instead:

User
 │
 ▼
Next.js Frontend
 │
 │ REST API
 ▼
FastAPI Backend
 │
 ▼
PyTorch Model
 │
 ▼
Prediction
 │
 ▼
FastAPI
 │
 ▼
Next.js
 │
 ▼
User
2. Product Objective

The primary objective is:

Build a professional, intuitive, responsive frontend that allows users to analyze coffee leaf images using the project's trained AI model while providing transparent prediction confidence and explainability.

The secondary objective is to expose selected research information so that the application also serves as a demonstration of the research work.

3. Target Users
3.1 Farmer / General User

Wants to:

Upload a coffee leaf image.
Quickly understand the predicted condition.
See confidence.
View an explanation.

Technical knowledge should not be required.

3.2 Student / Researcher

Wants to:

Understand which model is being used.
View model metrics.
Compare models.
Examine Grad-CAM results.
Understand dataset information.
3.3 Professor / Evaluator

Wants to quickly see:

Project objective.
Dataset.
Models evaluated.
Performance.
Research methodology.
Final model.
Demonstration of the working system.
4. Technology Requirements
Frontend
Next.js
React
TypeScript
Styling
Tailwind CSS
shadcn/ui
Visualization
Recharts
API Communication

Use standard HTTP/REST communication with FastAPI.

5. Design Principles

The frontend must follow these principles:

5.1 Simple

A user should understand what to do immediately.

5.2 Scientific

Results should not be presented as unexplained AI decisions.

5.3 Transparent

Show:

Prediction
Confidence
Class probabilities
Explanation
5.4 Responsive

The application must work on:

Desktop
Laptop
Tablet
Mobile
5.5 Accessible

Use:

readable typography
sufficient contrast
keyboard navigation
meaningful labels
accessible buttons
image alternative text
6. Application Routes

The initial frontend should contain:

/
├── Home
│
├── predict
│   └── Prediction
│
├── results/[id]
│   └── Detailed Result
│
├── history
│   └── Prediction History
│
├── model
│   └── Model Information
│
└── research
    └── Research Dashboard
7. Home Page
Route
/
Objective

Introduce the project and direct the user toward image analysis.

Hero section
Coffee Leaf Disease Detection

AI-powered analysis for coffee leaf health

[ Analyze a Leaf ]
Supported Classes

Display the disease classes supported by the currently deployed model.

For example:

Healthy
Rust
Miner
Red Spider Mite

Important: These should eventually be obtained from the backend/model metadata rather than hardcoded.

How It Works
01
Upload Leaf

↓

02
AI Analysis

↓

03
View Prediction

↓

04
Understand Result
8. Prediction Page
Route
/predict

This is the primary application screen.

Image Upload

The user should be able to:

Drag and drop an image.
Click to select an image.
Preview the selected image.
Remove the image.
Upload another image.

Supported formats:

JPG
JPEG
PNG
WEBP

Maximum file size should be controlled by the backend configuration.

The frontend should display an appropriate validation error if the file is invalid.

9. Upload UI

Example:

┌─────────────────────────────────────┐
│                                     │
│          Upload Coffee Leaf         │
│                                     │
│       Drag & drop your image        │
│               or                    │
│          [ Choose Image ]           │
│                                     │
│       JPG • PNG • WEBP              │
│                                     │
└─────────────────────────────────────┘

After selection:

┌───────────────────┐
│                   │
│   Leaf Preview    │
│                   │
└───────────────────┘

filename.jpg

[ Remove ]     [ Analyze ]
10. Prediction Processing State

After clicking Analyze:

Analyzing your coffee leaf...

Preparing image
      ↓
Running AI model
      ↓
Generating prediction

The UI must not appear frozen.

Display:

Loading indicator.
Progress/status message.
Disable duplicate submissions.
11. Prediction Result

After successful inference:

┌─────────────────────────────────────┐
│           Prediction                │
│                                     │
│              RUST                   │
│                                     │
│           Confidence               │
│              87.4%                 │
└─────────────────────────────────────┘
12. Class Probability Distribution

Display all model classes.

Example:

Rust              87.4%
██████████████████████████

Healthy            8.1%
████

Miner              3.2%
██

Red Spider Mite    1.3%
█

A Recharts visualization can be used.

13. Confidence Handling

We should not present confidence as certainty.

For example:

87.4% confidence

should be accompanied by appropriate UI language such as:

The model predicts Rust with a confidence score of 87.4%.

Avoid:

The leaf definitely has Rust.

This is especially important for an agricultural AI system.

14. Grad-CAM Explainability
Route
/results/[id]

The result page should eventually contain:

Original Image

        +

Grad-CAM Heatmap

Example:

┌───────────────┐     ┌───────────────┐
│ Original Leaf │     │  AI Attention │
│               │     │               │
│      🌿       │     │      🔥       │
└───────────────┘     └───────────────┘
Explanation

Display:

The highlighted regions indicate areas that contributed strongly to the model's prediction.

We should not claim that Grad-CAM proves the disease location.

It shows model attention/activation, not clinical or biological proof.

15. Prediction History
Route
/history

Display previous analyses.

Example:

Image	Prediction	Confidence	Date
Leaf 001	Rust	87.4%	Today
Leaf 002	Healthy	94.1%	Today
Leaf 003	Miner	71.3%	Yesterday
16. History Detail

Clicking a prediction should open:

/results/[id]

with:

Original image
Prediction
Confidence
Probability distribution
Grad-CAM
Model used
Timestamp
17. Model Information Page
Route
/model

Display information about the currently deployed model.

Example:

Current Model

MobileNetV3 Small

Test Accuracy
74.36%

Parameters
1,520,931

Trainable Parameters
353,619

Inference Speed
127.8 FPS

The current Sprint 5 report supports these MobileNetV3 experiment metrics, including 74.36% test accuracy, 1,520,931 total parameters, 353,619 trainable parameters, and 127.8 FPS for Experiment 006.

However, these values should not be hardcoded into the frontend. They are research results from the current experiment and should eventually come from model metadata/API.

18. Research Dashboard
Route
/research

This page is primarily for researchers and project evaluation.

Display:

Dataset
Dataset Size
Classes
Image Resolution
Class Distribution
Models
MobileNetV3
ResNet50
EfficientNet-B0
DenseNet121
Hybrid
Metrics
Accuracy
Precision
Recall
F1
Inference Speed
Parameters
19. Model Comparison

Create a comparison table:

Model	Accuracy	Precision	Recall	F1	Parameters	FPS
MobileNetV3	74.36%	—	—	—	1.52M	127.8
ResNet50	—	—	—	—	—	—
EfficientNet-B0	—	—	—	—	—	—
DenseNet121	—	—	—	—	—	—

Only populate values when verified from experiment artifacts.

20. Research Methodology Section

The research page should visually represent:

Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
Augmentation
   ↓
Baseline Models
   ↓
Fine-Tuning
   ↓
Model Comparison
   ↓
Hybrid Model
   ↓
Explainability
   ↓
Robustness
   ↓
Final Model

This will be extremely useful during your professor's evaluation.

21. Navigation

Desktop navigation:

Logo

Home
Analyze
History
Model
Research

                         About

Mobile navigation should collapse into a mobile menu.

22. Reusable Components

Create reusable components instead of putting everything inside page files.

components/

├── Navbar.tsx
├── Footer.tsx
├── ImageUploader.tsx
├── ImagePreview.tsx
├── PredictionCard.tsx
├── ConfidenceChart.tsx
├── ProbabilityChart.tsx
├── GradCAMViewer.tsx
├── ModelStats.tsx
├── ModelComparison.tsx
├── ResearchPipeline.tsx
├── LoadingState.tsx
├── ErrorState.tsx
└── EmptyState.tsx
23. API Layer

Create:

lib/api.ts

All backend communication should happen through this layer.

Example conceptual API:

POST /api/v1/predict

GET /api/v1/predictions/{id}

GET /api/v1/history

GET /api/v1/model

GET /api/v1/research

The exact backend API contract should be finalized when we build the FastAPI backend.

24. Type Safety

Create:

types/

prediction.ts
model.ts
research.ts
api.ts

Example conceptual prediction response:

interface PredictionResult {
    id: string;
    predicted_class: string;
    confidence: number;
    probabilities: Record<string, number>;
    image_url?: string;
    gradcam_url?: string;
    model_name: string;
    timestamp: string;
}

The actual interface should match the finalized FastAPI schema.

25. Error Handling

The frontend must gracefully handle:

Invalid image
Please upload a supported image format.
Image too large
Image exceeds the maximum allowed size.
Backend unavailable
The AI service is currently unavailable.
Please try again.
Prediction failure
We couldn't analyze this image.
Please try again.
Empty history
No predictions yet.

[ Analyze Your First Leaf ]
26. Security Requirements

The frontend must:

Validate file types.
Validate file size.
Never expose API secrets.
Never expose model files unnecessarily.
Never trust filenames from users.
Avoid rendering unsanitized user input.
Use HTTPS in production.

Authentication is not required for the first version unless we later decide prediction history needs user accounts.

27. Performance Requirements

Target:

Fast initial page load.
Optimized images.
Lazy-load heavy visualization components where appropriate.
Avoid unnecessary API requests.
Show immediate upload preview.
Provide responsive loading states.

The frontend should remain usable even on relatively modest hardware/mobile connections.

28. Responsive Requirements
Desktop

Full dashboard/navigation.

Tablet

Adaptive two-column layouts.

Mobile

Single-column layouts:

Image
 ↓
Prediction
 ↓
Confidence
 ↓
Probabilities
 ↓
Grad-CAM

No horizontal scrolling.

29. Visual Design Direction

The visual identity should communicate:

Agriculture + AI + Research

Recommended aesthetic:

Clean
Modern
Minimal
Scientific
Natural
Professional

Avoid:

excessive gradients
excessive animations
cartoon-like agricultural graphics
cluttered dashboards
unnecessary glassmorphism

The application should look like a real AI research product, not a generic college project.

30. Animation

Use subtle animations only:

Upload hover
Image preview transition
Loading state
Result appearance
Chart entrance

Avoid excessive animation because prediction results should remain the focus.

31. Accessibility

Implement:

Semantic HTML.
Keyboard navigation.
ARIA labels where needed.
Accessible upload control.
Visible focus states.
Meaningful error messages.
Alt text for images.
Don't rely solely on color to communicate disease status.
32. Frontend Data Flow

The complete user journey:

User opens application
        ↓
Home page
        ↓
Analyze Leaf
        ↓
Upload image
        ↓
Frontend validates image
        ↓
POST image to FastAPI
        ↓
Backend preprocessing
        ↓
PyTorch inference
        ↓
Prediction
        ↓
Confidence
        ↓
Class probabilities
        ↓
Grad-CAM
        ↓
Frontend displays result
        ↓
User can inspect detailed analysis
33. MVP

The first frontend version should not attempt to build everything.

MVP features:
Home
   ↓
Image Upload
   ↓
Prediction
   ↓
Confidence
   ↓
Class Probabilities
   ↓
Result Page

Once the API and final model are stable, add:

Grad-CAM
History
Model Information
Research Dashboard
34. Phase 2 Features

Potential future additions:

User accounts
Cloud prediction history
Multiple image analysis
Batch prediction
PDF report generation
Downloadable analysis report
Farmer-friendly disease guidance
Multilingual interface
PWA/mobile installation

These are not part of the initial implementation.

35. Important ML/Product Boundary

The frontend must never make biological claims beyond what the model supports.

For example, don't display:

"Your plant definitely has Rust."

Instead:

Model prediction: Rust
Confidence: 87.4%

And eventually:

"This result is an AI-assisted prediction and should not replace expert agricultural diagnosis."

This distinction is important for responsible AI.

36. Definition of Done

The frontend MVP is complete when:

 Next.js project is configured.
 TypeScript is enabled.
 Tailwind CSS is configured.
 shadcn/ui is configured.
 Responsive navigation works.
 Home page works.
 Image upload works.
 Client-side validation works.
 API integration works.
 Prediction result displays.
 Confidence displays.
 Probability chart displays.
 Error states work.
 Loading states work.
 Mobile layout works.
 No API secrets are exposed.
 Components are reusable.
 TypeScript has no avoidable type errors.
 Production build succeeds.
37. Final Frontend Architecture
frontend/
│
├── app/
│   ├── page.tsx
│   ├── predict/
│   │   └── page.tsx
│   ├── results/
│   │   └── [id]/
│   │       └── page.tsx
│   ├── history/
│   │   └── page.tsx
│   ├── model/
│   │   └── page.tsx
│   └── research/
│       └── page.tsx
│
├── components/
│   ├── Navbar.tsx
│   ├── ImageUploader.tsx
│   ├── PredictionCard.tsx
│   ├── ProbabilityChart.tsx
│   ├── GradCAMViewer.tsx
│   ├── ModelStats.tsx
│   └── ...
│
├── lib/
│   ├── api.ts
│   └── utils.ts
│
├── types/
│   ├── prediction.ts
│   ├── model.ts
│   └── research.ts
│
├── public/
│
├── package.json
├── tsconfig.json
└── README.md
38. Implementation Order

When we eventually start frontend development, I recommend this exact order:

Phase 1
Next.js Setup
       ↓
Phase 2
Design System
       ↓
Phase 3
Home Page
       ↓
Phase 4
Upload Interface
       ↓
Phase 5
FastAPI Integration
       ↓
Phase 6
Prediction Results
       ↓
Phase 7
Probability Visualization
       ↓
Phase 8
Grad-CAM
       ↓
Phase 9
History
       ↓
Phase 10
Model Dashboard
       ↓
Phase 11
Research Dashboard
       ↓
Phase 12
Responsive + Accessibility
       ↓
Phase 13
Production Build