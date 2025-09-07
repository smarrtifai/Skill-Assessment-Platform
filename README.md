# Skills Assessment Platform

A comprehensive platform for skills assessment and career roadmap generation based on resume analysis.

## Features

- **Resume Upload**: Supports PDF and DOC/DOCX formats
- **Skill Extraction**: AI-powered extraction of skills from resume text
- **MCQ Generation**: Dynamic generation of 30 multiple-choice questions
- **Skill Assessment**: Evaluation of responses to determine skill level
- **Career Roadmap**: Personalized learning path based on chosen tech field
- **Downloadable Results**: Export roadmap as text file

## Tech Fields Supported

- AI/ML
- Business Analytics and Data Analytics
- Gen AI
- Agentic AI

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open browser and navigate to `http://localhost:5000`

## Testing Routes

Run the test script to verify all routes are working:

```bash
python test_routes.py
```

## Usage

### Multi-Page Flow:
1. **Landing Page** (`/`) - Complete conversational onboarding
2. **Upload Page** (`/upload`) - Upload resume for analysis OR
3. **Assessment Page** (`/assessment`) - Take skills assessment
4. **Results Page** (`/results`) - View results and select career path
5. **Roadmap Page** (`/roadmap`) - View and download personalized roadmap

### Features:
- **Conversational Onboarding** - Interactive chat-based user journey
- **Dual Assessment Modes** - Resume-based or interest-based assessment
- **AI Mentor** - Get instant career advice and guidance
- **Secure Testing** - Tab-switching detection and time limits
- **Progress Tracking** - Visual progress indicators across pages
- **Session Management** - Data persistence across page navigation

## Project Structure

```
Skills Assessment Platform/
├── app.py                    # Main Flask application with routing
├── templates/
│   ├── index.html           # Landing page with onboarding
│   ├── upload.html          # Resume upload page
│   ├── assessment.html      # Skills assessment page
│   ├── results.html         # Results and career path selection
│   └── roadmap.html         # Career roadmap display
├── static/
│   ├── css/
│   │   ├── styles.css       # Main stylesheet
│   │   └── roadmap-styles.css # Roadmap-specific styles
│   └── js/
│       └── app.js           # JavaScript functionality
├── uploads/                 # Resume and roadmap storage
├── test_routes.py          # Route testing script
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Page Routes

- `GET /` - Landing page with conversational onboarding
- `GET /upload` - Resume upload page
- `GET /assessment` - Skills assessment page
- `GET /results` - Assessment results page
- `GET /roadmap` - Career roadmap page

## API Endpoints

- `POST /api/onboarding` - Handle onboarding flow
- `POST /api/upload` - Upload and analyze resume
- `POST /api/assessment/interest-based` - Start interest-based assessment
- `POST /api/assess` - Submit assessment answers
- `POST /api/roadmap` - Generate career roadmap
- `POST /api/download_roadmap` - Download roadmap file
- `GET /api/get_assessment_data` - Retrieve assessment data
- `GET /api/get_results_data` - Retrieve results data
- `GET /api/get_roadmap_data` - Retrieve roadmap data
- `POST /api/clear_session` - Clear session data
- `POST /api/mentor` - AI mentor chat