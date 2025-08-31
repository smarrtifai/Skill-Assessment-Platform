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

## Usage

1. Upload your resume (PDF/DOC format)
2. Complete the generated MCQ assessment
3. Select your desired tech field
4. View and download your personalized roadmap

## Project Structure

```
Skills Assessment Platform/
├── app.py              # Main Flask application
├── templates/
│   └── index.html      # Frontend interface
├── uploads/            # Resume and roadmap storage
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## API Endpoints

- `POST /upload` - Upload and analyze resume
- `POST /assess` - Submit assessment answers
- `POST /roadmap` - Generate career roadmap
- `POST /download_roadmap` - Download roadmap file