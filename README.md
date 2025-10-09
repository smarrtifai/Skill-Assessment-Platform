# Skills Assessment & Career Roadmap Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

An AI-powered platform to analyze your skills, identify gaps, and generate a personalized career roadmap. This application takes a user's resume or a skill assessment to provide a detailed analysis and a path forward to achieving their career goals in the tech industry.

![Skills Assessment Platform Screenshot](Assets/Screenshot%202025-09-24%20181235.png)

## 📋 Table of Contents

- [✨ Key Features](#-key-features)
- [🚀 How It Works](#-how-it-works)
- [🏆 Path to Certification](#-path-to-certification)
- [🛠️ Tech Stack](#️-tech-stack)
- [⚙️ Getting Started](#️-getting-started)
- [☁️ Deployment](#️-deployment)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## ✨ Key Features

-   **AI-Powered Resume Analysis**: Upload a resume (PDF or DOCX) to have it automatically parsed and analyzed for skills, project experience, and achievements.
-   **Dynamic Skill Assessment**: Instead of a resume, users can take a dynamically generated quiz based on their interests to evaluate their knowledge.
-   **Personalized Question Generation**: Utilizes the Groq API to generate relevant multiple-choice questions based on the user's resume or interests.
-   **In-Depth Performance Analysis**: Provides a detailed breakdown of the user's performance, highlighting strengths, weaknesses, and skill gaps.
-   **Custom Career Roadmaps**: Leverages the Google Gemini API to generate a comprehensive, step-by-step career roadmap tailored to the user's chosen tech field and skill level.
-   **Secure Payments**: Integrates with Razorpay for processing payments to unlock the full career roadmap.
-   **Meeting Scheduling**: Seamlessly schedule a follow-up meeting with a mentor using Calendly integration after payment.
-   **Conversational Onboarding**: A friendly and interactive onboarding experience to guide the user.

## 🚀 How It Works

1.  **Onboarding**: The user starts with a conversational onboarding to determine their interests and current situation.
2.  **Input**: The user can either upload their resume or take a quick, interest-based skill assessment.
3.  **Analysis & Assessment**: The platform analyzes the resume or quiz results to understand the user's skill set.
4.  **Results**: The user is presented with their assessment results, including their skill level and areas for improvement.
5.  **Roadmap Generation**: After selecting a desired tech field, the platform generates a personalized career roadmap. A portion of the roadmap is available for free, with the option to unlock the full roadmap via payment.
6.  **Scheduling**: Upon successful payment, the user is presented with a Calendly link to schedule a one-on-one session with a mentor.

## 🏆 Path to Certification

This platform is designed not just for skill assessment, but to guide you on your journey to becoming a certified professional. The personalized roadmaps are structured to align with the knowledge domains of popular industry certifications. By following your roadmap, you will build the necessary skills and project experience to confidently prepare for certifications in fields such as:

-   AWS Certified Cloud Practitioner
-   Certified Associate in Python Programming (PCAP)
-   TensorFlow Developer Certificate
-   And many more, depending on your chosen field.

## 🛠️ Tech Stack

-   **Backend**: Python, Flask
-   **AI & Machine Learning**:
    -   [Groq](https://groq.com/) for fast AI-powered question generation.
    -   [Google Gemini](https://gemini.google.com/) for advanced career roadmap generation.
-   **Frontend**: HTML, CSS, JavaScript
-   **Payment Gateway**: [Razorpay](https://razorpay.com/)
-   **Scheduling**: [Calendly](https://calendly.com/)
-   **Deployment**: [Render](https://render.com/)

## ⚙️ Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Python 3.9 or higher
-   A virtual environment tool (`venv`)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/skills-assessment-platform.git
cd skills-assessment-platform
```

### 2. Create and Activate a Virtual Environment

-   **On Windows**:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```
-   **On macOS/Linux**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a file named `.env` in the root of the project and add the following variables. These are essential for the application to function correctly.

```env
SECRET_KEY='your-super-secret-key'
RAZORPAY_KEY_ID='your-razorpay-key-id'
RAZORPAY_KEY_SECRET='your-razorpay-key-secret'
CALENDLY_API_KEY='your-calendly-api-key'
GEMINI_API_KEY='your-google-gemini-api-key'
GROQ_API_KEY='your-groq-api-key'
```

### 5. Run the Application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`.

## ☁️ Deployment

This application is configured for deployment on [Render](https://render.com/) using the `render.yaml` file. The `Procfile` is also included for deployment on platforms like Heroku.

The deployment process on Render will automatically build the application using `pip install -r requirements.txt` and start the server with `python app.py`. Remember to set the environment variables in the Render dashboard.

## 🤝 Contributing

Contributions are welcome! If you have a suggestion or want to improve the code, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
