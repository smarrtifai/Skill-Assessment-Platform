import os
import json
import random
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
import traceback
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from groq import Groq
import uuid
import razorpay
import hmac
import hashlib
from dotenv import load_dotenv
import calendly
import google.generativeai as genai
import threading
import time
import re
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor

load_dotenv()

# Application Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    TEST_DURATION = 1800  # 30 minutes
    RAZORPAY_KEY_ID = os.environ.get('RAZORPAY_KEY_ID', 'YOUR_KEY_ID')
    RAZORPAY_KEY_SECRET = os.environ.get('RAZORPAY_KEY_SECRET', 'YOUR_KEY_SECRET')
    CALENDLY_API_KEY = os.environ.get('CALENDLY_API_KEY', 'YOUR_CALENDLY_API_KEY')
    PAYMENT_MODE = os.environ.get('PAYMENT_MODE', 'live')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyB7JLNTFWY_Q5EFt-8J8kQDZ-UVNDpDZBY')

# Initialize Flask App
app = Flask(__name__, static_folder='static')
app.config.from_object(Config)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Razorpay Client
class MockRazorpayClient:
    def __init__(self, auth):
        pass

    class Order:
        def create(self, data):
            return {'id': f'order_{uuid.uuid4()}'}

    order = Order()

if app.config['PAYMENT_MODE'] == 'test':
    razorpay_client = MockRazorpayClient(auth=('', ''))
else:
    razorpay_client = razorpay.Client(
        auth=(app.config['RAZORPAY_KEY_ID'], app.config['RAZORPAY_KEY_SECRET'])
    )

# Initialize Calendly Client with error handling
def get_calendly_client():
    """Initialize Calendly client with error handling."""
    api_key = app.config['CALENDLY_API_KEY']
    if not api_key or api_key == 'YOUR_CALENDLY_API_KEY':
        return None
    try:
        return calendly.Calendly(api_key)
    except Exception:
        return None

calendly_client = get_calendly_client()


# Initialize Groq Client
client = None
groq_api_key = os.getenv('GROQ_API_KEY')
if groq_api_key:
    try:
        client = Groq(api_key=groq_api_key)
        print("[SUCCESS] Groq client initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq client: {e}")
else:
    print("[WARNING] GROQ_API_KEY not found in environment")

# Initialize Gemini Client with timeout configuration
genai.configure(
    api_key=app.config['GEMINI_API_KEY'],
    transport='rest'  # Use REST transport for better timeout handling
)

# Data Models for Non-Technical Skills Assessment
NON_TECH_SKILLS_DATABASE = {
    'communication': ['presented', 'wrote', 'negotiated', 'collaborated', 'documented', 'authored', 'public speaking', 'writing'],
    'leadership': ['managed', 'led', 'supervised', 'mentored', 'directed', 'coordinated', 'oversaw', 'team lead'],
    'project management': ['planned', 'scheduled', 'budgeted', 'delivered', 'launched', 'coordinated', 'managed projects'],
    'sales': ['sold', 'revenue', 'targets', 'clients', 'customers', 'business development', 'account management'],
    'marketing': ['campaigns', 'branding', 'social media', 'content', 'advertising', 'market research', 'digital marketing'],
    'human resources': ['recruitment', 'hiring', 'employee', 'training', 'performance', 'hr policies', 'talent acquisition'],
    'customer service': ['support', 'customer satisfaction', 'complaints', 'service quality', 'client relations'],
    'finance': ['budget', 'financial analysis', 'accounting', 'cost management', 'financial planning', 'excel'],
    'operations': ['process improvement', 'efficiency', 'workflow', 'operations management', 'logistics'],
    'analytical skills': ['analyzed', 'interpreted', 'forecasted', 'modeled', 'quantified', 'data analysis', 'reporting'],
    'problem-solving': ['solved', 'resolved', 'troubleshot', 'critical thinking', 'decision making'],
    'teamwork': ['collaborated', 'partnered', 'team player', 'worked with', 'cross-functional'],
    'creativity': ['designed', 'created', 'innovated', 'creative solutions', 'brainstorming']
}

# Technical skills for career transition context
TECH_SKILLS_DATABASE = {
    'python': ['programming', 'development', 'scripting', 'automation'],
    'javascript': ['web development', 'frontend', 'backend', 'node.js'],
    'java': ['programming', 'enterprise', 'spring', 'android'],
    'sql': ['database', 'queries', 'data analysis'],
    'machine learning': ['ai', 'ml', 'data science', 'algorithms'],
    'react': ['frontend', 'web development', 'javascript'],
    'aws': ['cloud', 'devops', 'infrastructure'],
    'docker': ['containerization', 'devops', 'deployment']
}

TECH_FIELDS = {
    'AI/ML': {
        'skills': ['python', 'sql', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'statistics', 'pandas', 'numpy'],
        'roadmap': {
            'beginner': [
                'Python Programming Fundamentals',
                '• Variables, data types, control structures',
                '• Functions, modules, and packages',
                '• Object-oriented programming basics',
                'Mathematics & Statistics Foundation',
                '• Descriptive statistics, probability, linear algebra, calculus',
                'SQL Database Fundamentals',
                '• SELECT, JOINs, Aggregate functions',
                'Data Manipulation with Pandas & NumPy',
                '• DataFrames, Series, and numerical operations'
            ],
            'intermediate': [
                'Machine Learning Fundamentals',
                '• Supervised & Unsupervised learning (Scikit-learn)',
                '• Model evaluation, cross-validation, and feature engineering',
                'Data Visualization Mastery',
                '• Matplotlib, Seaborn for static plots',
                '• Plotly for interactive dashboards',
                'Deep Learning & Neural Networks',
                '• TensorFlow and PyTorch frameworks',
                '• CNN for computer vision, RNN/LSTM for sequence data'
            ],
            'advanced': [
                'MLOps & Production Systems',
                '• Model versioning (MLflow/DVC), CI/CD pipelines',
                '• Containerization with Docker/Kubernetes',
                '• Model monitoring and drift detection',
                'Big Data Technologies',
                '• Apache Spark for distributed computing',
                '• Cloud platforms (AWS/GCP/Azure ML)',
                'Advanced Topics & Research',
                '• Natural Language Processing (NLP) with Transformers',
                '• Reinforcement learning, Computer Vision'
            ]
        }
    },
    'Business Analytics and Data Analytics': {
        'skills': ['sql', 'excel', 'tableau', 'power bi', 'python', 'statistics', 'communication'],
        'roadmap': {
            'beginner': [
                'SQL for Data Analytics',
                '• Advanced SELECT, JOINs, subqueries, window functions',
                'Data Visualization with Tableau/Power BI',
                '• Creating dashboards, calculated fields, connecting to data sources',
                'Statistical Fundamentals',
                '• Descriptive and inferential statistics, A/B testing',
                'Python for Data Analysis with Pandas',
                '• Data cleaning, manipulation, and Exploratory Data Analysis (EDA)'
            ],
            'intermediate': [
                'Advanced Data Visualization',
                '• Advanced charts, performance optimization, storytelling',
                'Predictive Analytics',
                '• Regression for forecasting, classification for prediction',
                '• Time series analysis',
                'Business Acumen and Storytelling',
                '• Translating business questions into data problems',
                '• Communicating insights to non-technical stakeholders'
            ],
            'advanced': [
                'Advanced Analytics Techniques',
                '• Market basket analysis, customer segmentation, LTV',
                'Data Engineering for Analytics',
                '• Building ETL/ELT pipelines, data warehousing concepts',
                '• Data governance and quality',
                'Leadership and Strategy',
                '• Leading analytics teams, defining strategy, measuring ROI'
            ]
        }
    },
    'Gen AI': {
        'skills': ['python', 'llms', 'langchain', 'transformers', 'prompt engineering', 'vector databases'],
        'roadmap': {
            'beginner': [
                'Introduction to LLMs & Transformers',
                '• Basics of neural networks and the Transformers architecture',
                'Prompt Engineering Fundamentals',
                '• Crafting effective prompts, few-shot prompting, chain-of-thought',
                'Working with LLM APIs',
                '• Using OpenAI, Cohere, or other LLM APIs',
                'Vector Databases & RAG',
                '• Understanding embeddings and building Retrieval-Augmented Generation'
            ],
            'intermediate': [
                'Fine-tuning LLMs',
                '• Understanding when and how to fine-tune models',
                '• Using libraries like Hugging Face TRL',
                'Frameworks like LangChain/LlamaIndex',
                '• Building complex LLM chains and data-aware applications',
                'Multimodal Models',
                '• Working with models that understand text, images, and audio',
                '• Building applications with models like GPT-4V or LLaVA'
            ],
            'advanced': [
                'LLM-powered Agents',
                '• Designing and building autonomous agents (planning, memory, tools)',
                'LLMOps',
                '• Deploying, monitoring, and managing LLMs in production',
                '• Cost management and performance optimization',
                'Advanced Topics',
                '• Quantization, model optimization, and research in foundation models'
            ]
        }
    },
    'Agentic AI': {
        'skills': ['python', 'ai agents', 'reasoning', 'planning', 'langchain', 'autogen', 'crewai'],
        'roadmap': {
            'beginner': [
                'Strong Python & OOP Skills',
                '• Master object-oriented and asynchronous programming',
                'Foundations of LLMs and Advanced Prompting',
                '• Deep understanding of LLMs and advanced prompt engineering',
                'Introduction to AI Agents',
                '• Core components: Planning, Memory, Tool Use',
                'Agentic Frameworks (Level 1)',
                '• Building basic agents with LangChain or LlamaIndex'
            ],
            'intermediate': [
                'Advanced Planning Algorithms',
                '• ReAct (Reasoning and Acting), Chain-of-Thought, Tree-of-Thought',
                'Memory Systems for Agents',
                '• Short-term and long-term memory using vector stores',
                'Tool Use and Function Calling',
                '• Integrating external APIs and tools reliably',
                'Multi-Agent Systems',
                '• Designing collaborative agents with AutoGen or CrewAI'
            ],
            'advanced': [
                'Self-Improving Agents',
                '• Implementing feedback loops for self-correction and learning',
                'Hierarchical Agent Architectures',
                '• Designing manager/worker agent teams for complex tasks',
                'Agent Evaluation and Testing',
                '• Building frameworks to test agent performance and reliability',
                'Productionizing AI Agents',
                '• State management, monitoring, security, and human-in-the-loop design'
            ]
        }
    }
}

SAMPLE_QUESTIONS = {
    'python': {
        'technical': [
            {
    "question": "What will be the output of the following code?\n\nx = [1, 2, 3]\nprint(x * 2)",
    "options": ['[1, 2, 3, 1, 2, 3]', '[2, 4, 6]', 'Error', '[1, 1, 2, 2, 3, 3]'],
    "correct": 0
} ,
            {'question': 'Which keyword is used to define a function in Python?', 'options': ['function', 'def', 'define', 'func'], 'correct': 1},
            {'question': 'What does len() function return?', 'options': ['Size in bytes', 'Number of elements', 'Memory address', 'Data type'], 'correct': 1},
            {'question': 'Which Python data structure is ordered and mutable?', 'options': ['tuple', 'list', 'set', 'frozenset'], 'correct': 1},
            {'question': 'What is the correct syntax for a lambda function?', 'options': ['lambda x: x*2', 'def lambda x: x*2', 'lambda(x): x*2', 'x => x*2'], 'correct': 0}
        ],
        'project': [
            {'question': 'In a Python web application, which framework would you choose for rapid development?', 'options': ['Django', 'Flask', 'FastAPI', 'Depends on requirements'], 'correct': 3},
            {'question': 'How would you handle database connections in a Python project?', 'options': ['Direct SQL queries', 'ORM like SQLAlchemy', 'Connection pooling', 'All of the above'], 'correct': 3},
            {'question': 'What approach would you take for error handling in a production Python application?', 'options': ['Try-except blocks only', 'Logging + exception handling', 'Ignore minor errors', 'Print statements'], 'correct': 1},
            {'question': 'How would you optimize a slow Python script processing large datasets?', 'options': ['Use pandas vectorization', 'Implement multiprocessing', 'Use NumPy arrays', 'All of the above'], 'correct': 3}
        ],
        'achievement': [
            {'question': 'If you improved application performance by 50%, what metrics would you track?', 'options': ['Response time only', 'CPU and memory usage', 'User satisfaction', 'Response time, throughput, resource usage'], 'correct': 3},
            {'question': 'How would you measure the success of a code optimization project?', 'options': ['Lines of code reduced', 'Performance benchmarks', 'User feedback', 'Performance + maintainability metrics'], 'correct': 3},
            {'question': 'What would indicate successful implementation of automated testing?', 'options': ['100% code coverage', 'Reduced bug reports', 'Faster deployment', 'Coverage + quality + speed'], 'correct': 3}
        ],
        'internship': [
            {'question': 'During a Python-focused internship, what was the most valuable non-technical skill you learned?', 'options': ['Time management', 'Team collaboration', 'Presenting technical concepts to non-technical people', 'All of the above'], 'correct': 3},
            {'question': 'Reflecting on an internship project using Python, what is one thing you would do differently if you started again?', 'options': ['Write more unit tests from the beginning', 'Choose a different primary library', 'Spend less time on initial setup', 'Focus only on my assigned tasks'], 'correct': 0}
        ]
    },
    'javascript': {
        'technical': [
            {'question': 'Which method adds an element to the end of an array?', 'options': ['push()', 'add()', 'append()', 'insert()'], 'correct': 0},
            {'question': 'What is the correct way to declare a variable in JavaScript?', 'options': ['var x;', 'variable x;', 'v x;', 'declare x;'], 'correct': 0},
            {'question': 'Which method removes the last element from an array?', 'options': ['pop()', 'remove()', 'delete()', 'shift()'], 'correct': 0},
            {'question': 'What does "===" operator do in JavaScript?', 'options': ['Assignment', 'Equality without type conversion', 'Equality with type conversion', 'Not equal'], 'correct': 1}
        ],
        'project': [
            {'question': 'How would you handle state management in a large React application?', 'options': ['useState only', 'Redux/Context API', 'Local storage', 'Global variables'], 'correct': 1},
            {'question': 'What approach would you take for API error handling in a web app?', 'options': ['Try-catch blocks', 'Error boundaries + interceptors', 'Alert messages', 'Console logging'], 'correct': 1},
            {'question': 'How would you optimize a JavaScript application for better performance?', 'options': ['Minification only', 'Code splitting + lazy loading', 'Remove comments', 'Use jQuery'], 'correct': 1}
        ],
        'achievement': [
            {'question': 'How would you measure improved user experience in a web application?', 'options': ['Page load time', 'User engagement metrics', 'Code quality', 'Load time + engagement + usability'], 'correct': 3},
            {'question': 'What metrics would show successful implementation of responsive design?', 'options': ['Mobile traffic increase', 'Cross-device compatibility', 'User satisfaction', 'All device metrics + performance'], 'correct': 3}
        ],
        'internship': [
            {'question': 'In your JavaScript internship, how did you ensure your code was maintainable for other developers?', 'options': ['By using many comments', 'By following a style guide and writing clear functions', 'By using the latest experimental features', 'By writing all code in a single file'], 'correct': 1},
            {'question': 'What was the biggest challenge you faced when working with a large existing JavaScript codebase during your internship?', 'options': ['Understanding the build process', 'Navigating the component structure', 'Dealing with legacy code or dependencies', 'All of the above'], 'correct': 3}
        ]
    },
    'sql': {
        'technical': [
            {'question': 'Which SQL statement is used to extract data from a database?', 'options': ['EXTRACT', 'SELECT', 'GET', 'OPEN'], 'correct': 1},
            {'question': 'Which clause is used to filter records in SQL?', 'options': ['FILTER', 'WHERE', 'HAVING', 'CONDITION'], 'correct': 1},
            {'question': 'Which SQL keyword is used to sort the result-set?', 'options': ['SORT', 'ORDER BY', 'SORT BY', 'ORDER'], 'correct': 1}
        ],
        'project': [
            {'question': 'How would you optimize a slow-running SQL query in production?', 'options': ['Add more RAM', 'Create indexes and analyze execution plan', 'Use NoSQL instead', 'Restart database'], 'correct': 1}
        ],
        'achievement': [
            {'question': 'How would you measure database performance improvements?', 'options': ['Query execution time', 'Throughput and response time', 'Database size', 'Execution time + throughput + resource usage'], 'correct': 3}
        ]
    },
    'java': {
        'technical': [
            {'question': 'Which keyword is used to create a class in Java?', 'options': ['class', 'Class', 'create', 'new'], 'correct': 0},
            {'question': 'What is the correct way to create an object in Java?', 'options': ['MyClass obj = new MyClass();', 'MyClass obj = MyClass();', 'new MyClass() obj;', 'create MyClass obj;'], 'correct': 0}
        ],
        'project': [
            {'question': 'How would you handle memory management in a large Java application?', 'options': ['Ignore it', 'JVM tuning and profiling', 'Use more RAM', 'Restart application'], 'correct': 1}
        ],
        'achievement': [
            {'question': 'How would you measure application scalability improvements?', 'options': ['User count', 'Response time under load', 'Code lines', 'Load capacity + performance metrics'], 'correct': 3}
        ]
    },
    'machine learning': {
        'technical': [
            {'question': 'What does supervised learning require?', 'options': ['Unlabeled data', 'Labeled data', 'No data', 'Random data'], 'correct': 1},
            {'question': 'Which algorithm is used for classification?', 'options': ['K-means', 'Decision Tree', 'PCA', 'DBSCAN'], 'correct': 1}
        ],
        'project': [
            {'question': 'How would you handle overfitting in a machine learning project?', 'options': ['Use more data', 'Cross-validation and regularization', 'Ignore it', 'Use simpler model'], 'correct': 1}
        ],
        'achievement': [
            {'question': 'How would you measure ML model performance improvement?', 'options': ['Accuracy only', 'Precision, recall, F1-score', 'Training time', 'Comprehensive metrics + business impact'], 'correct': 3}
        ]
    },
    'react': {
        'technical': [
            {'question': 'What is JSX in React?', 'options': ['JavaScript XML', 'Java Syntax Extension', 'JSON Extension', 'JavaScript Extension'], 'correct': 0},
            {'question': 'Which method is used to update state in React?', 'options': ['updateState()', 'setState()', 'changeState()', 'modifyState()'], 'correct': 1}
        ],
        'project': [
            {'question': 'How would you optimize React app performance?', 'options': ['Use more components', 'Memoization and code splitting', 'Add more CSS', 'Use jQuery'], 'correct': 1}
        ],
        'achievement': [
            {'question': 'How would you measure React app performance improvements?', 'options': ['Bundle size', 'Load time and user metrics', 'Component count', 'Performance + user experience metrics'], 'correct': 3}
        ]
    },
    'aws': {
        'technical': [
            {'question': 'What does EC2 stand for?', 'options': ['Elastic Compute Cloud', 'Electronic Commerce Cloud', 'Elastic Container Cloud', 'Enterprise Compute Cloud'], 'correct': 0},
            {'question': 'Which AWS service is used for object storage?', 'options': ['EC2', 'S3', 'RDS', 'Lambda'], 'correct': 1}
        ],
        'project': [
            {'question': 'How would you design a scalable architecture on AWS?', 'options': ['Single EC2 instance', 'Auto-scaling with load balancers', 'Manual scaling', 'Use largest instance'], 'correct': 1}
        ],
        'achievement': [
            {'question': 'How would you measure cloud migration success?', 'options': ['Migration time', 'Cost savings and performance', 'Service count', 'Cost + performance + reliability metrics'], 'correct': 3}
        ]
    },
    'docker': {
        'technical': [
            {'question': 'What is a Docker container?', 'options': ['Virtual machine', 'Lightweight, portable execution environment', 'Database', 'Web server'], 'correct': 1},
            {'question': 'Which command is used to build a Docker image?', 'options': ['docker create', 'docker build', 'docker make', 'docker compile'], 'correct': 1}
        ],
        'project': [
            {'question': 'How would you implement Docker in a CI/CD pipeline?', 'options': ['Manual deployment', 'Automated build and deployment', 'Copy files manually', 'Use FTP'], 'correct': 1}
        ],
        'achievement': [
            {'question': 'How would you measure containerization benefits?', 'options': ['Container count', 'Deployment speed and consistency', 'Image size', 'Speed + reliability + resource efficiency'], 'correct': 3}
        ]
    },
    'cybersecurity': {
        'technical': [
            {'question': 'What is the primary purpose of a firewall?', 'options': ['To block unauthorized access to a network', 'To speed up internet connection', 'To store user passwords', 'To detect viruses on a computer'], 'correct': 0},
            {'question': 'What does DDoS stand for?', 'options': ['Distributed Denial of Service', 'Data Denial of Security', 'Direct Denial of Service', 'Dynamic Domain of Service'], 'correct': 0},
            {'question': 'What is the main difference between symmetric and asymmetric encryption?', 'options': ['Symmetric uses one key, asymmetric uses two', 'Asymmetric is faster', 'Symmetric is more secure', 'Asymmetric is only for text'], 'correct': 0}
        ]
    },
    'gen ai': {
        'technical': [
            {'question': 'What is a "transformer" architecture in the context of LLMs?', 'options': ['A type of database', 'A neural network architecture based on self-attention', 'A data compression algorithm', 'A hardware accelerator'], 'correct': 1},
            {'question': 'What is Retrieval-Augmented Generation (RAG)?', 'options': ['A method for training models faster', 'A technique to fine-tune models on new data', 'A process to retrieve external knowledge to ground LLM responses', 'A way to encrypt model weights'], 'correct': 2},
            {'question': 'What is "prompt engineering"?', 'options': ['A type of software engineering', 'The process of designing and refining inputs for an AI model', 'A hardware design principle', 'A method for debugging code'], 'correct': 1}
        ]
    },
    'agentic ai': {
        'technical': [
            {'question': 'What are the core components of a typical AI agent?', 'options': ['Planning, Memory, and Tool Use', 'A user interface and a database', 'A CPU and RAM', 'A programming language and a compiler'], 'correct': 0},
            {'question': 'What is the "ReAct" (Reasoning and Acting) framework for agents?', 'options': ['A JavaScript library for UIs', 'A framework for building web servers', 'A paradigm that combines reasoning traces and actions for tasks', 'A type of memory storage'], 'correct': 2},
            {'question': 'What is the role of "memory" in an autonomous agent?', 'options': ['To store the agent\'s source code', 'To remember past interactions and learn from them', 'To cache web pages', 'To increase processing speed'], 'correct': 1}
        ]
    },
    'business analytics': {
        'technical': [
            {'question': 'What is the primary difference between a dashboard and a report?', 'options': ['Dashboards are static, reports are interactive', 'Dashboards provide a high-level visual overview, reports provide detailed data', 'Dashboards are for executives, reports are for analysts', 'There is no difference'], 'correct': 1},
            {'question': 'What is A/B testing primarily used for?', 'options': ['To test database performance', 'To compare two versions of a webpage or app to see which performs better', 'To check for bugs in code', 'To train machine learning models'], 'correct': 1},
            {'question': 'What does ETL stand for in the context of data warehousing?', 'options': ['Extract, Transform, Load', 'Execute, Test, Launch', 'Estimate, Track, Limit', 'Encode, Transmit, Link'], 'correct': 0}
        ]
    }
}

# Utility Functions
class FileProcessor:
    """Handle file processing operations."""
    
    @staticmethod
    def extract_text_from_pdf(file_path):
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    @staticmethod
    def extract_text_from_docx(file_path):
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            return "\n".join(text_parts)
        except Exception as e:
            print(f"DOCX extraction error: {e}")
            return ""

class SkillExtractor:
    """Extract skills and experience from resume text."""
    
    @staticmethod
    def extract_skills_from_text(text):
        """Extract skills, projects, and achievements from resume."""
        text_lower = text.lower()
        found_skills = set()
        projects = []
        achievements = []
        internships = []
        
        # Extract non-technical skills (primary focus)
        for skill, keywords in NON_TECH_SKILLS_DATABASE.items():
            if skill in text_lower or any(keyword in text_lower for keyword in keywords):
                found_skills.add(skill.replace('_', ' ').title())
        
        # Extract technical skills (for career transition context)
        for skill, keywords in TECH_SKILLS_DATABASE.items():
            if skill in text_lower or any(keyword in text_lower for keyword in keywords):
                found_skills.add(skill.capitalize())

        # Extract project experience
        project_keywords = ['project', 'developed', 'built', 'created', 'implemented', 'designed']
        for keyword in project_keywords:
            if keyword in text_lower:
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence.strip()) > 20:
                        projects.append(sentence.strip()[:100] + '...' if len(sentence) > 100 else sentence.strip())
                        break
        
        # Extract achievements
        achievement_keywords = ['achieved', 'improved', 'increased', 'reduced', 'optimized', 'award', 'recognition']
        for keyword in achievement_keywords:
            if keyword in text_lower:
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence.strip()) > 15:
                        achievements.append(sentence.strip()[:100] + '...' if len(sentence) > 100 else sentence.strip())
                        break
        
        # Extract internships
        internship_keywords = ['intern', 'internship', 'trainee', 'apprenticeship']
        for keyword in internship_keywords:
            if keyword in text_lower:
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence.strip()) > 20:
                        internships.append(sentence.strip()[:100] + '...' if len(sentence) > 100 else sentence.strip())
                        break
        
        # Extract education
        education = []
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'certification']
        for keyword in education_keywords:
            if keyword in text_lower:
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence.strip()) > 10:
                        education.append(sentence.strip()[:80] + '...' if len(sentence) > 80 else sentence.strip())
                        break
        
        # Extract domain expertise
        domain_keywords = ['experience in', 'expertise in', 'specialized in', 'domain knowledge']
        domain = []
        for keyword in domain_keywords:
            if keyword in text_lower:
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence.strip()) > 15:
                        domain.append(sentence.strip()[:80] + '...' if len(sentence) > 80 else sentence.strip())
                        break
        
        return {
            'skills': sorted(list(found_skills)),
            'projects': projects[:3],
            'achievements': achievements[:3],
            'internships': internships[:2],
            'education': education[:2],
            'domain': domain[:2]
        }

class QuestionGenerator:
    """Generate assessment questions."""
    
    @staticmethod
    def generate_questions_with_groq(extracted_data, num_questions=30):
        """Generate tailored MCQs using Groq based on extracted data."""
        if not client:
            print("[WARNING] Groq client not available. Using fallback questions.")
            return QuestionGenerator.generate_fallback_questions(extracted_data, num_questions)

        skills_str = ", ".join(extracted_data.get('skills', []))
        projects_str = "\n".join([f"- {p}" for p in extracted_data.get('projects', [])])
        achievements_str = "\n".join([f"- {a}" for a in extracted_data.get('achievements', [])])
        internships_str = "\n".join([f"- {i}" for i in extracted_data.get('internships', [])])

        prompt = f"""
You are an expert assessment designer creating UNIQUE questions based on this specific resume. Each question must test the candidate's actual expertise in their stated skills and experiences.

**RESUME CONTENT:**
- **Skills:** {skills_str if skills_str else 'Not specified'}
- **Projects:** {projects_str if projects_str else 'Not specified'}
- **Achievements:** {achievements_str if achievements_str else 'Not specified'}
- **Experience:** {internships_str if internships_str else 'Not specified'}

**CRITICAL REQUIREMENTS:**
1. **NO GENERIC QUESTIONS** - Every question must reference their specific resume content
2. **TEST REAL EXPERTISE** - Questions should reveal if they actually did what they claim
3. **UNIQUE & NON-REPEATABLE** - Each question tests different aspects, no similar scenarios
4. **PROGRESSIVE DIFFICULTY** - Start with basic recall, move to application, end with expert judgment
5. **RESUME-SPECIFIC SCENARIOS** - Use their actual projects/achievements as question context

**Question Types to Generate:**
- Technical implementation questions about their specific projects
- Problem-solving scenarios based on their achievements
- Best practices questions for their stated skills
- Troubleshooting questions for technologies they claim to know
- Architecture/design questions for their experience level

**EXAMPLE STRUCTURE:**
"In your [SPECIFIC PROJECT FROM RESUME], when you [SPECIFIC ACHIEVEMENT], what was the most critical factor for success?"
"Based on your experience with [SPECIFIC SKILL], how would you approach [REALISTIC SCENARIO]?"
"Given your background in [SPECIFIC DOMAIN], what would be your first step when [EXPERT-LEVEL PROBLEM]?"

**Generate exactly {num_questions} UNIQUE questions that:**
- Reference specific resume elements (projects, skills, achievements)
- Test actual knowledge depth, not just concepts
- Progress from basic to expert-level scenarios
- Cannot be answered by someone who didn't do the work
- Cover different aspects of their claimed expertise

**JSON OUTPUT:**
[
  {{
    "question": "[RESUME-SPECIFIC QUESTION]",
    "options": ["[OPTION A]", "[OPTION B]", "[OPTION C]", "[OPTION D]"],
    "correct": [0-3],
    "difficulty": "basic|intermediate|advanced",
    "skill_area": "[SPECIFIC SKILL FROM RESUME]"
  }}
]"""
        try:
            print("[INFO] Generating questions with Groq...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON generator for technical skill assessment questions. You will only output a valid JSON list of objects."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.6
            )
            
            response_content = chat_completion.choices[0].message.content.strip()
            
            # Clean response - remove markdown formatting if present
            if response_content.startswith('```json'):
                response_content = response_content[7:]
            if response_content.endswith('```'):
                response_content = response_content[:-3]
            response_content = response_content.strip()
            
            parsed_json = json.loads(response_content)
            questions = parsed_json if isinstance(parsed_json, list) else parsed_json.get('questions', [])
 
            if not isinstance(questions, list) or not all('question' in q and 'options' in q and 'correct' in q for q in questions):
                 print("[ERROR] Groq response JSON is malformed or empty. Using fallback.")
                 return QuestionGenerator.generate_fallback_questions(extracted_data, num_questions)
 
            if len(questions) < 10:
                print(f"[ERROR] Groq returned only {len(questions)} questions. Using fallback.")
                return QuestionGenerator.generate_fallback_questions(extracted_data, num_questions)

            print(f"[SUCCESS] Generated {len(questions)} questions from Groq.")
            return questions[:num_questions]

        except Exception as e:
            print(f"[ERROR] Groq API call failed: {e}. Using fallback questions.")
            return QuestionGenerator.generate_fallback_questions(extracted_data, num_questions)

    @staticmethod
    def generate_fallback_questions(extracted_data, num_questions):
        """Generate non-technical fallback questions."""
        non_tech_questions = [
            {"question": "When working on a team project, how do you typically handle conflicting opinions?", "options": ["Avoid the conflict", "Listen to all viewpoints and find common ground", "Impose your own solution", "Let others decide"], "correct": 1},
            {"question": "How do you approach learning a new process or tool at work?", "options": ["Wait for formal training", "Ask colleagues for help immediately", "Experiment and research on your own first", "Avoid using it"], "correct": 2},
            {"question": "When presenting information to stakeholders, what's most important?", "options": ["Using complex terminology", "Making it clear and actionable", "Showing all available data", "Keeping it brief regardless of clarity"], "correct": 1},
            {"question": "How do you prioritize tasks when everything seems urgent?", "options": ["Work on the easiest tasks first", "Assess impact and deadlines systematically", "Work on whatever was assigned last", "Ask your manager to prioritize everything"], "correct": 1},
            {"question": "When you notice a process could be improved, what do you do?", "options": ["Keep working the old way", "Document the issue and propose solutions", "Complain to colleagues", "Change it without telling anyone"], "correct": 1},
            {"question": "How do you handle feedback that you disagree with?", "options": ["Ignore it completely", "Consider the perspective and discuss constructively", "Argue immediately", "Accept it without question"], "correct": 1},
            {"question": "When managing multiple deadlines, what's your approach?", "options": ["Work on everything simultaneously", "Create a priority matrix and timeline", "Focus only on the nearest deadline", "Ask for extensions on everything"], "correct": 1},
            {"question": "How do you ensure effective communication in emails?", "options": ["Write as much detail as possible", "Use clear subject lines and concise language", "Copy everyone who might be interested", "Use technical jargon to sound professional"], "correct": 1},
            {"question": "When leading a meeting, what's most important?", "options": ["Talking the most to show expertise", "Having a clear agenda and keeping discussions focused", "Making it as long as possible", "Avoiding difficult topics"], "correct": 1},
            {"question": "How do you approach problem-solving in unfamiliar situations?", "options": ["Guess and hope for the best", "Break down the problem and research systematically", "Immediately ask someone else to solve it", "Avoid the situation entirely"], "correct": 1}
        ]
        
        # Repeat questions to reach desired count
        questions = (non_tech_questions * (num_questions // len(non_tech_questions) + 1))[:num_questions]
        random.shuffle(questions)
        return questions

class AssessmentEvaluator:
    """Evaluate assessment results."""
    
    @staticmethod
    def assess_with_groq(skills, score, total, percentage):
        """Assess skill level using fallback for speed."""
        return AssessmentEvaluator.get_fallback_level(percentage)

    @staticmethod
    def get_fallback_level(percentage):
        """Fallback skill level determination."""
        if percentage >= 80:
            return 'Advanced'
        elif percentage >= 60:
            return 'Intermediate'
        else:
            return 'Beginner'
    
    @staticmethod
    def analyze_performance(answers, questions, skills):
        """Advanced performance analysis with skill-specific insights."""
        strengths = []
        weaknesses = []
        skill_gaps = []
        skill_scores = {}
        
        # Analyze by difficulty level and skill area
        basic_correct = intermediate_correct = advanced_correct = 0
        basic_total = intermediate_total = advanced_total = 0
        
        for i, answer in enumerate(answers):
            if i < len(questions):
                question = questions[i]
                is_correct = answer == question.get('correct', -1)
                difficulty = question.get('difficulty', 'basic')
                skill_area = question.get('skill_area', 'general')
                
                # Track by difficulty
                if difficulty == 'basic':
                    basic_total += 1
                    if is_correct: basic_correct += 1
                elif difficulty == 'intermediate':
                    intermediate_total += 1
                    if is_correct: intermediate_correct += 1
                elif difficulty == 'advanced':
                    advanced_total += 1
                    if is_correct: advanced_correct += 1
                
                # Track by skill area
                if skill_area not in skill_scores:
                    skill_scores[skill_area] = {'correct': 0, 'total': 0}
                skill_scores[skill_area]['total'] += 1
                if is_correct:
                    skill_scores[skill_area]['correct'] += 1
        
        # Determine strengths and weaknesses
        for skill, score_data in skill_scores.items():
            percentage = (score_data['correct'] / score_data['total']) * 100 if score_data['total'] > 0 else 0
            if percentage >= 75:
                strengths.append(skill.title())
            elif percentage < 50:
                weaknesses.append(skill.title())
        
        # Identify skill gaps based on difficulty progression
        if basic_total > 0 and (basic_correct / basic_total) < 0.6:
            skill_gaps.append('Foundational concepts')
        if intermediate_total > 0 and (intermediate_correct / intermediate_total) < 0.5:
            skill_gaps.append('Practical application')
        if advanced_total > 0 and (advanced_correct / advanced_total) < 0.4:
            skill_gaps.append('Strategic thinking')
        
        return strengths[:3], weaknesses[:3], skill_gaps[:3]

class RoadmapGenerator:
    """Generate career roadmaps."""
    
    @staticmethod
    def _generate_with_timeout(model, prompt, generation_config, timeout=20):
        """Generate content with timeout using threading."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = model.generate_content(prompt, generation_config=generation_config)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Gemini API call timed out after {timeout} seconds")
        
        if exception[0]:
            raise exception[0]
            
        return result[0]

    @staticmethod
    def generate_roadmap_with_gemini(tech_field, skill_level, skills):
        """Generate roadmap with robust JSON handling."""
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            prompt = f"""Generate a career roadmap for {tech_field} at {skill_level} level.

Respond with ONLY this JSON structure (no extra text):
{{
"field": "{tech_field}",
"current_level": "{skill_level}",
"roadmap": {{
"beginner": ["item1", "item2", "item3"],
"intermediate": ["item1", "item2", "item3"],
"advanced": ["item1", "item2", "item3"]
}},
"timeline": {{"beginner": "3-6 months", "intermediate": "6-12 months", "advanced": "12+ months"}},
"learning_resources": {{"beginner": ["resource1"], "intermediate": ["resource2"], "advanced": ["resource3"]}},
"project_ideas": {{"beginner": ["project1"], "intermediate": ["project2"], "advanced": ["project3"]}},
"skill_validation": {{"beginner": ["method1"], "intermediate": ["method2"], "advanced": ["method3"]}},
"recommended_skills": ["skill1", "skill2", "skill3"]
}}"""
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1500
                )
            )
            
            text = response.text.strip()
            
            # Extract JSON more carefully
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found")
            
            json_str = json_match.group(0)
            
            # Fix common JSON issues
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control chars
            json_str = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', json_str)  # Fix backslashes
            
            parsed = json.loads(json_str)
            return parsed
            
        except Exception as e:
            print(f"[ERROR] Gemini API failed: {e}. Using fallback.")
            return RoadmapGenerator.get_fallback_roadmap(tech_field, skill_level)

    @staticmethod
    def get_fallback_roadmap(tech_field, skill_level):
        """Generate fallback roadmap."""
        field_data = TECH_FIELDS.get(tech_field, TECH_FIELDS['AI/ML'])
        return {
            'field': tech_field,
            'current_level': skill_level,
            'roadmap': field_data['roadmap'],
            'recommended_skills': field_data['skills'],
            'timeline': {
                'beginner': '3-6 months',
                'intermediate': '6-12 months', 
                'advanced': '12+ months'
            },
            'learning_resources': {
                'beginner': ['Codecademy courses', 'FreeCodeCamp', 'YouTube tutorials', 'Official documentation'],
                'intermediate': ['Udemy advanced courses', 'Pluralsight', 'GitHub projects', 'Tech blogs'],
                'advanced': ['Coursera specializations', 'Research papers', 'Open source contributions', 'Industry conferences']
            },
            'project_ideas': {
                'beginner': ['Personal portfolio', 'Simple CRUD app', 'Basic calculator', 'To-do list'],
                'intermediate': ['E-commerce site', 'API development', 'Database optimization', 'Testing framework'],
                'advanced': ['Microservices architecture', 'Machine learning pipeline', 'Scalable systems', 'Performance optimization']
            },
            'resources': {
                'beginner': ['Online courses', 'Documentation', 'Practice projects'],
                'intermediate': ['Advanced courses', 'Open source contributions', 'Certifications'],
                'advanced': ['Specialization courses', 'Research papers', 'Industry projects']
            }
        }


# Routes
@app.route('/')
def index():
    """Landing page with onboarding."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session['onboarding_step'] = 0
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Resume upload page."""
    return render_template('upload.html')

@app.route('/assessment')
def assessment_page():
    """Assessment page."""
    return render_template('assessment.html')

@app.route('/results')
def results_page():
    """Results page."""
    return render_template('results.html')

@app.route('/roadmap')
def roadmap_page():
    """Roadmap page."""
    if not session.get('roadmap_unlocked'):
        return redirect(url_for('payment'))
    calendly_url = session.get('calendly_scheduling_url')
    return render_template('roadmap.html', calendly_url=calendly_url)

@app.route('/schedule')
def schedule_page():
    """Schedule page."""
    return render_template('schedule.html')

@app.route('/post-payment')
def post_payment_page():
    """Page shown after successful payment."""
    if not session.get('roadmap_unlocked'):
        return redirect(url_for('payment'))
    calendly_url = session.get('calendly_scheduling_url')
    return render_template('post_payment.html', calendly_url=calendly_url)


# ----- Flowchart alignment helpers (non-breaking stubs) ----- 
@app.route('/api/send_assessment_link', methods=['POST'])
def send_assessment_link():
    """Simulate sending assessment link to user's email as per flowchart.
    No external email is sent; we just acknowledge and provide a link token.
    """
    # In a real system, you would integrate an email service like SendGrid or Mailgun here.
    # For this simulation, we'll just confirm the action.
    session['assessment_link_sent'] = True
    return jsonify({
        'success': True,
        'message': 'An assessment link has been sent to your email (simulated).',
        # In a real app, you'd redirect to a "check your email" page.
        # For this flow, we'll just proceed.
        'redirect': url_for('upload_page')
    })

@app.route('/Assets/<filename>')
def assets(filename):
    """Serve assets files."""
    return send_file(os.path.join('Assets', filename))

@app.route('/api/onboarding', methods=['POST'])
def handle_onboarding():
    """Handle conversational onboarding flow."""
    data = request.json
    step = data.get('step', 0)
    response = data.get('response', '')
    
    # This is the step *after* the user has answered.
    # The question for interests is at step 1 (0-indexed).
    # The JS sends step+1, so when it sends step=2, it's the response for question 1.
    if step == 2:
        session['user_interests'] = [interest.strip() for interest in response.split(',')]

    onboarding_responses = {
        0: {
            'question': 'Hi! 👋 I\'m here to help you start your tech journey. What\'s your current situation?',
            'options': ["I'm new to tech", "I have some basic knowledge about tech", "I'm looking to switch career"],
            'type': 'single'
        },
        1: {
            'question': 'Great! What interests you most about technology? You can select multiple.',
            'options': [
                'Frontend Development', 'Backend Development', 'Full-Stack Development',
                'Machine Learning', 'Deep Learning', 'NLP',
                'Ethical Hacking', 'Network Security', 'Cloud Security',
                'LLM Development', 'Prompt Engineering', 'RAG Systems',
                'Data Visualization', 'Business Intelligence', 'Data Engineering',
                'Autonomous Agents', 'Multi-Agent Systems', 'DevOps', 'Cloud Computing'
            ],
            'type': 'multi'
        },
        2: {
            'question': 'Perfect! Do you have a resume or would you like to start with a quick skill assessment?',
            'options': ['I have a resume to upload', 'Let me take a quick assessment first', 'Tell me more about career paths'],
            'type': 'single'
        }
    }
    
    session['onboarding_step'] = step + 1
    
    if step < len(onboarding_responses):
        return jsonify(onboarding_responses[step])
    else:
        # Handle final choice and redirect
        if step == 3:
            choice = data.get('response', '')
            if 'resume' in choice.lower():
                return jsonify({'complete': True, 'redirect': '/upload'})
            elif 'assessment' in choice.lower():
                return jsonify({'complete': True, 'redirect': '/assessment'})
            else:
                return jsonify({'complete': True, 'redirect': '/upload'})
        
        return jsonify({'complete': True, 'message': 'Let\'s get started with your personalized journey!'})

@app.route('/api/assessment/interest-based', methods=['POST'])
def start_interest_based_assessment():
    """Generate an assessment based on user interests from the session."""
    interests = session.get('user_interests', [])
    if not interests:
        return jsonify({'error': 'No interests selected. Please complete the onboarding questions first.'}), 400

    num_questions = 30
    
    # Try to generate with Groq first
    if client:
        interests_str = ", ".join(interests)
        prompt = f"""
You are a career transition specialist evaluating NON-TECHNICAL professionals interested in exploring technology. Create questions that assess their transferable skills and readiness for tech roles.

**User Interests:** {interests_str}

**CRITICAL: Focus on NON-TECHNICAL skills assessment:**
- Communication and presentation abilities
- Problem-solving and analytical thinking
- Project management and organization
- Leadership and teamwork
- Customer service and relationship building
- Business understanding and process improvement
- Learning agility and adaptability

**Question Requirements:**
1. Generate exactly {num_questions} questions for someone with NO technical background
2. NO coding questions - focus on soft skills and business acumen
3. Use workplace scenarios from their current field
4. Test logical thinking through business problems
5. Assess communication, leadership, and problem-solving
6. Include questions about learning new tools/processes
7. Each question needs exactly 4 options
8. Output valid JSON only

**Example Format:**
[
  {{
    "question": "In your current role, when you need to explain a complex process to someone unfamiliar with it, what approach works best?",
    "options": ["Use technical jargon to sound professional", "Break it down into simple steps with examples", "Send them documentation to read", "Have someone else explain it"],
    "correct": 1
  }},
  {{
    "question": "When facing a problem you've never encountered before, what's your typical first step?",
    "options": ["Ask someone else to solve it", "Research and gather information", "Try random solutions", "Avoid the problem"],
    "correct": 1
  }}
]"""
        try:
            print(f"[INFO] Generating questions for interests ({interests_str}) with Groq...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a JSON generator for non-technical professional skill assessment questions. Focus on soft skills, business acumen, and transferable abilities. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.6
            )
            
            response_content = chat_completion.choices[0].message.content.strip()
            
            # Clean response - remove markdown formatting if present
            if response_content.startswith('```json'):
                response_content = response_content[7:]
            if response_content.endswith('```'):
                response_content = response_content[:-3]
            response_content = response_content.strip()
            
            parsed_json = json.loads(response_content)
            questions = parsed_json if isinstance(parsed_json, list) else parsed_json.get('questions', [])

            if isinstance(questions, list) and len(questions) >= 10 and all('question' in q and 'options' in q and 'correct' in q for q in questions):
                print(f"[SUCCESS] Generated {len(questions)} questions from Groq for interests.")
                session['assessment_data'] = {
                    'skills': interests, 'projects': [], 'achievements': [], 'internships': [],
                    'questions': questions[:num_questions], 'test_duration': 1800
                }
                return jsonify({'redirect': '/assessment'})
            else:
                print(f"[ERROR] Invalid Groq response. Got {len(questions) if isinstance(questions, list) else 0} questions.")
        except Exception as e:
            print(f"[ERROR] Groq API call for interests failed: {e}. Using fallback questions.")

    # Fallback to non-technical questions if Groq fails
    print("[WARNING] Using non-technical fallback questions.")
    
    # Generate non-technical questions based on soft skills
    non_tech_questions = [
        {
            "question": "When working on a team project, how do you typically handle conflicting opinions?",
            "options": ["Avoid the conflict", "Listen to all viewpoints and find common ground", "Impose your own solution", "Let others decide"],
            "correct": 1
        },
        {
            "question": "How do you approach learning a new process or tool at work?",
            "options": ["Wait for formal training", "Ask colleagues for help immediately", "Experiment and research on your own first", "Avoid using it"],
            "correct": 2
        },
        {
            "question": "When presenting information to stakeholders, what's most important?",
            "options": ["Using complex terminology", "Making it clear and actionable", "Showing all available data", "Keeping it brief regardless of clarity"],
            "correct": 1
        },
        {
            "question": "How do you prioritize tasks when everything seems urgent?",
            "options": ["Work on the easiest tasks first", "Assess impact and deadlines systematically", "Work on whatever was assigned last", "Ask your manager to prioritize everything"],
            "correct": 1
        },
        {
            "question": "When you notice a process could be improved, what do you do?",
            "options": ["Keep working the old way", "Document the issue and propose solutions", "Complain to colleagues", "Change it without telling anyone"],
            "correct": 1
        }
    ]
    
    # Repeat questions to reach desired count
    questions = (non_tech_questions * (num_questions // len(non_tech_questions) + 1))[:num_questions]

    session['assessment_data'] = {
        'skills': ['Communication', 'Problem Solving', 'Leadership', 'Teamwork'], 
        'projects': [], 'achievements': [], 'internships': [],
        'questions': questions, 'test_duration': 1800
    }
    return jsonify({'redirect': '/assessment'})

@app.route('/api/mentor', methods=['POST'])
def ai_mentor():
    """AI mentor for instant career advice using Groq."""
    if not client:
        return jsonify({'response': 'The AI Mentor is currently unavailable. Please try again later.'})

    data = request.json
    question = data.get('question', '')

    # Get context from session
    results = session.get('assessment_results', {})
    tech_field = session.get('tech_field', 'Not selected yet')

    skills_str = ", ".join(results.get('skills', []))
    level = results.get('level', 'Unknown')
    strengths_str = ", ".join(results.get('strengths', []))
    weaknesses_str = ", ".join(results.get('weaknesses', []))

    # Build a detailed prompt
    prompt = f"""
You are a helpful and encouraging AI Career Mentor for people in the tech industry.
Your goal is to provide clear, concise, and actionable advice.

Here is the user's context based on their recent skills assessment:
- **Assessed Skill Level:** {level}
- **Identified Skills:** {skills_str if skills_str else 'Not specified'}
- **Strengths:** {strengths_str if strengths_str else 'None identified'}
- **Areas for Improvement:** {weaknesses_str if weaknesses_str else 'None identified'}
- **Chosen Career Path:** {tech_field}

The user's question is: "{question}"

Based on their context, provide a helpful and encouraging response to their question.
Keep the response to 2-4 sentences.
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI Career Mentor providing advice to a user based on their skill assessment results."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7
        )
        response = chat_completion.choices[0].message.content
        return jsonify({'response': response})
    except Exception as e:
        print(f"[ERROR] AI Mentor (Groq) API call failed: {e}")
        return jsonify({'response': 'Sorry, I encountered an error while trying to answer. Please try again.'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_resume():
    """Handle resume upload and analysis."""
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file size
        if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
        
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
            
        # Validate file extension
        allowed_extensions = {'.pdf', '.docx'}
        file_ext = '.' + filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Unsupported file format. Please upload PDF or DOCX files.'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
        text = ''
        if file_ext == '.pdf':
            text = FileProcessor.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            text = FileProcessor.extract_text_from_docx(file_path)
        
        if not text or len(text.strip()) < 50:
            os.remove(file_path)  # Clean up
            return jsonify({'error': 'Could not extract text from file or file is too short. Please ensure the file contains readable text.'}), 400
        
        # AI Processing: Extract skills, education, domain expertise
        extracted_data = SkillExtractor.extract_skills_from_text(text)
        
        if not extracted_data['skills']:
            os.remove(file_path)  # Clean up
            return jsonify({'error': 'No technical skills found in resume. Please upload a technical resume with relevant skills.'}), 400
        
        # AI Processing: Generate 30 tailored MCQs
        questions = QuestionGenerator.generate_questions_with_groq(extracted_data, 30)
        
        if not questions or len(questions) < 10:
            os.remove(file_path)  # Clean up
            return jsonify({'error': 'Could not generate enough questions from resume. Please try with a more detailed technical resume.'}), 400
        
        # Store in session for assessment page
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        questions_filename = f"{session_id}_questions.json"
        questions_filepath = os.path.join(app.config['UPLOAD_FOLDER'], questions_filename)
        with open(questions_filepath, 'w') as f:
            json.dump(questions, f)

        session['assessment_data'] = {
            'skills': extracted_data['skills'],
            'projects': extracted_data['projects'], 
            'achievements': extracted_data['achievements'],
            'internships': extracted_data.get('internships', []),
            'education': extracted_data.get('education', []),
            'domain_expertise': extracted_data.get('domain', []),
            'questions_file': questions_filename,
            'session_id': session_id,
            'test_duration': 1800
        }
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({'redirect': '/assessment'})
    
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Log error
        traceback.print_exc()
        return jsonify({'error': 'Processing failed. Please try again with a different file.'}), 500

@app.route('/api/assess', methods=['POST'])
def assess_skills():
    """Evaluate assessment responses."""
    data = request.json
    answers = data.get('answers', [])
    questions = data.get('questions', [])
    skills = data.get('skills', [])
    
    # AI Processing: Evaluate results and identify strengths, weaknesses, skill gaps
    score = sum(1 for i, answer in enumerate(answers) 
                if i < len(questions) and answer == questions[i]['correct'])
    total = len(questions)
    percentage = (score / total) * 100 if total > 0 else 0
    
    # AI Analysis: Determine skill level and gaps
    level = AssessmentEvaluator.assess_with_groq(skills, score, total, percentage)
    strengths, weaknesses, skill_gaps = AssessmentEvaluator.analyze_performance(answers, questions, skills)
    
    session['assessment_results'] = {
        'score': score,
        'total': total,
        'percentage': percentage,
        'level': level,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'skills': skills,
        'skill_gaps': skill_gaps,
        'tech_fields': list(TECH_FIELDS.keys())
    }
    
    return jsonify({'redirect': '/results'})

@app.route('/api/roadmap', methods=['POST'])
def request_roadmap():
    """Stores the selected tech field and redirects to the bill."""
    data = request.json
    tech_field = data.get('tech_field')
    
    if not tech_field:
        return jsonify({'error': 'No tech field selected'}), 400
        
    session['tech_field'] = tech_field
    return jsonify({'redirect': '/bill'})

@app.route('/bill')
def bill_page():
    """Display the bill before payment."""
    invoice_id = f"INV-{uuid.uuid4().hex[:8].upper()}"
    return render_template('bill.html', invoice_id=invoice_id)

@app.route('/payment', methods=['GET'])
def payment():
    """Create a Razorpay order and render the payment page."""
    amount = 99900  # Amount in paise (499 INR)
    order_data = {
        'amount': amount,
        'currency': 'INR',
        'receipt': f'receipt_{uuid.uuid4().hex}',
        'payment_capture': 1
    }
    
    try:
        order = razorpay_client.order.create(data=order_data)
        session['razorpay_order_id'] = order['id']
        
        return render_template(
            'payment.html',
            order_id=order['id'],
            amount=amount,
            key_id=app.config['RAZORPAY_KEY_ID']
        )
    except Exception as e:
        print(f"Error creating order: {str(e)}")
        return "Error creating payment order", 500

def generate_and_store_calendly_link():
    """Generate Calendly link with fallback to default URL."""
    default_url = "https://calendly.com/diekshapriyaamishra-smarrtifai/smarrtif-ai-services-discussion"
    
    try:
        if not calendly_client:
            print("[WARNING] Calendly client not available. Using default URL.")
            session['calendly_scheduling_url'] = default_url
            return True
            
        user_info = calendly_client.about()
        user_uri = user_info['resource']['uri']
        event_types = calendly_client.get_event_types(user=user_uri)
        
        if not event_types:
            print("[WARNING] No event types found. Using default URL.")
            session['calendly_scheduling_url'] = default_url
            return True

        event_type_uri = event_types[0]['uri']
        payload = {
            "max_event_count": 1,
            "owner": event_type_uri,
            "owner_type": "EventType"
        }
        scheduling_link = calendly_client.create_scheduling_link(payload=payload)
        session['calendly_scheduling_url'] = scheduling_link['booking_url']
        return True
        
    except Exception as e:
        print(f"⚠️ Calendly API error: {e}. Using default URL.")
        session['calendly_scheduling_url'] = default_url
        return True

@app.route('/charge', methods=['POST'])
def charge():
    """Handle successful payment callback from Razorpay."""
    payment_id = request.form.get('razorpay_payment_id')
    order_id = request.form.get('razorpay_order_id')
    signature = request.form.get('razorpay_signature')

    params_dict = {
        'razorpay_order_id': order_id,
        'razorpay_payment_id': payment_id,
        'razorpay_signature': signature
    }

    try:
        razorpay_client.utility.verify_payment_signature(params_dict)
        
        tech_field = session.get('tech_field')
        results = session.get('assessment_results', {})
        skill_level = results.get('level', 'Beginner')
        skills = results.get('skills', [])
        
        if not tech_field or not results:
            # This should not happen in a normal flow
            return redirect(url_for('index'))

        # Generate roadmap with Gemini
        roadmap = RoadmapGenerator.generate_roadmap_with_gemini(tech_field, skill_level, skills)
        session['roadmap_data'] = roadmap
        session['roadmap_unlocked'] = True
        
        # Generate Calendly link
        generate_and_store_calendly_link()
        
        return redirect(url_for('roadmap_page'))
        
    except razorpay.errors.SignatureVerificationError as e:
        print(f"Payment verification failed: {e}")
        return redirect(url_for('payment'))

@app.route('/failure')
def payment_failure():
    """Payment failure page"""
    return render_template('failure.html')

@app.route('/success')
def payment_success():
    """Payment success page"""
    # This page is not used in the new flow, but we keep it for now.
    return redirect(url_for('roadmap_page'))


@app.route('/api/send_full_roadmap_email', methods=['POST'])
def send_full_roadmap_email():
    """Simulate emailing the full roadmap to the user after payment."""
    if not session.get('roadmap_unlocked'):
        return jsonify({'error': 'Roadmap not unlocked yet'}), 400
    # Create a text export similar to download and pretend to email
    data = session.get('roadmap_data', {})
    filename = f"roadmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    content = f"Full roadmap for {data.get('field','N/A')} generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    try:
        with open(filepath, 'w') as f:
            f.write(content)
    except Exception:
        pass
    return jsonify({'success': True, 'message': 'Full roadmap emailed (simulated).', 'attachment': filename})

@app.route('/api/download_roadmap', methods=['GET', 'POST'])
def download_roadmap():
    if not session.get('roadmap_unlocked'):
        return jsonify({'error': 'Roadmap not unlocked. Please complete the payment first.'}), 403
    
    data = session.get('roadmap_data', {})
    if not data:
        return jsonify({'error': 'No roadmap data found'}), 400
    
    filename = f"Career_Roadmap_{data.get('field', 'Tech').replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=HexColor('#ef4444'),
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=HexColor('#1f2937')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        textColor=HexColor('#ef4444')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6
    )
    
    story = []
    
    # Title
    story.append(Paragraph("🚀 PERSONALIZED CAREER ROADMAP", title_style))
    story.append(Spacer(1, 20))
    
    # Overview
    story.append(Paragraph("📋 OVERVIEW", heading_style))
    story.append(Paragraph(f"<b>Tech Field:</b> {data.get('field', 'N/A')}", body_style))
    story.append(Paragraph(f"<b>Current Level:</b> {data.get('current_level', 'N/A')}", body_style))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 20))
    
    # Learning Path
    story.append(Paragraph("📚 LEARNING PATH", heading_style))
    roadmap = data.get('roadmap', {})
    timeline = data.get('timeline', {})
    
    for level in ['beginner', 'intermediate', 'advanced']:
        level_items = roadmap.get(level, [])
        level_timeline = timeline.get(level, 'N/A')
        if level_items:
            story.append(Paragraph(f"🎯 {level.upper()} LEVEL ({level_timeline})", subheading_style))
            for item in level_items:
                story.append(Paragraph(f"• {item}", body_style))
            story.append(Spacer(1, 10))
    
    # Learning Resources
    learning_resources = data.get('learning_resources', {})
    if learning_resources:
        story.append(Paragraph("📖 LEARNING RESOURCES", heading_style))
        for level in ['beginner', 'intermediate', 'advanced']:
            resources = learning_resources.get(level, [])
            if resources:
                story.append(Paragraph(f"{level.upper()} Resources:", subheading_style))
                for resource in resources:
                    story.append(Paragraph(f"• {resource}", body_style))
                story.append(Spacer(1, 8))
    
    # Project Ideas
    project_ideas = data.get('project_ideas', {})
    if project_ideas:
        story.append(Paragraph("💡 PROJECT IDEAS", heading_style))
        for level in ['beginner', 'intermediate', 'advanced']:
            projects = project_ideas.get(level, [])
            if projects:
                story.append(Paragraph(f"{level.upper()} Projects:", subheading_style))
                for project in projects:
                    story.append(Paragraph(f"• {project}", body_style))
                story.append(Spacer(1, 8))
    
    # Skill Validation
    skill_validation = data.get('skill_validation', {})
    if skill_validation:
        story.append(Paragraph("✅ SKILL VALIDATION", heading_style))
        for level in ['beginner', 'intermediate', 'advanced']:
            validations = skill_validation.get(level, [])
            if validations:
                story.append(Paragraph(f"{level.upper()} Validation:", subheading_style))
                for validation in validations:
                    story.append(Paragraph(f"• {validation}", body_style))
                story.append(Spacer(1, 8))
    
    # Recommended Skills
    recommended_skills = data.get('recommended_skills', [])
    if recommended_skills:
        story.append(Paragraph("🎯 RECOMMENDED SKILLS", heading_style))
        for skill in recommended_skills:
            story.append(Paragraph(f"• {skill}", body_style))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("Generated by Smarrtif AI Skills Assessment Platform", body_style))
    story.append(Paragraph("For personalized mentoring, schedule a session with our experts!", body_style))
    
    # Build PDF
    doc.build(story)
    
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/api/get_assessment_data')
def get_assessment_data():
    """Get assessment data for current session."""
    assessment_data = session.get('assessment_data', {})
    if 'questions_file' in assessment_data:
        questions_filepath = os.path.join(app.config['UPLOAD_FOLDER'], assessment_data['questions_file'])
        try:
            with open(questions_filepath, 'r') as f:
                assessment_data['questions'] = json.load(f)
        except FileNotFoundError:
            return jsonify({'error': 'Assessment data not found. Please start over.'}), 404
        except json.JSONDecodeError:
            return jsonify({'error': 'Failed to load assessment data. Please start over.'}), 500
    return jsonify(assessment_data)

@app.route('/api/get_results_data')
def get_results_data():
    """Get results data for current session."""
    return jsonify(session.get('assessment_results', {}))

@app.route('/api/get_roadmap_data')
def get_roadmap_data():
    """Get roadmap data for current session."""
    data = session.get('roadmap_data', {})
    # Add unlocked flag to align with flowchart after payment confirmation
    wrapped = dict(data)
    is_unlocked = bool(session.get('roadmap_unlocked'))
    wrapped['unlocked'] = is_unlocked

    # If not unlocked, only return the beginner module
    if not is_unlocked and 'roadmap' in wrapped and isinstance(wrapped['roadmap'], dict):
        beginner_roadmap = wrapped['roadmap'].get('beginner', [])
        wrapped['roadmap'] = {'beginner': beginner_roadmap}
        # Also clear other related data if not unlocked
        wrapped['timeline'] = {'beginner': wrapped.get('timeline', {}).get('beginner', 'N/A')}
        wrapped['learning_resources'] = {'beginner': wrapped.get('learning_resources', {}).get('beginner', [])}
        wrapped['project_ideas'] = {'beginner': wrapped.get('project_ideas', {}).get('beginner', [])}
        wrapped['skill_validation'] = {'beginner': wrapped.get('skill_validation', {}).get('beginner', [])}
        wrapped['recommended_skills'] = wrapped.get('recommended_skills', []) # Keep all recommended skills

    print(f"[DEBUG] get_roadmap_data - Session keys: {list(session.keys())}")
    print(f"[DEBUG] get_roadmap_data - Roadmap data: {data}")
    print(f"[DEBUG] get_roadmap_data - Wrapped data: {wrapped}")
    
    return jsonify(wrapped)

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """Clear session data for new assessment."""
    session.clear()
    return jsonify({'success': True})

@app.route('/schedule_meeting', methods=['POST'])
def schedule_meeting():
    """Schedule a meeting with a mentor."""
    default_url = "https://calendly.com/diekshapriyaamishra-smarrtifai/smarrtif-ai-services-discussion"
    
    try:
        if not calendly_client:
            return jsonify({'scheduling_url': default_url})
            
        user_info = calendly_client.about()
        user_uri = user_info['resource']['uri']
        event_types = calendly_client.get_event_types(user=user_uri)
        
        if not event_types:
            return jsonify({'scheduling_url': default_url})

        event_type_uri = event_types[0]['uri']
        payload = {
            "max_event_count": 1,
            "owner": event_type_uri,
            "owner_type": "EventType"
        }
        scheduling_link = calendly_client.create_scheduling_link(payload=payload)
        return jsonify({'scheduling_url': scheduling_link['booking_url']})
        
    except Exception as e:
        print(f"⚠️ Calendly API error: {e}. Using default URL.")
        return jsonify({'scheduling_url': default_url})

@app.route('/schedule_meeting_page')
def schedule_meeting_page():
    """Render the Calendly scheduling page."""
    calendly_url = session.get('calendly_scheduling_url', "https://calendly.com/diekshapriyaamishra-smarrtifai/smarrtif-ai-services-discussion")
    return render_template('calendly_schedule.html', calendly_url=calendly_url)


@app.route('/calendly_webhook', methods=['POST'])
def calendly_webhook():
    """Handle Calendly webhooks."""
    data = request.json
    print(f"Received Calendly webhook: {data}")
    return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print('\nStarting Skills Assessment Platform...')
    print(f'Server starting on port: {port}')
    if debug:
        print('Development mode enabled')
    print('\nMake sure to:')
    print('   • Check your internet connection')
    print('   • Set GROQ_API_KEY environment variable')
    print('   • Upload a technical resume for best results\n')
    
    app.run(debug=debug, host='0.0.0.0', port=port)
