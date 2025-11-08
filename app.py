import os
import json
import random
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
import traceback
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from groq import Groq
import uuid
import razorpay
from dotenv import load_dotenv
import calendly
import google.generativeai as genai
import re
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string



load_dotenv()

# Application Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    TEST_DURATION = 1800  # 30 minutes
    RAZORPAY_KEY_ID = os.environ.get('RAZORPAY_KEY_ID', 'rzp_test_6QHUKOTfZxR1DF')
    RAZORPAY_KEY_SECRET = os.environ.get('RAZORPAY_KEY_SECRET', 'YOUR_KEY_SECRET')
    CALENDLY_API_KEY = os.environ.get('CALENDLY_API_KEY', 'YOUR_CALENDLY_API_KEY')
    PAYMENT_MODE = os.environ.get('PAYMENT_MODE', 'live')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyB7JLNTFWY_Q5EFt-8J8kQDZ-UVNDpDZBY')
    MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
    DB_NAME = os.environ.get('DB_NAME', 'skills_assessment')
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
    EMAIL_USER = os.environ.get('EMAIL_USER', '')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')
    TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
    TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
    TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '')

# Initialize Flask App
app = Flask(__name__, static_folder='static')
app.config.from_object(Config)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Session configuration
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# OTP Verification Functions
def generate_otp():
    """Generate 6-digit OTP."""
    return ''.join(random.choices(string.digits, k=6))

def send_email_otp(email, otp):
    """Send OTP via email."""
    try:
        if not app.config['EMAIL_USER'] or not app.config['EMAIL_PASSWORD']:
            return False
            
        msg = MIMEMultipart()
        msg['From'] = app.config['EMAIL_USER']
        msg['To'] = email
        msg['Subject'] = "Email Verification - Smarrtif AI"
        
        body = f"Your verification code is: {otp}\n\nThis code will expire in 10 minutes."
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT'])
        server.starttls()
        server.login(app.config['EMAIL_USER'], app.config['EMAIL_PASSWORD'])
        server.sendmail(app.config['EMAIL_USER'], email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"[ERROR] Email OTP failed: {e}")
        return False

def send_sms_otp(phone, otp):
    """Send OTP via SMS using Twilio."""
    try:
        # Option 1: Twilio (Recommended)
        from twilio.rest import Client
        
        account_sid = app.config.get('TWILIO_ACCOUNT_SID')
        auth_token = app.config.get('TWILIO_AUTH_TOKEN')
        from_number = app.config.get('TWILIO_PHONE_NUMBER')
        
        if not all([account_sid, auth_token, from_number]):
            print("[ERROR] Twilio credentials not configured")
            return False
            
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=f"Your Smarrtif AI verification code is: {otp}. Valid for 10 minutes.",
            from_=from_number,
            to=f"+91{phone}"  # Assuming Indian numbers
        )
        print(f"[SUCCESS] SMS sent via Twilio: {message.sid}")
        return True
        
    except ImportError:
        print("[ERROR] Twilio not installed. Install with: pip install twilio")
        return False
    except Exception as e:
        print(f"[ERROR] SMS sending failed: {e}")
        return False

# MongoDB storage functions
def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def get_user_data(key, default=None):
    user_id = get_user_id()
    if user_sessions_collection is None:
        return session.get(key, default)
    try:
        doc = user_sessions_collection.find_one({'user_id': user_id})
        return doc.get(key, default) if doc else default
    except Exception as e:
        print(f"[ERROR] Failed to get user data: {e}")
        return session.get(key, default)

def set_user_data(key, value):
    user_id = get_user_id()
    # Store in MongoDB first, then session as backup
    if user_sessions_collection is not None:
        try:
            user_sessions_collection.update_one(
                {'user_id': user_id},
                {'$set': {key: value, 'updated_at': datetime.utcnow()}},
                upsert=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to set user data in MongoDB: {e}")
    
    # Only store essential data in session to avoid size limit
    if key in ['payment_verified', 'roadmap_unlocked', 'tech_field']:
        session[key] = value

# Initialize Razorpay Client
try:
    razorpay_client = razorpay.Client(
        auth=(app.config['RAZORPAY_KEY_ID'], app.config['RAZORPAY_KEY_SECRET'])
    )
    print(f"[SUCCESS] Razorpay client initialized with key: {app.config['RAZORPAY_KEY_ID'][:10]}...")
except Exception as e:
    print(f"[ERROR] Failed to initialize Razorpay client: {e}")
    razorpay_client = None

# Initialize Calendly Client
calendly_client = None
try:
    if app.config['CALENDLY_API_KEY'] and app.config['CALENDLY_API_KEY'] != 'YOUR_CALENDLY_API_KEY':
        calendly_client = calendly.Calendly(app.config['CALENDLY_API_KEY'])
except Exception:
    pass

try:
    # Use the original working URI temporarily
    uri = "mongodb+srv://smarrtifai_db_user:NPz75GhLTwm3dLQ8@cluster0.lzwqox9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    mongo_client = MongoClient(uri)
    db = mongo_client.skills_assessment
    users_collection = db.users
    assessments_collection = db.assessments
    user_sessions_collection = db.user_sessions
    mongo_client.admin.command('ping')
    print("[SUCCESS] MongoDB connected")
except Exception as e:
    print(f"[ERROR] MongoDB connection failed: {e}")
    # Use fallback or exit gracefully
    users_collection = None
    assessments_collection = None
    user_sessions_collection = None


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

# Initialize Gemini Client
genai.configure(api_key=app.config['GEMINI_API_KEY'])

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
You are a **professional assessment designer with 25+ years of experience** in developing advanced, resume-based evaluation questions for hiring and learning assessments.  
You specialize in creating *unique, realistic, and skill-relevant* questions that accurately measure a candidate’s ability to apply what they claim in their resume — whether they are from **technical** or **non-technical** backgrounds.

Your task is to generate **{num_questions} deeply personalized, resume-specific questions** that are **non-generic, insightful, and progressively challenging.**

---

### 📄 **RESUME SUMMARY**
- **Skills:** {skills_str if skills_str else 'Not specified'}
- **Projects:** {projects_str if projects_str else 'Not specified'}
- **Achievements:** {achievements_str if achievements_str else 'Not specified'}
- **Experience:** {internships_str if internships_str else 'Not specified'}

---

### 🎯 **PRIMARY OBJECTIVE**
Design assessment questions that:
- Test *authentic understanding* and *applied reasoning* rather than definitions or memorization.
- Are **grounded directly in the candidate’s resume** — each question must reference their actual projects, achievements, or experience.
- Are **professionally written** and realistic, as if designed by a seasoned interviewer with **25+ years of expertise**.
- Automatically adapt in tone:
  - For **technical resumes:** Ask implementation, design, debugging, optimization, or reasoning-based questions.
  - For **non-technical resumes:** Ask situational, behavioral, analytical, or management-based questions.

---

### ⚙️ **QUESTION GUIDELINES**
Each question **must**:
1. Reference the candidate’s own **projects, experiences, or listed skills**.
2. Be **unique** — every question should explore a different area or challenge.
3. Be **non-generic** — avoid textbook or HR-style questions.
   - ❌ Bad: “What is teamwork?”
   - ✅ Good: “In your role managing the NGO Club at IILM, how did you ensure team accountability under pressure?”
   - ✅ Good: “In your NO₂ air quality prediction model, how did you handle data gaps caused by cloudy satellite images?”
4. Maintain **progressive difficulty**:
   - **Basic:** Recall or explanation-based (understanding what they did)
   - **Intermediate:** Application or scenario-based reasoning
   - **Advanced:** Strategic, troubleshooting, or optimization-based questions
5. Stay **balanced and professional** — written in the confident, precise tone of a **25+ year expert**.

---

### 💡 **QUESTION TYPES TO INCLUDE**
Automatically adjust depending on the resume content:

**For Technical Candidates:**
- Implementation and optimization questions about their listed projects  
- Data preprocessing, model evaluation, or algorithm selection reasoning  
- Debugging or troubleshooting real-world technical issues  
- Design and scalability decisions based on their experience  
- Performance trade-offs and best practices  

**For Non-Technical Candidates:**
- Behavioral questions on teamwork, leadership, communication, and initiative  
- Situational reasoning on deadlines, conflict resolution, or decision-making  
- Analytical thinking and prioritization in real-life work situations  
- Reflection on achievements, learning, and impact  

---

### 🧩 **OUTPUT FORMAT (JSON only)**
Return a **valid JSON list** where each object follows this schema:

[
  {{
    "question": "A realistic, resume-based question",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct": 0,
    "difficulty": "basic|intermediate|advanced",
    "skill_area": "Specific skill, role, or project name from resume"
  }}
]

---

### 🧠 **EXAMPLES**

**Technical Example:**
- “In your movie recommendation system built using Scikit-learn, why might cosine similarity outperform Euclidean distance when comparing user preferences?”
- “During your NO₂ air quality mapping project with PyTorch, how would you validate model accuracy when ground sensor data is missing?”
- “When your AI-based interview prep platform started giving inconsistent quiz recommendations, what metrics would you check first?”

**Non-Technical Example:**
- “As Sponsorship Lead for IILM University, how did you handle sponsor negotiations when targets were not being met?”
- “During your NGO event coordination, how did you ensure volunteers remained motivated throughout the campaign?”
- “In your internship at ABC Enterprises, what approach did you take to manage communication across different departments?”

---

### 🚀 **FINAL TASK**
Generate exactly **{num_questions}** *resume-specific, unique, and progressively difficult* questions.  
Each question must:
- Be directly grounded in their resume,  
- Be appropriate for their background (technical or non-technical),  
- Be professional, insightful, and realistic,  
- Reflect the experience and tone of a **seasoned assessment designer with 25+ years of expertise**.
"""


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
            
            # Handle response properly - avoid accessing .text directly due to Part object issue
            text = None
            try:
                # Try to get candidates first
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                text_parts.append(part.text)
                        text = ''.join(text_parts).strip()
                
                # Fallback: try parts directly
                if not text and hasattr(response, 'parts') and response.parts:
                    text_parts = []
                    for part in response.parts:
                        if hasattr(part, 'text'):
                            text_parts.append(part.text)
                    text = ''.join(text_parts).strip()
                    
            except Exception as part_error:
                print(f"[WARNING] Part access failed: {part_error}")
                # Last resort: try direct text access with error handling
                try:
                    text = str(response).strip()
                except:
                    raise ValueError("Could not extract text from response")
            
            if not text:
                raise ValueError("No text content in response")
            
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
            print(f"[SUCCESS] Gemini API generated roadmap for {tech_field}")
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
def home():
    """Home page."""
    if session.get('user_authenticated'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login')
def login():
    """Login and signup page."""
    if session.get('user_authenticated'):
        return redirect(url_for('dashboard'))
    return render_template('auth.html')

@app.route('/auth')
def auth():
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    """User dashboard - main assessment page."""
    if not session.get('user_authenticated'):
        return redirect(url_for('login'))
    
    if 'onboarding_step' not in session:
        session['onboarding_step'] = 0
    
    return render_template('index.html')

@app.route('/assessment')
def assessment_page():
    """Assessment page."""
    return render_template('assessment.html')

@app.route('/take-assessment')
def take_assessment():
    """Assessment test page - requires authentication."""
    if not session.get('user_authenticated'):
        return redirect(url_for('login'))
    
    return render_template('assessment.html')

@app.route('/upload-resume')
def upload_resume_page():
    """Resume upload page."""
    return render_template('upload.html')

@app.route('/upload')
def upload_page():
    return redirect(url_for('upload_resume_page'))

@app.route('/assessment-results')
def assessment_results():
    """Assessment results page."""
    if not session.get('user_authenticated'):
        return redirect(url_for('login'))
    
    return render_template('results.html')

@app.route('/results')
def results_page():
    """Redirect /results to /assessment-results"""
    return redirect(url_for('assessment_results'))

@app.route('/career-roadmap')
def career_roadmap():
    """Career roadmap page."""
    print(f"[DEBUG] Roadmap access - Unlocked: {get_user_data('roadmap_unlocked')}, Payment: {get_user_data('payment_verified')}")
    
    if not get_user_data('roadmap_unlocked') or not get_user_data('payment_verified'):
        print(f"[WARNING] Roadmap access denied - redirecting to bill")
        return redirect(url_for('bill_page'))
    
    roadmap_data = get_user_data('roadmap_data')
    if not roadmap_data:
        print(f"[ERROR] No roadmap data found")
        return redirect(url_for('bill_page'))
    
    calendly_url = get_user_data('calendly_scheduling_url', "https://calendly.com/diekshapriyaamishra-smarrtifai/smarrtif-ai-services-discussion")
    return render_template('roadmap.html', calendly_url=calendly_url)

@app.route('/roadmap')
def roadmap_page():
    return redirect(url_for('career_roadmap'))

@app.route('/schedule')
def schedule_page():
    """Schedule page."""
    return render_template('schedule.html')

@app.route('/post-payment')
def post_payment_page():
    """Page shown after successful payment."""
    if not get_user_data('roadmap_unlocked'):
        return redirect(url_for('payment_page'))
    calendly_url = get_user_data('calendly_scheduling_url', "https://calendly.com/diekshapriyaamishra-smarrtifai/smarrtif-ai-services-discussion")
    return render_template('post_payment.html', calendly_url=calendly_url)






@app.route('/api/onboarding', methods=['POST'])
def handle_onboarding():
    """Handle conversational onboarding flow."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        step = data.get('step', 0)
        response = data.get('response', '')
        
        # Store interests when provided
        if step == 2 and response:
            interests = [interest.strip() for interest in response.split(',') if interest.strip()]
            set_user_data('user_interests', interests)

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
                    return jsonify({'complete': True, 'redirect': '/upload-resume'})
                elif 'assessment' in choice.lower():
                    return jsonify({'complete': True, 'redirect': '/take-assessment'})
                else:
                    return jsonify({'complete': True, 'redirect': '/upload-resume'})
            
            return jsonify({'complete': True, 'message': 'Let\'s get started with your personalized journey!'})
            
    except Exception as e:
        print(f"[ERROR] Onboarding error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/assessment/interest-based', methods=['POST'])
def start_interest_based_assessment():
    """Generate an assessment based on user interests from the session."""
    interests = get_user_data('user_interests', [])
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
                set_user_data('assessment_data', {
                    'skills': interests,
                    'questions': questions[:num_questions], 'test_duration': 1800
                })
                return jsonify({'redirect': '/take-assessment'})
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

    set_user_data('assessment_data', {
        'skills': ['Communication', 'Problem Solving', 'Leadership'], 
        'questions': questions, 'test_duration': 1800
    })
    return jsonify({'redirect': '/assessment'})

@app.route('/api/mentor', methods=['POST'])
def ai_mentor():
    """AI mentor for instant career advice using Groq."""
    if not client:
        return jsonify({'response': 'The AI Mentor is currently unavailable. Please try again later.'})

    data = request.json
    question = data.get('question', '')

    # Get context from MongoDB
    results = get_user_data('assessment_results', {})
    tech_field = get_user_data('tech_field', 'Not selected yet')

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
        print(f"[INFO] Upload request received")
        if 'resume' not in request.files:
            print(f"[ERROR] No resume file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            print(f"[ERROR] Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"[INFO] Processing file: {file.filename}")
        
        # Validate file size
        if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
        
        filename = secure_filename(file.filename)
        if not filename or '..' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
            
        # Validate file extension
        allowed_extensions = {'.pdf', '.docx'}
        file_ext = '.' + filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Unsupported file format. Please upload PDF or DOCX files.'}), 400
        
        # Generate safe filename with timestamp
        safe_filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(file_path)
        
        # Extract text based on file type
        text = ''
        if file_ext == '.pdf':
            text = FileProcessor.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            text = FileProcessor.extract_text_from_docx(file_path)
        
        print(f"[INFO] Extracted text length: {len(text) if text else 0}")
        if not text or len(text.strip()) < 20:  # Reduced minimum length
            print(f"[WARNING] Text too short, using fallback")
            text = "General professional with communication and problem-solving skills."
        
        # AI Processing: Extract skills, education, domain expertise
        extracted_data = SkillExtractor.extract_skills_from_text(text)
        
        # If no skills found, use fallback skills
        if not extracted_data['skills']:
            extracted_data['skills'] = ['Communication', 'Problem Solving', 'Leadership', 'Teamwork']
        
        # AI Processing: Generate 30 tailored MCQs
        questions = QuestionGenerator.generate_questions_with_groq(extracted_data, 30)
        
        # Always ensure we have questions (use fallback if needed)
        if not questions or len(questions) < 10:
            questions = QuestionGenerator.generate_fallback_questions(extracted_data, 30)
        
        # Store assessment data in MongoDB
        set_user_data('assessment_data', {
            'skills': extracted_data['skills'],
            'questions': questions,
            'test_duration': 1800
        })
        
        print(f"[SUCCESS] Assessment data stored in session for user: {session.get('user_email')}")
        print(f"[INFO] Skills found: {extracted_data['skills']}")
        print(f"[INFO] Questions generated: {len(questions)}")
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        print(f"[INFO] Redirecting to /take-assessment")
        return jsonify({'redirect': '/take-assessment'})
    
    except Exception as e:
        print(f"[ERROR] Upload error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

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
    
    assessment_results = {
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
    
    # Store in session AND MongoDB user_sessions
    session['assessment_results'] = assessment_results
    set_user_data('assessment_results', assessment_results)
    
    print(f"[INFO] Assessment results stored: {assessment_results}")
    
    return jsonify({'redirect': '/assessment-results'})

@app.route('/api/roadmap', methods=['POST'])
def request_roadmap():
    """Stores the selected tech field and redirects to the bill."""
    data = request.json
    tech_field = data.get('tech_field')
    
    if not tech_field:
        return jsonify({'error': 'No tech field selected'}), 400
        
    set_user_data('tech_field', tech_field)
    return jsonify({'redirect': '/bill'})

@app.route('/bill')
def bill_page():
    """Display the bill before payment."""
    invoice_id = f"INV-{uuid.uuid4().hex[:8].upper()}"
    return render_template('bill.html', invoice_id=invoice_id)

@app.route('/payment')
def payment_page():
    """Create a Razorpay order and render the payment page."""
    if razorpay_client is None:
        print("[ERROR] Razorpay client not initialized")
        return redirect(url_for('payment_failure_page'))
        
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
        return redirect(url_for('payment_failure_page'))

def send_invoice_email(user_email, user_name, payment_id, amount):
    """Send invoice email after successful payment."""
    try:
        if not app.config['EMAIL_USER'] or not app.config['EMAIL_PASSWORD']:
            print("[WARNING] Email credentials not configured")
            return False
            
        # Create message
        msg = MIMEMultipart()
        msg['From'] = app.config['EMAIL_USER']
        msg['To'] = user_email
        msg['Subject'] = f"Payment Invoice - Smarrtif AI Skills Assessment Platform"
        
        # Email body
        body = f"""
        Dear {user_name},
        
        Thank you for your payment! Here are your transaction details:
        
        Payment ID: {payment_id}
        Amount: ₹{amount/100:.2f}
        Service: Career Consultation & Roadmap Service
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Your personalized career roadmap is now available in your account.
        
        Best regards,
        Smarrtif AI Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT'])
        server.starttls()
        server.login(app.config['EMAIL_USER'], app.config['EMAIL_PASSWORD'])
        text = msg.as_string()
        server.sendmail(app.config['EMAIL_USER'], user_email, text)
        server.quit()
        
        print(f"[SUCCESS] Invoice email sent to: {user_email}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to send invoice email: {e}")
        return False

def generate_and_store_calendly_link():
    """Generate Calendly link with fallback to default URL."""
    default_url = "https://calendly.com/diekshapriyaamishra-smarrtifai/smarrtif-ai-services-discussion"
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
        # Verify payment signature
        razorpay_client.utility.verify_payment_signature(params_dict)
        print(f"[SUCCESS] Payment verified successfully. Payment ID: {payment_id}")
        
        # Double check payment status with Razorpay
        payment_details = razorpay_client.payment.fetch(payment_id)
        if payment_details['status'] != 'captured':
            print(f"[ERROR] Payment not captured. Status: {payment_details['status']}")
            return redirect(url_for('payment_failure_page'))
        
        # Set payment success flag first
        set_user_data('payment_verified', True)
        set_user_data('payment_id', payment_id)
        
        tech_field = get_user_data('tech_field')
        results = get_user_data('assessment_results', {})
        skill_level = results.get('level', 'Beginner')
        skills = results.get('skills', [])
        
        print(f"[INFO] Generating roadmap for {tech_field} at {skill_level} level")
        # Always generate roadmap regardless of API success
        try:
            roadmap = RoadmapGenerator.generate_roadmap_with_gemini(tech_field, skill_level, skills)
        except Exception as e:
            print(f"[WARNING] Roadmap generation failed, using fallback: {e}")
            roadmap = RoadmapGenerator.get_fallback_roadmap(tech_field, skill_level)
        
        # Store roadmap and unlock status
        set_user_data('roadmap_data', roadmap)
        set_user_data('roadmap_unlocked', True)
        set_user_data('calendly_scheduling_url', "https://calendly.com/diekshapriyaamishra-smarrtifai/smarrtif-ai-services-discussion")
        
        print(f"[SUCCESS] Payment processing complete. Roadmap unlocked.")
        return redirect(url_for('career_roadmap'))
        
    except razorpay.errors.SignatureVerificationError as e:
        print(f"[ERROR] Payment verification failed: {e}")
        return redirect(url_for('payment_failure_page'))
    except Exception as e:
        print(f"[ERROR] Unexpected error in payment processing: {e}")
        return redirect(url_for('payment_failure_page'))

@app.route('/payment-failed')
def payment_failure_page():
    """Payment failure page"""
    return render_template('failure.html')

@app.route('/success')
def payment_success():
    """Payment success page"""
    # This page is not used in the new flow, but we keep it for now.
    return redirect(url_for('roadmap_page'))




@app.route('/api/download_roadmap', methods=['GET', 'POST'])
def download_roadmap():
    if not get_user_data('roadmap_unlocked'):
        return jsonify({'error': 'Roadmap not unlocked. Please complete the payment first.'}), 403
    
    data = get_user_data('roadmap_data', {})
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
    assessment_data = get_user_data('assessment_data', {})
    
    # If no assessment data, create fallback
    if not assessment_data:
        print(f"[WARNING] No assessment data found, creating fallback")
        fallback_questions = QuestionGenerator.generate_fallback_questions({}, 30)
        assessment_data = {
            'skills': ['Communication', 'Problem Solving', 'Leadership'],
            'projects': [],
            'achievements': [],
            'internships': [],
            'questions': fallback_questions,
            'test_duration': 1800
        }
        set_user_data('assessment_data', assessment_data)
    
    print(f"[INFO] Returning assessment data with {len(assessment_data.get('questions', []))} questions")
    return jsonify(assessment_data)

@app.route('/api/get_results_data')
def get_results_data():
    """Get results data for current session."""
    # Try session first, then MongoDB
    results = session.get('assessment_results', {})
    if not results:
        results = get_user_data('assessment_results', {})
    return jsonify(results)

@app.route('/api/get_roadmap_data')
def get_roadmap_data():
    """Get roadmap data for current session."""
    data = get_user_data('roadmap_data', {})
    # Add unlocked flag to align with flowchart after payment confirmation
    wrapped = dict(data)
    is_unlocked = bool(get_user_data('roadmap_unlocked'))
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
    user_id = get_user_id()
    try:
        user_sessions_collection.delete_one({'user_id': user_id})
    except:
        pass
    session.clear()
    return jsonify({'success': True})

@app.route('/api/debug_session')
def debug_session():
    """Debug route to check session data - REMOVE IN PRODUCTION."""
    # Only allow in development
    if os.environ.get('FLASK_ENV') != 'development':
        return jsonify({'error': 'Not available in production'}), 403
        
    user_id = get_user_id()
    # Don't expose sensitive session data
    safe_session = {
        'user_authenticated': session.get('user_authenticated'),
        'user_email': session.get('user_email'),
        'has_assessment_results': bool(session.get('assessment_results'))
    }
    
    return jsonify({
        'session': safe_session,
        'user_id': user_id
    })



@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Handle login API request."""
    data = request.json
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password required'}), 400
    
    try:
        if users_collection is None:
            return jsonify({'success': False, 'message': 'Database unavailable'}), 503
            
        print(f"[INFO] Attempting MongoDB login for: {email}")
        user = users_collection.find_one({'email': email})
        
        if user and check_password_hash(user['password'], password):
            session.clear()
            session['user_authenticated'] = True
            session['user_email'] = email
            session['user_name'] = user['name']
            session['user_id'] = str(user['_id'])
            session.permanent = True
            print(f"[SUCCESS] MongoDB login successful for: {email}")
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
            
    except Exception as e:
        print(f"[ERROR] Login error: {e}")
        return jsonify({'success': False, 'message': 'Login failed. Please try again.'}), 500

@app.route('/api/auth/signup', methods=['POST'])
def api_signup():
    """Handle signup API request with verification."""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
            
        name = data.get('name', '').strip()
        email = data.get('email', '').lower().strip()
        mobile = data.get('mobile', '').strip()
        password = data.get('password', '')
        
        if not name or not email or not mobile or not password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
            
        if len(mobile) != 10 or not mobile.isdigit():
            return jsonify({'success': False, 'message': 'Please enter a valid 10-digit mobile number'}), 400
            
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
        
        # Check verification status
        email_verified = session.get('email_verified', False)
        phone_verified = session.get('phone_verified', False)
        
        if not email_verified:
            return jsonify({'success': False, 'message': 'Please verify your email first', 'require_verification': 'email'}), 400
        
        if not phone_verified:
            return jsonify({'success': False, 'message': 'Please verify your phone number first', 'require_verification': 'phone'}), 400
        
        if users_collection is None:
            return jsonify({'success': False, 'message': 'Database unavailable'}), 503
            
        print(f"[INFO] Attempting MongoDB signup for: {email}")
        if users_collection.find_one({'email': email}):
            return jsonify({'success': False, 'message': 'Email already registered'}), 409
        
        user_data = {
            'name': name,
            'email': email,
            'mobile': mobile,
            'password': generate_password_hash(password),
            'email_verified': True,
            'phone_verified': True,
            'created_at': datetime.utcnow(),
            'last_login': None
        }
        
        result = users_collection.insert_one(user_data)
        user_id = str(result.inserted_id)
        print(f"[SUCCESS] User saved to MongoDB: {email} with ID: {user_id}")
        
        session.clear()
        session['user_authenticated'] = True
        session['user_email'] = email
        session['user_name'] = name
        session['user_id'] = user_id
        session.permanent = True
        
        return jsonify({'success': True, 'message': 'Account created successfully'})
        
    except Exception as e:
        print(f"[ERROR] Signup error: {e}")
        return jsonify({'success': False, 'message': 'Account creation failed. Please try again.'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """Handle logout API request."""
    session.pop('user_authenticated', None)
    session.pop('user_email', None)
    session.pop('user_name', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/auth/send-otp', methods=['POST'])
def send_otp():
    """Send OTP for email/phone verification."""
    data = request.json
    email = data.get('email', '').lower().strip()
    phone = data.get('phone', '').strip()
    verification_type = data.get('type', 'email')  # 'email' or 'phone'
    
    if verification_type == 'email' and not email:
        return jsonify({'success': False, 'message': 'Email required'}), 400
    if verification_type == 'phone' and not phone:
        return jsonify({'success': False, 'message': 'Phone number required'}), 400
    
    otp = generate_otp()
    expiry = datetime.utcnow() + timedelta(minutes=10)
    
    # Store OTP in session
    session[f'otp_{verification_type}'] = otp
    session[f'otp_{verification_type}_expiry'] = expiry
    session[f'otp_{verification_type}_target'] = email if verification_type == 'email' else phone
    
    # Send OTP
    if verification_type == 'email':
        success = send_email_otp(email, otp)
        message = 'OTP sent to email' if success else 'Failed to send email OTP'
    else:
        success = send_sms_otp(phone, otp)
        message = 'OTP sent to phone' if success else 'Failed to send SMS OTP'
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/auth/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP for email/phone."""
    data = request.json
    otp = data.get('otp', '').strip()
    verification_type = data.get('type', 'email')
    
    if not otp:
        return jsonify({'success': False, 'message': 'OTP required'}), 400
    
    stored_otp = session.get(f'otp_{verification_type}')
    expiry = session.get(f'otp_{verification_type}_expiry')
    
    if not stored_otp or not expiry:
        return jsonify({'success': False, 'message': 'No OTP found. Please request a new one.'}), 400
    
    if datetime.utcnow() > expiry:
        return jsonify({'success': False, 'message': 'OTP expired. Please request a new one.'}), 400
    
    if otp != stored_otp:
        return jsonify({'success': False, 'message': 'Invalid OTP'}), 400
    
    # Mark as verified
    session[f'{verification_type}_verified'] = True
    
    # Clean up OTP data
    session.pop(f'otp_{verification_type}', None)
    session.pop(f'otp_{verification_type}_expiry', None)
    
    return jsonify({'success': True, 'message': f'{verification_type.title()} verified successfully'})

@app.route('/Assets/<filename>')
def assets(filename):
    """Serve assets files."""
    return send_file(os.path.join('Assets', filename))

@app.route('/schedule_meeting', methods=['POST'])
def schedule_meeting():
    """Schedule a meeting with a mentor."""
    default_url = "https://calendly.com/diekshapriyaamishra-smarrtifai/smarrtif-ai-services-discussion"
    return jsonify({'scheduling_url': default_url})



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
