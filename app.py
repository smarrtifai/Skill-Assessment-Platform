import os
import json
import random
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from groq import Groq
import uuid

# Application Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    TEST_DURATION = 1800  # 30 minutes

# Initialize Flask App
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Groq Client
def get_groq_client():
    """Initialize Groq client with error handling."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key or api_key == 'your-groq-api-key-here':
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

client = get_groq_client()

# Data Models
SKILLS_DATABASE = {
    'python': ['programming', 'development', 'scripting', 'automation'],
    'javascript': ['web development', 'frontend', 'backend', 'node.js'],
    'java': ['programming', 'enterprise', 'spring', 'android'],
    'sql': ['database', 'queries', 'data analysis'],
    'machine learning': ['ai', 'ml', 'data science', 'algorithms'],
    'react': ['frontend', 'web development', 'javascript'],
    'aws': ['cloud', 'devops', 'infrastructure'],
    'docker': ['containerization', 'devops', 'deployment']
}

SOFT_SKILLS_MAP = {
    'leadership': ['managed', 'led', 'supervised', 'mentored', 'directed', 'coordinated', 'oversaw'],
    'communication': ['presented', 'wrote', 'negotiated', 'collaborated', 'documented', 'authored'],
    'problem-solving': ['solved', 'analyzed', 'optimized', 'resolved', 'debugged', 'troubleshot'],
    'project management': ['planned', 'scheduled', 'budgeted', 'delivered', 'launched'],
    'teamwork': ['collaborated', 'partnered', 'team player', 'worked with'],
    'creativity': ['designed', 'created', 'innovated', 'prototyped'],
    'analytical skills': ['analyzed', 'interpreted', 'forecasted', 'modeled', 'quantified']
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
        
        # Extract hard skills
        for skill, keywords in SKILLS_DATABASE.items():
            if skill in text_lower or any(keyword in text_lower for keyword in keywords):
                found_skills.add(skill.capitalize())
        
        # Extract soft skills from action verbs
        for skill, verbs in SOFT_SKILLS_MAP.items():
            if any(verb in text_lower for verb in verbs):
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
            print("⚠️ Groq client not available. Using fallback questions.")
            return QuestionGenerator.generate_fallback_questions(extracted_data, num_questions)

        skills_str = ", ".join(extracted_data.get('skills', []))
        projects_str = "\n".join([f"- {p}" for p in extracted_data.get('projects', [])])
        achievements_str = "\n".join([f"- {a}" for a in extracted_data.get('achievements', [])])
        internships_str = "\n".join([f"- {i}" for i in extracted_data.get('internships', [])])

        prompt = f"""
You are an expert technical interviewer creating a skills assessment. Based on the following resume data, generate {num_questions} multiple-choice questions (MCQs) that cross-question their experience.

**Resume Data:**
- **Skills:** {skills_str if skills_str else 'Not specified'}
- **Projects:** {projects_str if projects_str else 'Not specified'}
- **Achievements:** {achievements_str if achievements_str else 'Not specified'}
- **Internships:** {internships_str if internships_str else 'Not specified'}

**Instructions:**
1. Generate exactly {num_questions} unique MCQs that are relevant to the provided skills, projects, and internships.
2. The questions should cover a range of difficulties from easy to hard.
3. Each question must have exactly 4 options.
4. For about 30-50% of the questions, include a relevant code snippet within the 'question' field to test practical application. The code snippet should be properly escaped for JSON, using `\\n` for newlines.
5. Indicate the correct answer using a zero-based index (0, 1, 2, or 3).
6. The output **must** be a valid JSON list of objects, and nothing else. Do not include any introductory text, explanations, or markdown formatting like ```json.

**JSON Format Example:**
[
  {{
    "question": "What is the primary purpose of a 'useEffect' hook in React?",
    "options": ["To manage component state", "To perform side effects in function components", "To declare a new component", "To handle routing"],
    "correct": 1
  }},
  {{
    "question": "What will be the output of the following Python code?\\n\\n```python\\nmy_list = [1, 2, 3]\\nprint(my_list[3])\\n```",
    "options": ["3", "None", "IndexError", "SyntaxError"],
    "correct": 2
  }},
  ...
]
"""
        try:
            print("🧠 Generating questions with Groq...")
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
                model="llama3-70b-8192",
                temperature=0.6,
                response_format={"type": "json_object"},
            )
            
            response_content = chat_completion.choices[0].message.content
            parsed_json = json.loads(response_content)

            questions = parsed_json.get('questions') if isinstance(parsed_json, dict) else parsed_json
 
            if not isinstance(questions, list) or not all('question' in q and 'options' in q and 'correct' in q for q in questions):
                 print("🚨 Groq response JSON is malformed or empty. Using fallback.")
                 return QuestionGenerator.generate_fallback_questions(extracted_data, num_questions)
 
            if len(questions) < num_questions:
                print(f"🚨 Groq returned only {len(questions)}/{num_questions} questions. Using fallback to ensure count.")
                return QuestionGenerator.generate_fallback_questions(extracted_data, num_questions)

            print(f"✅ Successfully generated {len(questions)} questions from Groq.")
            return questions

        except Exception as e:
            print(f"🚨 Groq API call failed: {e}. Using fallback questions.")
            return QuestionGenerator.generate_fallback_questions(extracted_data, num_questions)

    @staticmethod
    def generate_fallback_questions(extracted_data, num_questions):
        """Generate contextual questions, ensuring the final count is met."""
        skills = extracted_data.get('skills', [])
        projects = extracted_data.get('projects', [])
        achievements = extracted_data.get('achievements', [])
        internships = extracted_data.get('internships', [])
        
        specific_questions = []
        
        # 1. Gather questions for matched skills
        for skill in skills:
            if skill in SAMPLE_QUESTIONS:
                skill_data = SAMPLE_QUESTIONS[skill]
                if isinstance(skill_data, dict):
                    specific_questions.extend(skill_data.get('technical', []))
                    if projects:
                        specific_questions.extend(skill_data.get('project', []))
                    if achievements:
                        specific_questions.extend(skill_data.get('achievement', []))
                    if internships:
                        specific_questions.extend(skill_data.get('internship', []))
                else:
                    specific_questions.extend(skill_data)
        
        # 2. Create a unique set of questions based on the question text
        unique_questions_map = {q['question']: q for q in specific_questions}

        # 3. If not enough unique questions, supplement from the general pool
        if len(unique_questions_map) < num_questions:
            print(f"⚠️ Not enough specific questions ({len(unique_questions_map)}). Supplementing with general questions.")
            all_questions_pool = []
            for skill_data in SAMPLE_QUESTIONS.values():
                if isinstance(skill_data, dict):
                    for question_list in skill_data.values():
                        all_questions_pool.extend(question_list)
                else:
                    all_questions_pool.extend(skill_data)
            
            for q in all_questions_pool:
                if len(unique_questions_map) >= num_questions:
                    break
                if q['question'] not in unique_questions_map:
                    unique_questions_map[q['question']] = q
        
        # 4. Convert map to list, shuffle, and return the required number
        final_question_pool = list(unique_questions_map.values())
        random.shuffle(final_question_pool)
        
        return final_question_pool[:num_questions]

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
        """Analyze performance to identify strengths, weaknesses, and skill gaps."""
        strengths = []
        weaknesses = []
        skill_gaps = []
        
        # Simple analysis based on correct/incorrect answers
        correct_count = sum(1 for i, answer in enumerate(answers) 
                          if i < len(questions) and answer == questions[i]['correct'])
        
        if correct_count >= len(questions) * 0.8:
            strengths = skills[:3]  # Top skills as strengths
        elif correct_count >= len(questions) * 0.6:
            strengths = skills[:2]
            weaknesses = skills[2:3] if len(skills) > 2 else []
        else:
            weaknesses = skills[:2]
            skill_gaps = ['Problem solving', 'Technical depth', 'Best practices']
        
        return strengths, weaknesses, skill_gaps

class RoadmapGenerator:
    """Generate career roadmaps."""
    
    @staticmethod
    def generate_roadmap_with_groq(tech_field, skill_level, skills):
        """Generate detailed learning roadmap using Groq API."""
        if not client:
            print("⚠️ Groq client not available. Using fallback roadmap.")
            return RoadmapGenerator.get_fallback_roadmap(tech_field, skill_level)

        skills_str = ", ".join(skills) if skills else "General technical skills"
        
        prompt = f"""
You are a senior career mentor and expert curriculum designer. 
Create a **detailed, professional learning roadmap** for a beginner who wants to go from absolute basics to an advanced level in **{tech_field}**. 

📌 **Context**:
- User's current skill level: {skill_level}
- Identified skills: {skills_str}

🎯 **Your Goal**:
Design a step-by-step learning path that:
- Covers fundamentals, intermediate, and advanced topics in a logical sequence.
- Includes hands-on projects and challenges at every stage.
- Stays engaging, motivating, and achievable for a self-learner.

✅ **Roadmap Requirements**:
1. **Foundational Stage (Beginner)**  
   - Core topics explained simply  
   - Practical exercises, small tasks, and real-life examples  

2. **Intermediate Stage**  
   - More challenging topics, industry best practices  
   - Small-to-medium projects that build confidence  

3. **Advanced Stage**  
   - Expert-level techniques, tools, and optimization strategies  
   - Complex projects that can be added to a professional portfolio  

4. **Skill Validation**  
   - Self-assessment methods: quizzes, challenges, peer review, certifications  

5. **Learning Resources**  
   - Specific recommendations: books, blogs, YouTube channels, online courses, communities  

6. **Timeline**  
   - Approximate time for each stage assuming 5-10 hours/week  

7. **Output Format**  
Return a **valid JSON object** with the following structure only, no explanations, no markdown:

{{
  "field": "{tech_field}",
  "current_level": "{skill_level}",
  "roadmap": {{
    "beginner": ["Step-by-step beginner learning items with examples and exercises"],
    "intermediate": ["Step-by-step intermediate learning items with projects"],
    "advanced": ["Step-by-step advanced learning items with portfolio projects"]
  }},
  "timeline": {{
    "beginner": "X-Y months",
    "intermediate": "X-Y months", 
    "advanced": "X+ months"
  }},
  "learning_resources": {{
    "beginner": ["Specific beginner resources"],
    "intermediate": ["Intermediate resources"],
    "advanced": ["Advanced resources"]
  }},
  "project_ideas": {{
    "beginner": ["Beginner project ideas"],
    "intermediate": ["Intermediate project ideas"],
    "advanced": ["Advanced project ideas"]
  }},
  "skill_validation": {{
    "beginner": ["Quizzes", "Hands-on tasks"],
    "intermediate": ["Portfolio projects", "Peer reviews"],
    "advanced": ["Open-source contributions", "Certifications"]
  }},
  "recommended_skills": ["List of most important skills to master"]
}}
"""
        try:
            print(f"🧠 Generating detailed roadmap for {tech_field} with Groq...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON generator for detailed learning roadmaps. You will only output a valid JSON object with the specified structure."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-70b-8192",
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            
            response_content = chat_completion.choices[0].message.content
            parsed_json = json.loads(response_content)

            # Validate the response structure
            required_fields = ['field', 'current_level', 'roadmap', 'timeline', 'learning_resources', 'project_ideas']
            if not all(field in parsed_json for field in required_fields):
                print("🚨 Groq response missing required fields. Using fallback.")
                return RoadmapGenerator.get_fallback_roadmap(tech_field, skill_level)

            print(f"✅ Successfully generated detailed roadmap from Groq for {tech_field}.")
            return parsed_json

        except Exception as e:
            print(f"🚨 Groq API call failed: {e}. Using fallback roadmap.")
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
    return render_template('roadmap.html')

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
You are an expert technical interviewer creating a skills assessment. Based on the following user interests, generate {num_questions} multiple-choice questions (MCQs).

**User Interests:** {interests_str}

**Instructions:**
1. Generate exactly {num_questions} unique MCQs that are relevant to the provided interests.
2. The questions should cover a range of difficulties from easy to hard, suitable for a screening test.
3. Each question must have exactly 4 options.
4. For about 30-50% of the questions, include a relevant code snippet within the 'question' field to test practical application. The code snippet should be properly escaped for JSON, using `\\n` for newlines.
5. Indicate the correct answer using a zero-based index (0, 1, 2, or 3).
6. The output **must** be a valid JSON list of objects, and nothing else. Do not include any introductory text, explanations, or markdown formatting like ```json.

**JSON Format Example:**
[
  {{
    "question": "What is a common use case for WebSockets in Web Development?",
    "options": ["Storing user data", "Real-time bi-directional communication", "Styling web pages", "Querying a database"],
    "correct": 1
  }},
  {{
    "question": "What will be the output of the following Python code?\\n\\n```python\\nmy_list = [1, 2, 3]\\nprint(my_list[3])\\n```",
    "options": ["3", "None", "IndexError", "SyntaxError"],
    "correct": 2
  }}
]
"""
        try:
            print(f"🧠 Generating questions for interests ({interests_str}) with Groq...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a JSON generator for technical skill assessment questions. You will only output a valid JSON list of objects."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.6,
                response_format={"type": "json_object"},
            )
            
            response_content = chat_completion.choices[0].message.content
            parsed_json = json.loads(response_content)
            questions = parsed_json.get('questions') if isinstance(parsed_json, dict) else parsed_json

            if isinstance(questions, list) and len(questions) >= num_questions and all('question' in q and 'options' in q and 'correct' in q for q in questions):
                print(f"✅ Successfully generated {len(questions)} questions from Groq for interests.")
                session['assessment_data'] = {
                    'skills': interests, 'projects': [], 'achievements': [], 'internships': [],
                    'questions': questions, 'test_duration': 1800
                }
                return jsonify({'redirect': '/assessment'})
            else:
                print("🚨 Groq response for interests is malformed or incomplete. Using fallback.")
                if isinstance(questions, list):
                    print(f"   (Received {len(questions)} questions, expected {num_questions})")
        except Exception as e:
            print(f"🚨 Groq API call for interests failed: {e}. Using fallback questions.")

    # Fallback to static questions if Groq fails or is not available
    print("⚠️ Using fallback question generator for interests.")
    interest_to_skill_map = {
        'Frontend Development': ['javascript', 'react'],
        'Backend Development': ['python', 'sql', 'java'],
        'Full-Stack Development': ['javascript', 'react', 'python', 'sql'],
        'Machine Learning': ['machine learning', 'python'],
        'Deep Learning': ['machine learning', 'python'],
        'NLP': ['machine learning', 'python'],
        'Ethical Hacking': ['cybersecurity'],
        'Network Security': ['cybersecurity'],
        'Cloud Security': ['cybersecurity', 'aws'],
        'LLM Development': ['gen ai', 'python'],
        'Prompt Engineering': ['gen ai'],
        'RAG Systems': ['gen ai', 'python'],
        'Data Visualization': ['business analytics', 'python'],
        'Business Intelligence': ['business analytics', 'sql'],
        'Data Engineering': ['sql', 'python', 'aws'],
        'Autonomous Agents': ['agentic ai', 'python'],
        'Multi-Agent Systems': ['agentic ai', 'python'],
        'DevOps': ['docker', 'aws'],
        'Cloud Computing': ['aws', 'docker']
    }
    skills_to_query = list(set(skill for interest in interests for skill in interest_to_skill_map.get(interest, [])))

    questions = QuestionGenerator.generate_fallback_questions(
        {'skills': skills_to_query, 'projects': [], 'achievements': []}, num_questions=num_questions
    )

    if not questions:
        return jsonify({'error': 'We couldn\'t generate an assessment for your selected interests at this time.'}), 400

    session['assessment_data'] = {
        'skills': interests, 'projects': [], 'achievements': [], 'internships': [],
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
            model="llama3-8b-8192",
            temperature=0.7,
        )
        response = chat_completion.choices[0].message.content
        return jsonify({'response': response})
    except Exception as e:
        print(f"🚨 AI Mentor (Groq) API call failed: {e}")
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
        allowed_extensions = {'.pdf', '.doc', '.docx'}
        file_ext = '.' + filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Unsupported file format. Please upload PDF, DOC, or DOCX files.'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
        text = ''
        if file_ext == '.pdf':
            text = FileProcessor.extract_text_from_pdf(file_path)
        elif file_ext in ['.doc', '.docx']:
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
        session['assessment_data'] = {
            'skills': extracted_data['skills'],
            'projects': extracted_data['projects'], 
            'achievements': extracted_data['achievements'],
            'internships': extracted_data.get('internships', []),
            'education': extracted_data.get('education', []),
            'domain_expertise': extracted_data.get('domain', []),
            'questions': questions,
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
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
def generate_roadmap():
    """Generate personalized career roadmap."""
    data = request.json
    tech_field = data.get('tech_field')
    results = session.get('assessment_results', {})
    skill_level = results.get('level', 'Beginner')
    skills = results.get('skills', [])
    
    print(f"🔧 Roadmap generation - tech_field: {tech_field}")
    print(f"🔧 Roadmap generation - results: {results}")
    print(f"🔧 Roadmap generation - skill_level: {skill_level}")
    
    if not tech_field:
        print("❌ No tech_field provided")
        return jsonify({'error': 'No tech field selected'}), 400
    
    if not results:
        print("❌ No assessment results found")
        return jsonify({'error': 'No assessment results found. Please complete assessment first.'}), 400
    
    session['tech_field'] = tech_field
    
    # AI Processing: Create personalized roadmap with learning resources, project ideas, timelines
    roadmap = RoadmapGenerator.generate_roadmap_with_groq(tech_field, skill_level, skills)
    session['roadmap_data'] = roadmap
    
    print(f"✅ Roadmap generated successfully: {roadmap.get('field', 'No field')}")
    print(f"🔍 Roadmap data structure: {list(roadmap.keys())}")
    print(f"🔍 Roadmap field value: {roadmap.get('field')}")
    print(f"🔍 Session roadmap_data: {session.get('roadmap_data', {}).get('field')}")
    
    return jsonify({'redirect': '/roadmap'})

@app.route('/api/payment/create', methods=['POST'])
def create_payment_link():
    """Simulate generating a payment link and redirecting to a payment page."""
    # In a real app, you'd call a payment provider (Stripe, Razorpay) here.
    token = str(uuid.uuid4())
    session['payment_token'] = token
    payment_url = url_for('payment_page', token=token, _external=True)
    return jsonify({'redirect_url': payment_url})

@app.route('/pay')
def payment_page():
    """Simple payment page that confirms payment (stub)."""
    token = request.args.get('token', '')
    # Only allow if token matches what we created
    if token and session.get('payment_token') == token:
        return render_template('payment.html', token=token)
    return redirect(url_for('index'))

@app.route('/payment/confirm', methods=['POST'])
def confirm_payment():
    """Confirm payment and mark roadmap as unlocked. Simulates gateway callback."""
    token = request.json.get('token') if request.is_json else request.form.get('token')
    # In a real app, this would be a webhook from the payment provider.
    # For simulation, we check the session token.
    if token and session.get('payment_token') == token and not session.get('payment_confirmed'):
        print(f"✅ Payment confirmed for token: {token}")
        session['payment_confirmed'] = True
        session['roadmap_unlocked'] = True
        return jsonify({'redirect': url_for('roadmap_page')})
    return jsonify({'error': 'Invalid payment token'}), 400

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

@app.route('/api/download_roadmap', methods=['POST'])
def download_roadmap():
    """Generate and download roadmap file."""
    data = session.get('roadmap_data', {})
    
    roadmap_text = f"""
PERSONALIZED CAREER ROADMAP
===========================

Tech Field: {data.get('field', 'N/A')}
Current Level: {data.get('current_level', 'N/A')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

LEARNING PATH:
--------------
Beginner Level ({data.get('timeline', {}).get('beginner', 'N/A')}):
{chr(10).join('- ' + item for item in data.get('roadmap', {}).get('beginner', []))}

Intermediate Level ({data.get('timeline', {}).get('intermediate', 'N/A')}):
{chr(10).join('- ' + item for item in data.get('roadmap', {}).get('intermediate', []))}

Advanced Level ({data.get('timeline', {}).get('advanced', 'N/A')}):
{chr(10).join('- ' + item for item in data.get('roadmap', {}).get('advanced', []))}

RECOMMENDED SKILLS:
------------------
{chr(10).join('- ' + skill for skill in data.get('recommended_skills', []))}
"""
    
    filename = f"roadmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    with open(filepath, 'w') as f:
        f.write(roadmap_text)
    
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/api/get_assessment_data')
def get_assessment_data():
    """Get assessment data for current session."""
    return jsonify(session.get('assessment_data', {}))

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

    print(f"🔍 get_roadmap_data - Session keys: {list(session.keys())}")
    print(f"🔍 get_roadmap_data - Roadmap data: {data}")
    print(f"🔍 get_roadmap_data - Wrapped data: {wrapped}")
    
    return jsonify(wrapped)

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """Clear session data for new assessment."""
    session.clear()
    return jsonify({'success': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print('\n🚀 Starting Skills Assessment Platform...')
    print(f'📍 Server starting on port: {port}')
    if debug:
        print('📍 Development mode enabled')
    print('\n💡 Make sure to:')
    print('   • Check your internet connection')
    print('   • Set GROQ_API_KEY environment variable')
    print('   • Upload a technical resume for best results\n')
    
    app.run(debug=debug, host='0.0.0.0', port=port)
