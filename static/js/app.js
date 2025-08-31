// Application State
const AppState = {
    currentQuestions: [],
    userAnswers: [],
    assessmentResults: {},
    currentRoadmap: {},
    currentSkills: [],
    testTimer: null,
    timeRemaining: 600,
    tabSwitchCount: 0,
    testActive: false,
    timerStarted: false,
    selectedField: null,
    pendingAssessmentData: null
};

// DOM Elements
const Elements = {
    uploadSection: () => document.getElementById('upload-section'),
    assessmentSection: () => document.getElementById('assessment-section'),
    resultsSection: () => document.getElementById('results-section'),
    roadmapSection: () => document.getElementById('roadmap-section'),
    timerContainer: () => document.getElementById('timer-container'),
    timerDisplay: () => document.getElementById('timer-display'),
    warningModal: () => document.getElementById('warning-modal'),
    progressBar: () => document.getElementById('progress-bar'),
    questionCounter: () => document.getElementById('question-counter'),
    skillsFound: () => document.getElementById('skills-found'),
    questionsContainer: () => document.getElementById('questions-container'),
    submitButton: () => document.getElementById('submit-assessment'),
    prevButton: () => document.getElementById('prev-question'),
    nextButton: () => document.getElementById('next-question'),
    backToAssessmentButton: () => document.getElementById('back-to-assessment-btn'),
    resultsContent: () => document.getElementById('results-content'),
    techField: () => document.getElementById('tech-field'),
    roadmapContent: () => document.getElementById('roadmap-content'),
    uploadStatus: () => document.getElementById('upload-status'),
    resumeFile: () => document.getElementById('resume-file')
};

// Utility Functions
const Utils = {
    showElement: (element) => element.classList.remove('hidden'),
    hideElement: (element) => element.classList.add('hidden'),
    formatTime: (seconds) => {
        const minutes = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    },
    makeRequest: async (url, options = {}) => {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Request error:', error);
            throw new Error(`Request failed: ${error.message}`);
        }
    }
};

function showRules(assessmentData) {
    // Store assessment data to be used when the user starts the test
    AppState.pendingAssessmentData = assessmentData;

    // Hide previous sections
    Utils.hideElement(document.getElementById('onboarding-section'));
    const uploadSection = Elements.uploadSection();
    Utils.hideElement(uploadSection);
    uploadSection.classList.remove('active');
    
    // Show rules section
    const rulesSection = document.getElementById('rules-section');
    Utils.showElement(rulesSection);
    rulesSection.classList.add('active');
}

function startAssessmentFromRules() {
    if (AppState.pendingAssessmentData) {
        // Hide rules section
        const rulesSection = document.getElementById('rules-section');
        Utils.hideElement(rulesSection);
        rulesSection.classList.remove('active');

        // Start the assessment
        Assessment.displayQuestions(AppState.pendingAssessmentData);
        AppState.pendingAssessmentData = null; // Clear it after use
    } else {
        console.error("No pending assessment data found.");
        alert("An error occurred. Please try starting the assessment again.");
        location.reload();
    }
}

// Resume Upload Module
const ResumeUpload = {
    async upload() {
        const file = Elements.resumeFile().files[0];
        
        if (!file) {
            alert('Please select a file');
            return;
        }

        const formData = new FormData();
        formData.append('resume', file);

        Elements.uploadStatus().innerHTML = '<p>Analyzing resume...</p>';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();

            if (data.error) {
                Elements.uploadStatus().innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            } else {
                showRules(data);
            }
        } catch (error) {
            console.error('Upload error:', error);
            if (error.message.includes('Network') || error.message.includes('fetch')) {
                Elements.uploadStatus().innerHTML = `
                    <div style="color: red; padding: 1rem; background: #fee; border-radius: 8px; margin-top: 1rem;">
                        <i class="fas fa-wifi"></i> <strong>Network Error:</strong><br>
                        • Check your internet connection<br>
                        • Ensure the server is running on port 5000<br>
                        • Try refreshing the page
                    </div>
                `;
            } else {
                Elements.uploadStatus().innerHTML = `<p style="color: red;">Upload failed: ${error.message}</p>`;
            }
        }
    }
};

// Assessment Module
const Assessment = {
    renderCurrentQuestion() {
        const q = AppState.currentQuestions[AppState.currentQuestionIndex];
        const index = AppState.currentQuestionIndex;
        const total = AppState.currentQuestions.length;

        // Update counter
        Elements.questionCounter().textContent = `Question ${index + 1} of ${total}`;

        // Render question HTML
        const questionHtml = `
            <div class="question" data-question="${index}">
                <div class="question-header">
                    <span class="question-number">${index + 1}</span>
                    <h4>${q.question.replace(/\n/g, '<br>')}</h4>
                </div>
                <div class="options-grid">
                    ${q.options.map((option, optIndex) => `
                        <div class="option" onclick="Assessment.selectOption(${index}, ${optIndex})">
                            <input type="radio" name="q${index}" value="${optIndex}" id="q${index}_${optIndex}" ${AppState.userAnswers[index] === optIndex ? 'checked' : ''}>
                            <label for="q${index}_${optIndex}">${option}</label>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        Elements.questionsContainer().innerHTML = questionHtml;

        // Restore selected state visually
        if (AppState.userAnswers[index] !== -1) {
            const selectedOptionDiv = Elements.questionsContainer().querySelectorAll('.option')[AppState.userAnswers[index]];
            if (selectedOptionDiv) {
                selectedOptionDiv.classList.add('selected');
            }
        }

        // Update button visibility
        Utils.hideElement(Elements.prevButton());
        Utils.hideElement(Elements.nextButton());
        Utils.hideElement(Elements.submitButton());
        Utils.hideElement(Elements.backToAssessmentButton());

        if (index > 0) {
            Utils.showElement(Elements.prevButton());
        }
        if (index < total - 1) {
            const nextBtn = Elements.nextButton();
            nextBtn.innerHTML = 'Next <i class="fas fa-arrow-right"></i>';
            nextBtn.onclick = () => Assessment.nextQuestion();
            Utils.showElement(nextBtn);
        } else { // Show review button on the last question
            const reviewBtn = Elements.nextButton();
            reviewBtn.innerHTML = 'Review Answers <i class="fas fa-tasks"></i>';
            reviewBtn.onclick = () => Assessment.showReview();
            Utils.showElement(reviewBtn);
        }
    },

    showReview() {
        AppState.isReviewing = true;
        Elements.questionCounter().textContent = 'Review Your Answers';

        // Update button visibility for review mode
        Utils.hideElement(Elements.prevButton());
        Utils.hideElement(Elements.nextButton());
        Utils.showElement(Elements.backToAssessmentButton());
        Utils.showElement(Elements.submitButton());

        const reviewHtml = AppState.currentQuestions.map((q, index) => {
            const userAnswerIndex = AppState.userAnswers[index];
            const answerText = userAnswerIndex !== -1 
                ? q.options[userAnswerIndex] 
                : '<span class="unanswered">Not Answered</span>';
            
            return `
                <div class="review-question" id="review-q-${index}">
                    <div class="review-question-content">
                        <h5>${index + 1}. ${q.question.replace(/\n/g, '<br>')}</h5>
                        <p class="selected-answer">${answerText}</p>
                    </div>
                    <div class="review-actions">
                        <button class="btn-secondary" onclick="Assessment.editQuestion(${index})">
                            <i class="fas fa-edit"></i> Edit
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        Elements.questionsContainer().innerHTML = reviewHtml;
    },

    editQuestion(index) {
        AppState.isReviewing = false;
        AppState.currentQuestionIndex = index;
        this.renderCurrentQuestion();
    },

    backToAssessment() {
        AppState.isReviewing = false;
        this.renderCurrentQuestion();
    },

    nextQuestion() {
        if (AppState.currentQuestionIndex < AppState.currentQuestions.length - 1) {
            AppState.currentQuestionIndex++;
            this.renderCurrentQuestion();
        }
    },

    prevQuestion() {
        if (AppState.currentQuestionIndex > 0) {
            AppState.currentQuestionIndex--;
            this.renderCurrentQuestion();
        }
    },

    displayQuestions(data) {
        // Update navigation
        document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
        document.getElementById('step-2').classList.add('active');
        
        // Update sections
        document.getElementById('upload-section').classList.remove('active');
        Utils.hideElement(Elements.uploadSection());
        Utils.showElement(Elements.assessmentSection());
        Elements.assessmentSection().classList.add('active');
        Utils.showElement(Elements.timerContainer());

        AppState.currentSkills = data.skills;
        AppState.currentQuestions = data.questions;
        AppState.userAnswers = new Array(data.questions.length).fill(-1);
        AppState.timeRemaining = data.test_duration || 600;
        AppState.currentQuestionIndex = 0; // Initialize index

        // Display skills, projects, and achievements
        const skillsBadges = data.skills.map(skill => 
            `<span class="skill-badge">${skill}</span>`
        ).join('');
        
        const projectsHtml = data.projects && data.projects.length > 0 ? 
            `<div class="resume-section">
                <h5><i class="fas fa-project-diagram"></i> Project Experience</h5>
                <ul>${data.projects.map(project => `<li>${project}</li>`).join('')}</ul>
            </div>` : '';
            
        const achievementsHtml = data.achievements && data.achievements.length > 0 ? 
            `<div class="resume-section">
                <h5><i class="fas fa-trophy"></i> Key Achievements</h5>
                <ul>${data.achievements.map(achievement => `<li>${achievement}</li>`).join('')}</ul>
            </div>` : '';
        
        const isResumeAnalysis = data.projects && data.projects.length > 0;
        const headerText = isResumeAnalysis ? 'Resume Analysis' : 'Interests Overview';
        const contextText = isResumeAnalysis 
            ? 'Questions will be based on your skills, projects, and achievements'
            : 'Questions will be based on your selected interests';

        Elements.skillsFound().innerHTML = `
            <h4><i class="fas fa-cogs"></i> ${headerText}</h4>
            <div class="skills-badges">${skillsBadges}</div>
            ${projectsHtml}
            ${achievementsHtml}
            <p class="context-note"><i class="fas fa-info-circle"></i> ${contextText}</p>
        `;
        
        // Initial render
        this.renderCurrentQuestion();
        this.updateProgress();
        
        TestSecurity.startTest();
    },

    selectOption(questionIndex, optionIndex) {
        // Visual feedback
        const question = document.querySelector(`[data-question="${questionIndex}"]`);
        question.querySelectorAll('.option').forEach(opt => opt.classList.remove('selected'));
        question.querySelectorAll('.option')[optionIndex].classList.add('selected');
        
        // Update answer
        document.getElementById(`q${questionIndex}_${optionIndex}`).checked = true;
        this.updateAnswer(questionIndex, optionIndex);
    },

    updateAnswer(questionIndex, answerIndex) {
        AppState.userAnswers[questionIndex] = answerIndex;
        this.updateProgress();
    },

    updateProgress() {
        const answered = AppState.userAnswers.filter(a => a !== -1).length;
        const total = AppState.currentQuestions.length;
        const progress = total > 0 ? (answered / total) * 100 : 0;
        Elements.progressBar().style.width = progress + '%';
        AppState.progressText().textContent = `${Math.round(progress)}%`;
    },

    async submit() {
        const unansweredIndex = AppState.userAnswers.indexOf(-1);
        if (unansweredIndex !== -1) {
            alert('Please answer all questions before submitting.');
            if (!AppState.isReviewing) {
                this.showReview();
            }
            // Scroll to the first unanswered question and highlight it
            setTimeout(() => {
                const unansweredEl = document.getElementById(`review-q-${unansweredIndex}`);
                if (unansweredEl) {
                    unansweredEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    unansweredEl.style.border = '2px solid var(--danger)';
                    unansweredEl.style.boxShadow = '0 0 10px var(--danger)';
                }
            }, 100);
            return;
        }

        TestSecurity.endTest();

        try {
            const response = await fetch('/assess', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    answers: AppState.userAnswers,
                    questions: AppState.currentQuestions,
                    skills: AppState.currentSkills
                })
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            AppState.assessmentResults = data;
            
            Results.display(data);
        } catch (error) {
            console.error('Assessment error:', error);
            alert('Error submitting assessment. Please check server connection.');
        }
    }
};

// Results Module
const Results = {
    display(data) {
        // Update navigation
        document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
        document.getElementById('step-3').classList.add('active');
        
        Elements.assessmentSection().classList.remove('active');
        Utils.hideElement(Elements.assessmentSection());
        Utils.showElement(Elements.resultsSection());
        Elements.resultsSection().classList.add('active');

        // Create animated results display
        const scoreColor = data.percentage >= 80 ? 'var(--secondary)' : 
                          data.percentage >= 60 ? 'var(--warning)' : 'var(--danger)';
        
        const resultsHtml = `
            <div class="results-card">
                <div class="score-circle" style="--score-color: ${scoreColor}">
                    <div class="score-number">${data.percentage.toFixed(0)}%</div>
                    <div class="score-label">Overall Score</div>
                </div>
                <div class="results-details">
                    <div class="result-item">
                        <i class="fas fa-check-circle"></i>
                        <span>Correct: ${data.score}/${data.total}</span>
                    </div>
                    <div class="result-item">
                        <i class="fas fa-medal"></i>
                        <span>Level: ${data.level}</span>
                    </div>
                </div>
            </div>
        `;
        Elements.resultsContent().innerHTML = resultsHtml;

        // Create field selection grid
        const fieldIcons = {
            'AI/ML': 'fas fa-brain',
            'Business Analytics and Data Analytics': 'fas fa-chart-pie',
            'Gen AI': 'fas fa-robot',
            'Agentic AI': 'fas fa-project-diagram'
        };
        
        const fieldGrid = document.getElementById('field-grid');
        fieldGrid.innerHTML = data.tech_fields.map(field => `
            <div class="field-card" onclick="Results.selectField('${field}')" data-field="${field}">
                <i class="${fieldIcons[field] || 'fas fa-laptop-code'}"></i>
                <h4>${field}</h4>
            </div>
        `).join('');
    },
    
    selectField(field) {
        document.querySelectorAll('.field-card').forEach(card => card.classList.remove('selected'));
        document.querySelector(`[data-field="${field}"]`).classList.add('selected');
        AppState.selectedField = field;
    }
};

// Roadmap Module
const Roadmap = {
    async generate() {
        const techField = AppState.selectedField;
        if (!techField) {
            alert('Please select a tech field');
            return;
        }
        
        const loadingOverlay = document.getElementById('loading-overlay');
        Utils.showElement(loadingOverlay);

        try {
            const response = await fetch('/roadmap', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tech_field: techField,
                    skill_level: AppState.assessmentResults.level,
                    skills: AppState.currentSkills
                })
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            AppState.currentRoadmap = data;
            
            this.display(data);
        } catch (error) {
            console.error('Roadmap error:', error);
            alert('Error generating roadmap. Please check server connection.');
        } finally {
            Utils.hideElement(loadingOverlay);
        }
    },

    display(data) {
        // Update navigation
        document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
        document.getElementById('step-4').classList.add('active');
        
        Elements.resultsSection().classList.remove('active');
        Utils.hideElement(Elements.resultsSection());
        Utils.showElement(Elements.roadmapSection());
        Elements.roadmapSection().classList.add('active');
        Utils.showElement(document.getElementById('mentor-section'));

        const roadmapContentEl = document.getElementById('roadmap-content');
        const lockOverlay = document.getElementById('roadmap-lock-overlay');

        const roadmapHtml = `
            <div class="roadmap-overview">
                <div class="field-info">
                    <h3><i class="fas fa-target"></i> ${data.field}</h3>
                    <p>Current Level: <span class="level-badge">${data.current_level}</span></p>
                </div>
            </div>
            
            <div class="roadmap-timeline">
                <div class="roadmap-level beginner">
                    <div class="level-header">
                        <i class="fas fa-play-circle"></i>
                        <h4>Beginner Level</h4>
                        <span class="timeline">${data.timeline.beginner}</span>
                    </div>
                    <div class="roadmap-content">${data.roadmap.beginner.map(item => 
                        item.startsWith('•') ? `<div class="subtopic">${item}</div>` : 
                        `<div class="topic">${item}</div>`
                    ).join('')}</div>
                </div>
                
                <div class="roadmap-level intermediate">
                    <div class="level-header">
                        <i class="fas fa-rocket"></i>
                        <h4>Intermediate Level</h4>
                        <span class="timeline">${data.timeline.intermediate}</span>
                    </div>
                    <div class="roadmap-content">${data.roadmap.intermediate.map(item => 
                        item.startsWith('•') ? `<div class="subtopic">${item}</div>` : 
                        `<div class="topic">${item}</div>`
                    ).join('')}</div>
                </div>
                
                <div class="roadmap-level advanced">
                    <div class="level-header">
                        <i class="fas fa-crown"></i>
                        <h4>Advanced Level</h4>
                        <span class="timeline">${data.timeline.advanced}</span>
                    </div>
                    <div class="roadmap-content">${data.roadmap.advanced.map(item => 
                        item.startsWith('•') ? `<div class="subtopic">${item}</div>` : 
                        `<div class="topic">${item}</div>`
                    ).join('')}</div>
                </div>
            </div>
            
            <div class="skills-recommendation">
                <h4><i class="fas fa-lightbulb"></i> Recommended Skills</h4>
                <div class="skills-tags">
                    ${data.recommended_skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
                </div>
            </div>
        `;

        roadmapContentEl.innerHTML = roadmapHtml;

        // Add blur and show lock overlay for non-subscribed users
        roadmapContentEl.classList.add('blurred');
        Utils.showElement(lockOverlay);
        document.getElementById('download-roadmap-btn').disabled = true;
    },

    async download() {
        try {
            const response = await fetch('/download_roadmap', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(AppState.currentRoadmap)
            });

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `roadmap_${new Date().getTime()}.txt`;
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            alert('Error downloading roadmap: ' + error.message);
        }
    }
};

// Timer Module
const Timer = {
    start() {
        AppState.timerStarted = true;
        AppState.testTimer = setInterval(() => {
            AppState.timeRemaining--;
            this.updateDisplay();
            
            if (AppState.timeRemaining <= 60) {
                Elements.timerContainer().classList.add('timer-warning');
            }
            
            if (AppState.timeRemaining <= 0) {
                TestSecurity.terminateTest('Time expired!');
            }
        }, 1000);
    },

    updateDisplay() {
        Elements.timerDisplay().textContent = Utils.formatTime(AppState.timeRemaining);
    },

    stop() {
        AppState.timerStarted = false;
        clearInterval(AppState.testTimer);
        Utils.hideElement(Elements.timerContainer());
    }
};

// Test Security Module
const TestSecurity = {
    startTest() {
        AppState.testActive = true;
        Timer.start();
        this.setupTabSwitchDetection();
        this.disableCopyPaste();
    },

    endTest() {
        AppState.testActive = false;
        Timer.stop();
    },

    setupTabSwitchDetection() {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && AppState.testActive && AppState.timerStarted) {
                AppState.tabSwitchCount++;
                if (AppState.tabSwitchCount === 1) {
                    this.showWarning();
                } else if (AppState.tabSwitchCount >= 2) {
                    this.terminateTest('Test terminated due to multiple tab switches.');
                }
            }
        });
    },

    showWarning() {
        Utils.showElement(Elements.warningModal());
    },

    closeWarning() {
        Utils.hideElement(Elements.warningModal());
    },

    terminateTest(reason) {
        AppState.testActive = false;
        Timer.stop();
        alert(reason + ' Your test has been terminated.');
        location.reload();
    },

    disableCopyPaste() {
        document.addEventListener('keydown', (e) => {
            if (AppState.testActive && (e.ctrlKey || e.metaKey) && 
                ['c', 'v', 'a', 's', 'p'].includes(e.key)) {
                e.preventDefault();
                return false;
            }
            if (e.key === 'F12' || (e.ctrlKey && e.shiftKey && e.key === 'I')) {
                e.preventDefault();
                return false;
            }
        });
    }
};

async function startQuickAssessment() {
    // Show loading state
    const chatMessages = document.getElementById('chatMessages');
    const botMessage = document.createElement('div');
    botMessage.className = 'message bot-message';
    botMessage.innerHTML = `
        <div class="message-content">
            <span class="bot-avatar">🤖</span>
            <p>Great! I'm preparing a short assessment based on your interests...</p>
        </div>
    `;
    chatMessages.appendChild(botMessage);
    document.getElementById('chatOptions').innerHTML = '';
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch('/api/assessment/interest-based', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const data = await response.json();

        // Hide onboarding and show assessment
        setTimeout(() => {
            showRules(data);
        }, 1500);

    } catch (error) {
        console.error('Quick assessment error:', error);
        alert(`Could not create assessment: ${error.message}`);
    }
}

// Onboarding Functions
function handleOnboardingResponse(step, response) {
    const chatMessages = document.getElementById('chatMessages');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.innerHTML = `
        <div class="message-content">
            <p>${response}</p>
        </div>
    `;
    chatMessages.appendChild(userMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Handle special actions from onboarding
    if (response === 'Let me take a quick assessment first') {
        startQuickAssessment();
        return; // Exit here, don't proceed with standard onboarding fetch
    }
    
    fetch('/api/onboarding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step: step + 1, response: response })
    })
    .then(response => response.json())
    .then(data => {
        if (data.complete) {
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot-message';
            botMessage.innerHTML = `
                <div class="message-content">
                    <span class="bot-avatar">🤖</span>
                    <p>${data.message}</p>
                </div>
            `;
            chatMessages.appendChild(botMessage);
            
            setTimeout(() => {
                document.getElementById('upload-section').classList.remove('hidden');
                document.getElementById('onboarding-section').classList.add('hidden');
                Elements.uploadSection().classList.add('active');
            }, 2000);
        } else {
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot-message';
            botMessage.innerHTML = `
                <div class="message-content">
                    <span class="bot-avatar">🤖</span>
                    <p>${data.question}</p>
                </div>
            `;
            chatMessages.appendChild(botMessage);
            
            const chatOptions = document.getElementById('chatOptions');
            chatOptions.innerHTML = '';

            if (data.type === 'multi') {
                // Render chips for multi-select
                const chipContainer = document.createElement('div');
                chipContainer.className = 'chip-container';
                data.options.forEach((option) => {
                    const chip = document.createElement('button');
                    chip.className = 'chip';
                    chip.textContent = option;
                    chip.dataset.value = option;
                    chip.onclick = () => {
                        chip.classList.toggle('active');
                    };
                    chipContainer.appendChild(chip);
                });
                chatOptions.appendChild(chipContainer);

                const submitButton = document.createElement('button');
                submitButton.className = 'option-btn';
                submitButton.textContent = 'Continue';
                submitButton.style.marginTop = '1rem'; // Add some space
                submitButton.onclick = () => {
                    const selectedOptions = Array.from(document.querySelectorAll('.chip.active'))
                                                 .map(chip => chip.dataset.value);
                    if (selectedOptions.length > 0) {
                        handleOnboardingResponse(step + 1, selectedOptions.join(', '));
                    } else {
                        alert('Please select at least one interest.');
                    }
                };
                chatOptions.appendChild(submitButton);
            } else {
                data.options.forEach((option, index) => {
                    const button = document.createElement('button');
                    button.className = 'option-btn';
                    button.textContent = option;
                    button.onclick = () => handleOnboardingResponse(step + 1, option);
                    chatOptions.appendChild(button);
                });
            }
        }
        
        chatMessages.scrollTop = chatMessages.scrollHeight;
    })
    .catch(error => console.error('Onboarding error:', error));
}

function askMentor() {
    const input = document.getElementById('mentorInput');
    const question = input.value.trim();
    
    if (!question) return;
    
    fetch('/api/mentor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        const responseDiv = document.getElementById('mentorResponse');
        responseDiv.innerHTML = `<p><strong>AI Mentor:</strong> ${data.response}</p>`;
        input.value = '';
    })
    .catch(error => {
        console.error('Mentor error:', error);
        document.getElementById('mentorResponse').innerHTML = '<p>Sorry, I\'m having trouble right now. Please try again later.</p>';
    });
}

function unlockRoadmap() {
    const roadmapContentEl = document.getElementById('roadmap-content');
    const lockOverlay = document.getElementById('roadmap-lock-overlay');
    const downloadBtn = document.getElementById('download-roadmap-btn');

    // In a real app, this would involve a payment flow.
    // For this demo, we'll just unlock it.
    roadmapContentEl.classList.remove('blurred');
    Utils.hideElement(lockOverlay);
    downloadBtn.disabled = false;

    alert('Roadmap unlocked! You can now view and download it.');
}

// Enhanced Upload with Drag & Drop
const setupUpload = () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('resume-file');
    
    if (uploadArea && fileInput) {
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--primary-dark)';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = 'var(--primary)';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--primary)';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                ResumeUpload.upload();
            }
        });
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    setupUpload();
    
    // Ensure all functions are globally accessible
    window.handleOnboardingResponse = handleOnboardingResponse;
    window.askMentor = askMentor;
    window.unlockRoadmap = unlockRoadmap;
    window.uploadResume = () => ResumeUpload.upload();
    window.startAssessmentFromRules = startAssessmentFromRules;
    window.submitAssessment = () => Assessment.submit();
    window.nextQuestion = () => Assessment.nextQuestion();
    window.prevQuestion = () => Assessment.prevQuestion();
    window.generateRoadmap = () => Roadmap.generate();
    window.downloadRoadmap = () => Roadmap.download();
    window.closeWarning = () => TestSecurity.closeWarning();
    window.shareRoadmap = () => alert('Share functionality coming soon!');
    window.backToAssessment = () => Assessment.backToAssessment();
    window.editQuestion = (index) => Assessment.editQuestion(index);
});
