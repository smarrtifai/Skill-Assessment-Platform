# Deployment Guide for Render

## Prerequisites
1. **Groq API Key**: Get your API key from [Groq Console](https://console.groq.com/)
2. **GitHub Repository**: Push your code to GitHub
3. **Render Account**: Sign up at [render.com](https://render.com)

## Deployment Steps

### 1. Prepare Repository
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy on Render
1. **Connect Repository**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**:
   - **Name**: `skills-assessment-platform`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`

3. **Set Environment Variables**:
   - `GROQ_API_KEY`: Your Groq API key
   - `SECRET_KEY`: Generate a secure random string
   - `FLASK_ENV`: `production`

### 3. Environment Variables Setup
In Render Dashboard → Your Service → Environment:
```
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
SECRET_KEY=your_secure_random_secret_key_here
FLASK_ENV=production
```

### 4. Deploy
- Click "Create Web Service"
- Render will automatically build and deploy your application
- Your app will be available at: `https://your-service-name.onrender.com`

## Important Notes

### File Upload Limitations
- Render has a 100MB request size limit
- Current app limit: 16MB (suitable for resumes)

### Cold Starts
- Free tier services may experience cold starts
- First request after inactivity may take 30+ seconds

### Persistent Storage
- Uploaded files are temporarily stored in memory
- Files are automatically cleaned up after processing

## Troubleshooting

### Common Issues
1. **Build Failures**: Check requirements.txt for correct versions
2. **API Errors**: Verify GROQ_API_KEY is set correctly
3. **Memory Issues**: Ensure file sizes are within limits

### Logs
View logs in Render Dashboard → Your Service → Logs

## Production Optimizations
- Set `FLASK_ENV=production`
- Use Gunicorn for better performance (already included)
- Monitor resource usage in Render dashboard

## Security
- Never commit API keys to repository
- Use environment variables for all secrets
- Regularly rotate API keys