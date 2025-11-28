# Deployment Guide for User Study Webapp

This guide explains how to make your Overcooked user study accessible to participants outside your local network.

## Option 1: ngrok (Easiest - Recommended for Quick Testing)

ngrok creates a secure tunnel to your local server, giving you a public URL instantly.

### Setup:
1. Install ngrok:
   ```bash
   # Download and install from https://ngrok.com/download
   # Or use snap:
   sudo snap install ngrok
   ```

2. Sign up for free account at https://ngrok.com and get your auth token

3. Configure ngrok:
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```

4. Start your Flask server:
   ```bash
   cd /home/asurite.ad.asu.edu/ubiswas2/adaptive-RL
   PYTHONPATH=$PWD python webapp/app.py
   ```

5. In a new terminal, start ngrok:
   ```bash
   ngrok http 5000
   ```

6. ngrok will display a public URL like: `https://xxxx-xx-xx-xx-xx.ngrok-free.app`
   - Share this URL with your participants!
   - The URL remains active as long as ngrok is running

### Pros:
- ✅ No server configuration needed
- ✅ HTTPS included automatically
- ✅ Works behind firewalls/NAT
- ✅ Can use custom domain (paid plans)

### Cons:
- ❌ URL changes each time you restart ngrok (free plan)
- ❌ Connection goes through ngrok servers
- ❌ Limited to 40 connections/minute (free plan)

---

## Option 2: Cloud Deployment (Best for Actual Study)

Deploy to a cloud provider for a stable, permanent URL.

### A. Deploy to Google Cloud Run (Serverless, Auto-scaling)

1. Install Google Cloud SDK:
   ```bash
   curl https://sdk.cloud.google.com | bash
   gcloud init
   ```

2. Create `webapp/Dockerfile`:
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   COPY . /app
   
   RUN pip install --no-cache-dir flask flask-socketio eventlet torch pyyaml numpy pettingzoo pygame stable-baselines3
   
   EXPOSE 8080
   ENV PORT=8080
   
   CMD ["python", "webapp/app.py"]
   ```

3. Deploy:
   ```bash
   gcloud run deploy overcooked-study \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2
   ```

### B. Deploy to Heroku (Simple Platform)

1. Install Heroku CLI and login:
   ```bash
   curl https://cli-assets.heroku.com/install.sh | sh
   heroku login
   ```

2. Create `Procfile`:
   ```
   web: gunicorn --worker-class eventlet -w 1 webapp.app:app
   ```

3. Create `requirements.txt`:
   ```bash
   pip freeze > requirements.txt
   ```

4. Deploy:
   ```bash
   git init
   heroku create your-overcooked-study
   git add .
   git commit -m "Initial deployment"
   git push heroku main
   ```

### C. Deploy to Your University Server

If you have access to a university server (common for research):

1. Copy files to server:
   ```bash
   scp -r webapp/ your-uni-server:~/overcooked-study/
   ```

2. SSH into server and set up:
   ```bash
   ssh your-uni-server
   cd overcooked-study
   pip install -r requirements.txt
   ```

3. Run with gunicorn:
   ```bash
   gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 webapp.app:app
   ```

4. Configure reverse proxy (nginx/Apache) and HTTPS

---

## Option 3: Port Forwarding (If You Have Admin Access)

If you have access to your router:

1. Forward port 5000 to your machine's local IP
2. Find your public IP: https://whatismyipaddress.com
3. Configure firewall to allow incoming connections on port 5000
4. Access via: `http://YOUR_PUBLIC_IP:5000`

**Security Warning**: Requires proper security setup (HTTPS, authentication, etc.)

---

## Recommended Setup for User Study

**For a small pilot study (< 50 participants):**
- Use **ngrok** - it's the fastest to set up

**For a full user study (50+ participants):**
- Use **Google Cloud Run** or **Heroku** for reliability and scalability
- Benefits:
  - Stable URL that doesn't change
  - Better performance and reliability
  - Automatic HTTPS
  - Usage analytics

---

## Quick Start with ngrok

```bash
# Terminal 1: Start Flask server
cd /home/asurite.ad.asu.edu/ubiswas2/adaptive-RL
PYTHONPATH=$PWD python webapp/app.py

# Terminal 2: Start ngrok
ngrok http 5000

# Share the ngrok URL with participants!
# Example: https://abc123.ngrok-free.app
```

---

## Configuration for Production

Before deploying, update `webapp/app.py`:

```python
# Change debug mode
if __name__ == '__main__':
    socketio.run(app, 
                 host='0.0.0.0', 
                 port=int(os.environ.get('PORT', 5000)),
                 debug=False)  # Set to False for production!
```

Also consider:
- Adding user authentication
- Setting up a database for participant data
- Implementing rate limiting
- Adding data backup/export tools

---

## Data Collection

All participant data is saved to `webapp/data/`:
- `participants.csv` - Participant info
- `sessions.csv` - Session summaries
- `actions.csv` - Detailed action logs
- `rounds_*.json` - Complete round data

Remember to:
1. Backup data regularly
2. Follow IRB protocols for data storage
3. Ensure participant privacy (anonymization)
