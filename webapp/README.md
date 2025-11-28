# Overcooked User Study Webapp

Flask-based user study application for human-AI cooperative gameplay in Overcooked.

## Quick Start (Local Testing)

### 1. Install Dependencies

```bash
cd webapp
pip install -r requirements.txt
```

### 2. Run Server

```bash
python app.py
```

Server will start on `http://localhost:5000`

### 3. Test Flow

1. Open browser to `http://localhost:5000`
2. Enter pseudonym + check consent
3. Complete practice round (demo)
4. Play 8 game rounds (2 per model)
5. View completion pages

## Public Access for User Studies

### Option 1: ngrok (Fastest Setup - Recommended)

Make your study accessible to anyone with a public URL:

```bash
# Quick start with provided script
./start_with_ngrok.sh
```

Or manually:
```bash
# Terminal 1: Start server
PYTHONPATH=$PWD python webapp/app.py

# Terminal 2: Start ngrok (install from https://ngrok.com/download)
ngrok http 5000
```

You'll get a public URL like `https://xxxx-xx-xx.ngrok-free.app` - share this with participants!

**See `DEPLOYMENT.md` for detailed deployment options including cloud hosting.**

## Architecture

### Components

- **app.py**: Main Flask+SocketIO server
  - Routes: `/`, `/start`, `/demo`, `/game`, `/thanks`
  - SocketIO: `join_session`, `action`, `frame` streaming
  - Session management with randomized round sequences

- **models.py**: Model registry
  - Auto-discovers checkpoints from `logs/Overcooked/*/ppo/latest.pt`
  - ModelPolicy class (stub for PyTorch loading)

- **env_wrapper.py**: Environment manager
  - Creates per-session game environments
  - MockEnv fallback for testing without full Overcooked env
  - Renders frames as base64 PNG

- **logger.py**: Data collection
  - CSV: Per-step interactions (action, reward, timestamps)
  - JSON: Session metadata (participant, rounds, duration)

- **game.js**: Frontend client
  - Socket.IO connection
  - Keyboard input (Arrow keys + Space)
  - Canvas rendering at 10Hz
  - Action throttling

### Data Flow

```
Browser (game.js)
  ↓ Socket.IO
Flask app.py
  ↓
EnvManager → MockEnv/Overcooked
  ↓
Logger → CSV + JSON
```

## Study Configuration

Edit `config/study_config.yaml`:

```yaml
models:
  - ppo_pace_seed1
  - ppo_pace_two_ing_1_seed1
  - ppo_pace_two_ing_2_seed1
  - ppo_pace_two_ing_3_seed1

rounds_per_model: 2  # Total: 8 rounds per participant
tick_rate_hz: 10

key_map:
  up: 1
  down: 2
  left: 3
  right: 4
  space: 5

admin_token: "your_secret_token"
```

## Testing with MockEnv

The webapp includes a **MockEnv** fallback that generates dummy frames when the real Overcooked environment isn't available:

- Colored rectangles simulate game state
- NOOP action = gray, other actions = random colors
- 50-step episodes with random rewards

This lets you test the full webapp flow (UI, Socket.IO, logging) without needing the Overcooked environment.

## Admin Endpoints

- `GET /admin/models`: List discovered models
- `POST /admin/models/refresh?token=YOUR_TOKEN`: Refresh registry

## Data Files

Created in `webapp/data/`:

- `{participant_id}_interactions.csv`: Per-step logs
- `{participant_id}_{session_id}_session.json`: Metadata

### CSV Columns

```
participant_id, alias_hash, session_id, session_ts, round_idx, 
model_id, level, step, action, reward, done, info_json, 
client_ts, server_ts
```

## Deployment (Production)

### Install System Dependencies

```bash
sudo apt update
sudo apt install python3-pip nginx
```

### Setup App

```bash
cd /opt
git clone <your_repo>
cd adaptive-RL/webapp
pip install -r requirements.txt
```

### Create Systemd Service

`/etc/systemd/system/overcooked-study.service`:

```ini
[Unit]
Description=Overcooked User Study
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/adaptive-RL
Environment="PYTHONPATH=/opt/adaptive-RL"
ExecStart=/usr/bin/python3 -m webapp.app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable overcooked-study
sudo systemctl start overcooked-study
```

### Nginx Reverse Proxy

`/etc/nginx/sites-available/study`:

```nginx
server {
    listen 80;
    server_name your.domain.edu;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable and reload:

```bash
sudo ln -s /etc/nginx/sites-available/study /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Next Steps (TODOs)

1. **Implement PyTorch model loading** in `models.py`:
   - Load `.pt` checkpoints
   - Implement `ModelPolicy.act(obs)` using trained policy
   - Handle observation preprocessing

2. **Connect real Overcooked environment** in `env_wrapper.py`:
   - Import from `environment.overcooked`
   - Pass config from `config/study_config.yaml`
   - Handle multi-agent observation space

3. **Improve round tracking**:
   - Properly increment `session['current_round_idx']` after each round
   - Pass round index to logger

4. **Add session persistence**:
   - Store Flask session data in database/Redis
   - Allow resume after disconnect

5. **Admin dashboard**:
   - View active participants
   - Download aggregated CSV data
   - Real-time monitoring

## Troubleshooting

**"ImportError: No module named environment.overcooked"**
- Expected - MockEnv fallback is active
- To use real env, ensure `environment/overcooked` is importable

**Frames not updating**
- Check browser console for Socket.IO errors
- Verify server logs: `python app.py` should show "Client connected"
- Test `/admin/models` endpoint

**CSV not created**
- Check `webapp/data/` directory exists and is writable
- Verify `LOGGER.log_step()` is called in `app.py`

**Port 5000 already in use**
- Change port in `app.py`: `socketio.run(app, port=5001)`
- Or kill existing process: `lsof -ti:5000 | xargs kill`

## Contact

For issues or questions about the user study setup, contact your lab admin.
