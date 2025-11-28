# Quick Start Guide 🚀

## For Someone New to Web Development

### What You Have Now

You have a **complete website** for running your user study! It's like having a video game website where participants can:
1. Sign up with a nickname
2. Play practice rounds
3. Play real game rounds with AI partners
4. Get a "thanks for participating" message at the end

All their gameplay data (button presses, scores, timing) gets saved to files automatically.

---

## Testing It Right Now (5 minutes)

### Step 1: Start the Server
```bash
cd /home/asurite.ad.asu.edu/ubiswas2/adaptive-RL/webapp
./start_server.sh
```

You should see:
```
============================================================
Overcooked User Study - Test Server
============================================================
Discovered 4 models:
  - ppo_pace_two_ing_2_seed1
  ...

Starting server on http://localhost:5000
```

**Leave this terminal open!** The server needs to keep running.

### Step 2: Open the Website
1. Open a web browser
2. Go to: `http://localhost:5000`
3. You'll see a page asking for your name

### Step 3: Test the Flow
1. Type a fake name like "TestUser"
2. Check the consent box
3. Click "Start Study"
4. You'll see a practice round with a black rectangle (that's the game canvas)
5. Click "Start Practice"
6. Press arrow keys on your keyboard
7. The rectangle will change colors (this simulates the game)
8. After about 50 steps, it says "Round complete!"
9. Click "Continue to Game"
10. Play through the 8 rounds
11. See the thank you page

### Step 4: Check the Data
Open a new terminal and run:
```bash
ls webapp/data/
```

You'll see files like:
- `abc123-xyz_interactions.csv` (every button press logged)
- `abc123-xyz_session.json` (summary of the session)

Look inside:
```bash
cat webapp/data/*_interactions.csv | head -20
```

You'll see rows with timestamps, which button was pressed, rewards, etc.

---

## What's a "MockEnv"?

Right now, the website uses a **fake game** (called MockEnv) that just shows colored rectangles. This is perfect for testing! You can make sure:
- The website loads ✓
- Buttons work ✓  
- Data gets saved ✓
- Pages flow correctly ✓

**Later**, you'll connect your real Overcooked game environment. But for now, MockEnv lets you test everything else.

---

## Architecture in Simple Terms

Think of your website like a restaurant:

### The Kitchen (Backend)
- **app.py** = Head chef (coordinates everything)
- **models.py** = Recipe book (knows about your 4 AI models)
- **env_wrapper.py** = Cooking station (runs the game, right now using MockEnv)
- **logger.py** = Notebook (writes down everything that happens)

### The Dining Room (Frontend)  
- **templates/** = The menu and dining room layout (HTML pages)
- **static/css/** = Restaurant decoration (colors, fonts, layout)
- **static/js/game.js** = Waiter (takes orders from customers, brings food back)

### The Flow
1. Customer (participant) walks in (opens website)
2. Waiter (game.js) takes their order (name, consent)
3. Head chef (app.py) prepares meal (creates game session)
4. Kitchen sends out food (game frames) continuously
5. Customer eats and gives feedback (presses keys)
6. Notebook (logger.py) writes down every bite (logs actions)

---

## Files You Created

```
webapp/
├── app.py              ← Main server (the "brain")
├── models.py           ← Finds your 4 trained AI models
├── env_wrapper.py      ← Runs the game (currently MockEnv)
├── logger.py           ← Saves all data to CSV files
├── test_server.py      ← Simple way to start the server
├── start_server.sh     ← Even simpler! (just ./start_server.sh)
├── requirements.txt    ← List of Python packages needed
├── README.md           ← Full documentation
├── STATUS.md           ← Technical summary (what works, what's next)
├── QUICKSTART.md       ← This file!
│
├── config/
│   └── study_config.yaml   ← Settings (which models, how many rounds)
│
├── templates/
│   ├── index.html      ← Landing page
│   ├── demo.html       ← Practice round page
│   ├── game.html       ← Main game page
│   └── thanks.html     ← Completion page
│
├── static/
│   ├── css/style.css   ← All the pretty colors and fonts
│   └── js/game.js      ← Handles keyboard input, talks to server
│
└── data/               ← Your participant data gets saved here
    ├── {uuid}_interactions.csv    ← Every action logged
    └── {uuid}_{sid}_session.json  ← Session metadata
```

---

## What Works Right Now

✅ **Full website flow** (landing → practice → games → thanks)  
✅ **4 AI models discovered** automatically from your logs/  
✅ **Keyboard controls** (arrow keys + space)  
✅ **Real-time game streaming** (10 frames per second)  
✅ **Data logging** (CSV + JSON files)  
✅ **Randomized rounds** (2 per model = 8 total)  
✅ **Privacy** (pseudonyms are hashed)  

---

## What Needs Work

⚠️ **Real Overcooked game** - Currently using colored rectangles (MockEnv)  
⚠️ **AI model loading** - AI currently does nothing (placeholder)  
⚠️ **Production deployment** - This is just running on your local machine  

---

## Next Steps

### If You Want to Test More
Just run `./start_server.sh` again and play through the study multiple times. Each time creates new data files with unique IDs.

### If You Want Real Overcooked
You'll need to:
1. Understand how `environment/overcooked` works
2. Update `env_wrapper.py` to use it instead of MockEnv
3. Make sure it renders frames as images

### If You Want AI Opponents
You'll need to:
1. Look at your model checkpoint files (the .pt files)
2. Figure out how to load them with PyTorch
3. Update `models.py` to actually call the AI

### If You Want to Deploy for Real Users
You'll need to:
1. Put this on a server (not your laptop)
2. Get a domain name (like study.yourlab.edu)
3. Set up nginx and SSL certificates
4. Create a systemd service so it starts automatically

**But for now, you have a working website you can test!**

---

## Troubleshooting

**"Port 5000 already in use"**
- Another program is using that port
- Run: `lsof -ti:5000 | xargs kill`
- Or change the port in test_server.py

**"ModuleNotFoundError: No module named 'webapp'"**
- Use the start_server.sh script (it sets paths correctly)
- Or run: `PYTHONPATH=$PWD python webapp/test_server.py`

**"Nothing happens when I press keys"**
- Check browser console (F12) for JavaScript errors
- Make sure you clicked "Start Practice" first
- Check that the server terminal shows "Client connected"

**"Frames aren't updating"**
- Look at the server terminal for errors
- Check if MockEnv is working: `python -c "from webapp.env_wrapper import MockEnv; env = MockEnv(); print('OK')"`

**"Data files aren't being created"**
- Check if webapp/data/ folder exists
- Check server terminal for errors when you press keys
- Make sure you completed at least one round

---

## Getting Help

1. **Check STATUS.md** - Technical details about what's implemented
2. **Check README.md** - Full documentation with examples
3. **Look at server terminal** - Errors will show up there
4. **Check browser console** (press F12) - JavaScript errors show up there

---

## Summary

**You built a complete user study website from scratch!** 🎉

It's like the reference repo (coop-eval-user-study) you showed me, but customized for:
- Your 4 specific AI models
- Your local Overcooked environment (not the Overcooked-AI repo)
- Your study design (2 rounds per model)
- Your keyboard controls

Right now it works with a "dummy game" (MockEnv) so you can test everything. Later you'll plug in your real game and AI models. But the website infrastructure is complete and working!

Try it out: `./start_server.sh` and go to http://localhost:5000 🚀
