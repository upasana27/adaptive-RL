#!/bin/bash
# Start the Overcooked User Study with ngrok tunnel

set -e

cd /home/asurite.ad.asu.edu/ubiswas2/adaptive-RL

echo "Starting Overcooked User Study..."
echo ""

# Check if processes are already running
FLASK_PID=$(lsof -i :5000 2>/dev/null | grep python | awk '{print $2; exit}' || echo "")
NGROK_PID=$(ps aux | grep 'ngrok http 5000' | grep -v grep | awk '{print $2; exit}' || echo "")

if [ ! -z "$FLASK_PID" ]; then
    echo "⚠ Flask server already running (PID: $FLASK_PID)"
    echo "   Use: kill $FLASK_PID"
    echo ""
fi

if [ ! -z "$NGROK_PID" ]; then
    echo "⚠ ngrok already running (PID: $NGROK_PID)"
    echo "   Use: kill $NGROK_PID"
    echo ""
fi

if [ -z "$FLASK_PID" ]; then
    echo "Starting Flask server..."
    nohup bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate pace && python webapp/test_server.py' > /tmp/flask_server.log 2>&1 &
    sleep 3
fi

if [ -z "$NGROK_PID" ]; then
    echo "Starting ngrok tunnel..."
    nohup ngrok http 5000 > /tmp/ngrok.log 2>&1 &
    sleep 3
fi

echo ""
echo "✓ Services Started!"
echo ""
echo "=== ACCESS INFORMATION ==="
echo ""
echo "Local URL:   http://localhost:5000"
echo ""

# Get ngrok public URL
PUBLIC_URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"[^"]*"' | sed 's/"public_url":"//' | sed 's/"$//' | head -1)
if [ ! -z "$PUBLIC_URL" ]; then
    echo "Public URL:  $PUBLIC_URL"
    echo ""
    echo "Share this link with participants:"
    echo "  $PUBLIC_URL"
else
    echo "Public URL:  (waiting for ngrok to initialize...)"
fi

echo ""
echo "=== LOG FILES ==="
echo "Flask: tail -f /tmp/flask_server.log"
echo "ngrok: tail -f /tmp/ngrok.log"
echo ""
echo "=== STOP SERVICES ==="
echo "pkill -f 'python.*test_server'"
echo "pkill -f 'ngrok http 5000'"
