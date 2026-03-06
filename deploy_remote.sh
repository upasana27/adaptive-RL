#!/bin/bash
# Deployment script for remote server
# This keeps the server running even after you disconnect

echo "========================================="
echo "Overcooked Study - Remote Deployment"
echo "========================================="
echo ""

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux not found. Installing..."
    if command -v apt &> /dev/null; then
        sudo apt install -y tmux
    elif command -v yum &> /dev/null; then
        sudo yum install -y tmux
    else
        echo "Please install tmux manually: sudo apt install tmux"
        exit 1
    fi
fi

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Please install it:"
    echo "   https://ngrok.com/download"
    echo "   Then configure: ngrok config add-authtoken YOUR_TOKEN"
    exit 1
fi

# Kill existing sessions if they exist
tmux kill-session -t overcooked_server 2>/dev/null
tmux kill-session -t overcooked_ngrok 2>/dev/null

echo "✓ Starting Flask server in tmux session..."
tmux new-session -d -s overcooked_server "cd webapp && bash start_server.sh"

echo "✓ Waiting for server to start..."
sleep 5

echo "✓ Starting ngrok in tmux session..."
tmux new-session -d -s overcooked_ngrok "ngrok http 5000"

echo "✓ Waiting for ngrok tunnel..."
sleep 3

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""

# Get ngrok URL
echo "🌐 Getting public URL..."
sleep 2
URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['tunnels'][0]['public_url'])" 2>/dev/null)

if [ -n "$URL" ]; then
    echo "✅ Public URL: $URL"
else
    echo "⚠️  URL not ready yet. Run: curl -s http://localhost:4040/api/tunnels | python3 -m json.tool"
fi

echo ""
echo "📋 Useful commands:"
echo "  View server logs:  tmux attach -t overcooked_server"
echo "  View ngrok:        tmux attach -t overcooked_ngrok"
echo "  List sessions:     tmux ls"
echo "  Detach from tmux:  Press Ctrl+B then D"
echo "  Kill server:       tmux kill-session -t overcooked_server"
echo "  Kill ngrok:        tmux kill-session -t overcooked_ngrok"
echo ""
echo "🎉 Server is now running in the background!"
echo "   You can safely disconnect. The server will keep running."
echo ""
