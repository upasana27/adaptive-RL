#!/bin/bash
# Quick start script for running the webapp with ngrok for public access

echo "========================================="
echo "Overcooked User Study - Public Access"
echo "========================================="
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok is not installed!"
    echo ""
    echo "Install ngrok:"
    echo "  1. Visit: https://ngrok.com/download"
    echo "  2. Or use snap: sudo snap install ngrok"
    echo "  3. Get auth token from: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "  4. Run: ngrok config add-authtoken YOUR_TOKEN"
    echo ""
    exit 1
fi

echo "✓ ngrok is installed"
echo ""

# Check if Flask server is already running
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "✓ Flask server is already running on port 5000"
else
    echo "Starting Flask server..."
    cd "$(dirname "$0")/.."
    PYTHONPATH=$PWD python webapp/app.py &
    SERVER_PID=$!
    echo "✓ Flask server started (PID: $SERVER_PID)"
    sleep 3
fi

echo ""
echo "Starting ngrok tunnel..."
echo "========================================="
echo ""

# Start ngrok
ngrok http 5000 --log=stdout

# Cleanup on exit
trap "kill $SERVER_PID 2>/dev/null" EXIT
