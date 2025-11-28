#!/bin/bash
# Quick launcher for the Overcooked User Study webapp

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/environment/overcooked:$(pwd)/environment/overcooked/gym_cooking/rebar:$PYTHONPATH"

echo "================================================"
echo "  Overcooked User Study Webapp"
echo "================================================"
echo ""
echo "Starting server..."
echo "Access at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================"
echo ""

python webapp/test_server.py
