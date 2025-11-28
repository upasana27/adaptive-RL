#!/usr/bin/env python
"""Simple test script to verify webapp works."""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from webapp.app import app, socketio, REGISTRY

if __name__ == '__main__':
    print("=" * 60)
    print("Overcooked User Study - Test Server")
    print("=" * 60)
    print(f"Discovered {len(REGISTRY.list_models())} models:")
    for m in REGISTRY.list_models():
        print(f"  - {m}")
    print()
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
