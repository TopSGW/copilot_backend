#!/bin/bash

# Change to the directory containing the main.py file
cd /workspace/copilot_backend

# Run the FastAPI app with nohup
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

echo $! > backend.pid

echo "Backend started with PID $(cat backend.pid)"
