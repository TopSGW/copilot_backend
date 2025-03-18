#!/bin/bash

# Change to the directory containing the main.py file
cd /workspace/copilot_backend

# Run the FastAPI app with nohup
nohup hypercorn main:app --bind 0.0.0.0:8000 > hypercorn.log 2>&1 &
echo $! > hypercorn.pid

echo $! > backend.pid

echo "Backend started with PID $(cat backend.pid)"
