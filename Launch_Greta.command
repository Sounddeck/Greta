#!/bin/bash
echo "🎭 Starting Greta PAI..."
cd /Users/macone/Desktop/Greta
source greta_env/bin/activate
export PYTHONPATH=/Users/macone/Desktop/Greta:$PYTHONPATH
cd backend
python main.py
