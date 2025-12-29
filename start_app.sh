#!/bin/bash

echo "============================================"
echo "   Car Damage Detection - Application Start"
echo "============================================"
echo ""

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "[INFO] Checking Python dependencies..."
pip install -r requirements.txt -q

# Start Backend in background
echo "[INFO] Starting Backend API server..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to initialize
sleep 3

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "[INFO] Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start Frontend
echo "[INFO] Starting Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "============================================"
echo "   Application Started Successfully!"
echo "============================================"
echo ""
echo "   Backend API: http://localhost:8000"
echo "   API Docs:    http://localhost:8000/docs"
echo "   Frontend:    http://localhost:3000"
echo ""
echo "   Press Ctrl+C to stop all services"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
