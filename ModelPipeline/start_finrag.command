
#!/bin/bash
# start_finrag.command
# FinRAG Startup Script for Mac/Linux
# Launches both backend (FastAPI) and frontend (Streamlit) servers

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_ENV="$SCRIPT_DIR/finrag_ml_tg1/venv_ml_rag/bin/activate"
FRONTEND_ENV="$SCRIPT_DIR/serving/frontend/venv_frontend/bin/activate"
SERVING_DIR="$SCRIPT_DIR/serving"

BACKEND_PORT=8000
FRONTEND_PORT=8501

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# ==============================================================================
# BANNER
# ==============================================================================

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${WHITE}           FinRAG - Financial Document Intelligence         ${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# ==============================================================================
# VALIDATION
# ==============================================================================

echo -e "${YELLOW}[*] Validating environment...${NC}"

# Check if backend environment exists
if [ ! -f "$BACKEND_ENV" ]; then
    echo -e "${RED}[ERROR] Backend virtual environment not found!${NC}"
    echo -e "${RED}   Expected: $BACKEND_ENV${NC}"
    echo ""
    echo -e "${YELLOW}   Please create the backend environment first:${NC}"
    echo -e "${YELLOW}   cd finrag_ml_tg1${NC}"
    echo -e "${YELLOW}   python -m venv venv_ml_rag${NC}"
    echo -e "${YELLOW}   source venv_ml_rag/bin/activate${NC}"
    echo -e "${YELLOW}   pip install -r environments/requirements.txt${NC}"
    echo ""
    exit 1
fi

# Check if frontend environment exists
if [ ! -f "$FRONTEND_ENV" ]; then
    echo -e "${RED}[ERROR] Frontend virtual environment not found!${NC}"
    echo -e "${RED}   Expected: $FRONTEND_ENV${NC}"
    echo ""
    echo -e "${YELLOW}   Please create the frontend environment first:${NC}"
    echo -e "${YELLOW}   cd serving/frontend${NC}"
    echo -e "${YELLOW}   python -m venv venv_frontend${NC}"
    echo -e "${YELLOW}   source venv_frontend/bin/activate${NC}"
    echo -e "${YELLOW}   pip install -r requirements.txt${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}[OK] Environments validated${NC}"
echo ""

# ==============================================================================
# PORT CHECKING
# ==============================================================================

echo -e "${YELLOW}[*] Checking if ports are available...${NC}"

# Check backend port (Mac/Linux)
if lsof -Pi :$BACKEND_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}[WARNING] Port $BACKEND_PORT is already in use!${NC}"
    echo -e "${YELLOW}   Backend may already be running or port is occupied.${NC}"
    echo ""
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}[CANCELLED] Startup cancelled${NC}"
        exit 1
    fi
fi

# Check frontend port
if lsof -Pi :$FRONTEND_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}[WARNING] Port $FRONTEND_PORT is already in use!${NC}"
    echo -e "${YELLOW}   Frontend may already be running or port is occupied.${NC}"
    echo ""
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}[CANCELLED] Startup cancelled${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}[OK] Ports available${NC}"
echo ""

# ==============================================================================
# START BACKEND SERVER
# ==============================================================================

echo -e "${CYAN}============================================================${NC}"
echo -e "${WHITE}[BACKEND] Starting Backend Server (FastAPI + ML Pipeline)${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "${YELLOW}   Port: $BACKEND_PORT${NC}"
echo -e "${YELLOW}   URL:  http://localhost:$BACKEND_PORT${NC}"
echo -e "${YELLOW}   Docs: http://localhost:$BACKEND_PORT/docs${NC}"
echo ""

# Start backend in new terminal (Mac)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - FIXED: Removed color codes from osascript
    osascript -e "tell application \"Terminal\" to do script \"cd '$SERVING_DIR' && source '$BACKEND_ENV' && echo 'Backend environment activated' && echo 'Starting uvicorn server...' && echo '' && uvicorn backend.api_service:app --reload --host 0.0.0.0 --port $BACKEND_PORT\""
else
    # Linux (using gnome-terminal, xterm, or konsole)
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$SERVING_DIR' && source '$BACKEND_ENV' && echo 'Backend environment activated' && echo 'Starting uvicorn server...' && echo '' && uvicorn backend.api_service:app --reload --host 0.0.0.0 --port $BACKEND_PORT; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$SERVING_DIR' && source '$BACKEND_ENV' && echo 'Backend environment activated' && echo 'Starting uvicorn server...' && echo '' && uvicorn backend.api_service:app --reload --host 0.0.0.0 --port $BACKEND_PORT; bash" &
    else
        echo -e "${RED}[ERROR] No suitable terminal emulator found${NC}"
        echo -e "${YELLOW}   Please install gnome-terminal or xterm${NC}"
        exit 1
    fi
fi

echo -e "${YELLOW}[*] Backend starting... (waiting 8 seconds)${NC}"
sleep 8

# ==============================================================================
# START FRONTEND SERVER
# ==============================================================================

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${WHITE}[FRONTEND] Starting Frontend Server (Streamlit UI)${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "${YELLOW}   Port: $FRONTEND_PORT${NC}"
echo -e "${YELLOW}   URL:  http://localhost:$FRONTEND_PORT${NC}"
echo ""

# Start frontend in new terminal
# NOTE: Removed --server.headless false to prevent double browser open
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - FIXED: Removed color codes from osascript
    osascript -e "tell application \"Terminal\" to do script \"cd '$SERVING_DIR' && source '$FRONTEND_ENV' && echo 'Frontend environment activated' && echo 'Starting Streamlit server...' && echo 'Browser will open automatically...' && echo '' && streamlit run frontend/app.py --server.port $FRONTEND_PORT --server.address localhost\""
else
    # Linux
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$SERVING_DIR' && source '$FRONTEND_ENV' && echo 'Frontend environment activated' && echo 'Starting Streamlit server...' && echo 'Browser will open automatically...' && echo '' && streamlit run frontend/app.py --server.port $FRONTEND_PORT --server.address localhost; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$SERVING_DIR' && source '$FRONTEND_ENV' && echo 'Frontend environment activated' && echo 'Starting Streamlit server...' && echo 'Browser will open automatically...' && echo '' && streamlit run frontend/app.py --server.port $FRONTEND_PORT --server.address localhost; bash" &
    fi
fi

echo -e "${YELLOW}[*] Frontend starting... (waiting 6 seconds)${NC}"
sleep 6

# ==============================================================================
# OPEN BROWSER (BACKUP)
# ==============================================================================

echo ""
echo -e "${CYAN}[*] Opening browser...${NC}"
sleep 2

# Open browser based on OS (backup in case Streamlit didn't auto-open)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "http://localhost:$FRONTEND_PORT" 2>/dev/null || true
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:$FRONTEND_PORT" 2>/dev/null || true
    elif command -v gnome-open &> /dev/null; then
        gnome-open "http://localhost:$FRONTEND_PORT" 2>/dev/null || true
    fi
fi

# ==============================================================================
# COMPLETION
# ==============================================================================

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${WHITE}[SUCCESS] FinRAG Successfully Started!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${YELLOW}Access Points:${NC}"
echo -e "${CYAN}   Backend API:  http://localhost:$BACKEND_PORT${NC}"
echo -e "${CYAN}   API Docs:     http://localhost:$BACKEND_PORT/docs${NC}"
echo -e "${CYAN}   Frontend UI:  http://localhost:$FRONTEND_PORT${NC}"
echo ""
echo -e "${YELLOW}Status:${NC}"
echo -e "${GREEN}   [OK] Backend server started in separate window${NC}"
echo -e "${GREEN}   [OK] Frontend server started in separate window${NC}"
echo -e "${GREEN}   [OK] Browser opened automatically${NC}"
echo ""
echo -e "${YELLOW}Tips:${NC}"
echo -e "${WHITE}   - Both servers are running in separate terminal windows${NC}"
echo -e "${WHITE}   - Close those windows to stop the servers${NC}"
echo -e "${WHITE}   - If browser didn't open, visit: http://localhost:$FRONTEND_PORT${NC}"
echo ""
echo -e "${GREEN}[READY] Ready to analyze SEC 10-K filings!${NC}"
echo ""