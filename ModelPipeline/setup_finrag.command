#!/bin/bash
# setup_finrag.command
# FinRAG Setup Script for Mac/Linux
# Automates environment creation and dependency installation using UV

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/finrag_ml_tg1"
FRONTEND_DIR="$SCRIPT_DIR/serving/frontend"
BACKEND_ENV="$BACKEND_DIR/venv_ml_rag"
FRONTEND_ENV="$FRONTEND_DIR/venv_frontend"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# ==============================================================================
# HELPER FUNCTION: FORCE DELETE LOCKED DIRECTORY
# ==============================================================================

remove_locked_directory() {
    local path=$1
    local name=$2
    
    echo -e "${YELLOW}[*] Attempting to remove $name environment...${NC}"
    
    # Try 1: Normal deletion
    if rm -rf "$path" 2>/dev/null; then
        echo -e "${GREEN}[OK] $name environment removed successfully${NC}"
        return 0
    else
        echo -e "${YELLOW}[WARNING] Normal deletion failed (files locked)${NC}"
    fi
    
    # Try 2: Kill Python processes and retry
    echo -e "${YELLOW}[*] Killing Python processes and retrying...${NC}"
    
    # Kill all Python processes
    pkill -9 python 2>/dev/null || true
    pkill -9 pythonw 2>/dev/null || true
    pkill -9 uvicorn 2>/dev/null || true
    sleep 3
    
    # Retry deletion
    if rm -rf "$path" 2>/dev/null; then
        echo -e "${GREEN}[OK] $name environment removed after killing processes${NC}"
        return 0
    else
        echo -e "${YELLOW}[WARNING] Still locked after killing processes${NC}"
    fi
    
    # Try 3: Rename and mark for deletion
    echo -e "${YELLOW}[*] Using rename workaround...${NC}"
    
    local temp_name="${path}.old_$(date +%Y%m%d%H%M%S)"
    
    if mv "$path" "$temp_name" 2>/dev/null; then
        echo -e "${GREEN}[OK] Environment renamed to: $(basename $temp_name)${NC}"
        echo -e "${CYAN}   (Will be cleaned up manually later or on reboot)${NC}"
        return 0
    else
        echo -e "${RED}[ERROR] Could not remove or rename environment!${NC}"
        echo ""
        echo -e "${YELLOW}Manual fix required:${NC}"
        echo -e "${WHITE}   1. Close ALL terminal windows${NC}"
        echo -e "${WHITE}   2. Open Activity Monitor (Cmd+Space, type 'Activity Monitor')${NC}"
        echo -e "${WHITE}   3. Search for 'python' and quit all processes${NC}"
        echo -e "${WHITE}   4. Manually delete: $path${NC}"
        echo -e "${WHITE}   5. Run this script again${NC}"
        echo ""
        return 1
    fi
}

# ==============================================================================
# BANNER
# ==============================================================================

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${WHITE}           FinRAG Setup - Environment Configuration         ${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# ==============================================================================
# CHECK PYTHON
# ==============================================================================

echo -e "${YELLOW}[*] Checking Python installation...${NC}"

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        # BSD-compatible version parsing
        version=$($cmd --version 2>&1 | awk '{print $2}')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON_CMD=$cmd
            echo -e "${GREEN}[OK] Found Python: $($cmd --version)${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}[ERROR] Python 3.11+ not found!${NC}"
    echo -e "${YELLOW}   Please install Python 3.11 or higher from python.org${NC}"
    exit 1
fi

echo ""

# ==============================================================================
# CHECK/INSTALL UV
# ==============================================================================

echo -e "${YELLOW}[*] Checking for UV package installer...${NC}"

UV_INSTALLED=false
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>&1)
    echo -e "${GREEN}[OK] UV already installed: $UV_VERSION${NC}"
    UV_INSTALLED=true
else
    echo -e "${YELLOW}[INFO] UV not found, will install...${NC}"
fi

if [ "$UV_INSTALLED" = false ]; then
    echo ""
    echo -e "${CYAN}[*] Installing UV (fast package installer)...${NC}"
    echo -e "${WHITE}   This will make dependency installation 10-20x faster!${NC}"
    echo ""
    
    # Install UV using the official installer
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        # Add UV to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        if command -v uv &> /dev/null; then
            echo -e "${GREEN}[OK] UV installed successfully!${NC}"
            UV_VERSION=$(uv --version 2>&1)
            echo -e "${CYAN}   Version: $UV_VERSION${NC}"
            UV_INSTALLED=true
        else
            echo -e "${YELLOW}[WARNING] Failed to install UV automatically${NC}"
            echo -e "${YELLOW}   Falling back to pip (will be slower)${NC}"
            echo ""
            UV_INSTALLED=false
        fi
    else
        echo -e "${YELLOW}[WARNING] Failed to install UV automatically${NC}"
        echo -e "${YELLOW}   Falling back to pip (will be slower)${NC}"
        echo ""
        UV_INSTALLED=false
    fi
fi

echo ""

# ==============================================================================
# SETUP BACKEND ENVIRONMENT
# ==============================================================================

echo -e "${CYAN}============================================================${NC}"
echo -e "${WHITE}[BACKEND] Setting up ML environment${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "${YELLOW}   Location: $BACKEND_ENV${NC}"
echo ""

SKIP_BACKEND=false

# Check if environment already exists
if [ -d "$BACKEND_ENV" ]; then
    echo -e "${YELLOW}[WARNING] Backend environment already exists!${NC}"
    read -p "   Delete and recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Use the robust deletion function
        if ! remove_locked_directory "$BACKEND_ENV" "Backend"; then
            echo -e "${RED}[ERROR] Failed to remove backend environment${NC}"
            echo -e "${CYAN}[SKIP] Cannot proceed with backend setup${NC}"
            SKIP_BACKEND=true
        fi
    else
        echo -e "${CYAN}[SKIP] Keeping existing backend environment${NC}"
        SKIP_BACKEND=true
    fi
fi

if [ "$SKIP_BACKEND" = false ]; then
    # Create virtual environment
    echo ""
    echo -e "${YELLOW}[*] Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv "$BACKEND_ENV"
    
    if [ ! -d "$BACKEND_ENV" ]; then
        echo -e "${RED}[ERROR] Failed to create backend environment!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}[OK] Virtual environment created${NC}"
    echo ""
    
    # Activate and install dependencies
    source "$BACKEND_ENV/bin/activate"
    
    echo -e "${YELLOW}[*] Installing dependencies (this may take 1-2 minutes with UV)...${NC}"
    echo -e "${CYAN}   Requirements: finrag_ml_tg1/environments/requirements.txt${NC}"
    echo ""
    
    REQUIREMENTS_FILE="$BACKEND_DIR/environments/requirements.txt"
    
    if [ "$UV_INSTALLED" = true ]; then
        # Use UV for fast installation
        echo -e "${CYAN}[UV] Using fast package installer...${NC}"
        uv pip install -r "$REQUIREMENTS_FILE"
    else
        # Fallback to pip
        echo -e "${YELLOW}[PIP] Using standard pip (slower)...${NC}"
        python -m pip install --upgrade pip
        pip install -r "$REQUIREMENTS_FILE"
    fi
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}[OK] Backend dependencies installed successfully!${NC}"
    else
        echo ""
        echo -e "${RED}[ERROR] Failed to install backend dependencies!${NC}"
        exit 1
    fi
    
    deactivate
fi

echo ""

# ==============================================================================
# SETUP SERVING ENVIRONMENT (MINIMAL - for deployment testing)
# Users can skip serving env if they only want analytics, venv_serving as optional step 
# ==============================================================================

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${WHITE}[SERVING] Setting up minimal serving environment${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "${YELLOW}   This environment contains ONLY serving dependencies${NC}"
echo -e "${YELLOW}   Location: $BACKEND_DIR/venv_serving${NC}"
echo -e "${YELLOW}   Purpose: Production deployment testing + Sevalla compatibility${NC}"
echo ""

SERVING_ENV="$BACKEND_DIR/venv_serving"
SKIP_SERVING=false

# Check if environment already exists
if [ -d "$SERVING_ENV" ]; then
    echo -e "${YELLOW}[WARNING] Serving environment already exists!${NC}"
    read -p "   Delete and recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if ! remove_locked_directory "$SERVING_ENV" "Serving"; then
            echo -e "${CYAN}[SKIP] Cannot proceed with serving environment setup${NC}"
            SKIP_SERVING=true
        fi
    else
        echo -e "${CYAN}[SKIP] Keeping existing serving environment${NC}"
        SKIP_SERVING=true
    fi
fi

if [ "$SKIP_SERVING" = false ]; then
    # Create virtual environment
    echo ""
    echo -e "${YELLOW}[*] Creating serving virtual environment...${NC}"
    $PYTHON_CMD -m venv "$SERVING_ENV"
    
    if [ ! -d "$SERVING_ENV" ]; then
        echo -e "${RED}[ERROR] Failed to create serving environment!${NC}"
        echo -e "${CYAN}[SKIP] Serving environment will not be available${NC}"
    else
        echo -e "${GREEN}[OK] Virtual environment created${NC}"
        echo ""
        
        # Activate and install minimal dependencies
        source "$SERVING_ENV/bin/activate"
        
        SERVING_REQUIREMENTS="$BACKEND_DIR/environments/requirements_sevalla.txt"
        
        echo -e "${YELLOW}[*] Installing minimal serving dependencies...${NC}"
        echo -e "${CYAN}   Requirements: environments/requirements_sevalla.txt${NC}"
        echo -e "${CYAN}   This should be MUCH faster than full environment (2-3 min)${NC}"
        echo ""
        
        if [ "$UV_INSTALLED" = true ]; then
            # Use UV for fast installation
            echo -e "${CYAN}[UV] Using fast package installer...${NC}"
            uv pip install -r "$SERVING_REQUIREMENTS"
        else
            # Fallback to pip
            echo -e "${YELLOW}[PIP] Using standard pip...${NC}"
            python -m pip install --upgrade pip
            pip install -r "$SERVING_REQUIREMENTS"
        fi
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}[OK] Serving environment ready! This is your deployment-testing environment.${NC}"
        else
            echo ""
            echo -e "${RED}[ERROR] Failed to install serving dependencies!${NC}"
        fi
        
        deactivate
    fi
fi

echo ""

# ==============================================================================
# SETUP FRONTEND ENVIRONMENT
# ==============================================================================

echo -e "${CYAN}============================================================${NC}"
echo -e "${WHITE}[FRONTEND] Setting up UI environment${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "${YELLOW}   Location: $FRONTEND_ENV${NC}"
echo ""

SKIP_FRONTEND=false

# Check if environment already exists
if [ -d "$FRONTEND_ENV" ]; then
    echo -e "${YELLOW}[WARNING] Frontend environment already exists!${NC}"
    read -p "   Delete and recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Use the robust deletion function
        if ! remove_locked_directory "$FRONTEND_ENV" "Frontend"; then
            echo -e "${RED}[ERROR] Failed to remove frontend environment${NC}"
            echo -e "${CYAN}[SKIP] Cannot proceed with frontend setup${NC}"
            SKIP_FRONTEND=true
        fi
    else
        echo -e "${CYAN}[SKIP] Keeping existing frontend environment${NC}"
        SKIP_FRONTEND=true
    fi
fi

if [ "$SKIP_FRONTEND" = false ]; then
    # Create virtual environment
    echo ""
    echo -e "${YELLOW}[*] Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv "$FRONTEND_ENV"
    
    if [ ! -d "$FRONTEND_ENV" ]; then
        echo -e "${RED}[ERROR] Failed to create frontend environment!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}[OK] Virtual environment created${NC}"
    echo ""
    
    # Activate and install dependencies
    source "$FRONTEND_ENV/bin/activate"
    
    echo -e "${YELLOW}[*] Installing dependencies (this should be fast)...${NC}"
    echo -e "${CYAN}   Requirements: serving/frontend/requirements.txt${NC}"
    echo ""
    
    REQUIREMENTS_FILE="$FRONTEND_DIR/requirements.txt"
    
    if [ "$UV_INSTALLED" = true ]; then
        # Use UV for fast installation
        echo -e "${CYAN}[UV] Using fast package installer...${NC}"
        uv pip install -r "$REQUIREMENTS_FILE"
    else
        # Fallback to pip
        echo -e "${YELLOW}[PIP] Using standard pip...${NC}"
        python -m pip install --upgrade pip
        pip install -r "$REQUIREMENTS_FILE"
    fi
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}[OK] Frontend dependencies installed successfully!${NC}"
    else
        echo ""
        echo -e "${RED}[ERROR] Failed to install frontend dependencies!${NC}"
        exit 1
    fi
    
    deactivate
fi

echo ""

# ==============================================================================
# VERIFY AWS CREDENTIALS
# ==============================================================================

echo -e "${CYAN}============================================================${NC}"
echo -e "${WHITE}[CONFIG] Checking AWS credentials${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

AWS_CREDS_FILE="$BACKEND_DIR/.aws_secrets/aws_credentials.env"

if [ -f "$AWS_CREDS_FILE" ]; then
    echo -e "${GREEN}[OK] AWS credentials file found${NC}"
    echo -e "${CYAN}   Location: $AWS_CREDS_FILE${NC}"
else
    echo -e "${YELLOW}[WARNING] AWS credentials file not found!${NC}"
    echo ""
    echo -e "${YELLOW}   Expected location: $AWS_CREDS_FILE${NC}"
    echo ""
    echo -e "${WHITE}   Please create this file with your AWS credentials:${NC}"
    echo -e "${CYAN}   AWS_ACCESS_KEY_ID=your_key_here${NC}"
    echo -e "${CYAN}   AWS_SECRET_ACCESS_KEY=your_secret_here${NC}"
    echo -e "${CYAN}   AWS_REGION=us-east-1${NC}"
    echo ""
    echo -e "${RED}   The backend will not work without AWS credentials!${NC}"
fi

echo ""

# ==============================================================================
# CLEANUP OLD RENAMED ENVIRONMENTS (OPTIONAL)
# ==============================================================================

echo -e "${CYAN}============================================================${NC}"
echo -e "${WHITE}[CLEANUP] Checking for old renamed environments${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Find all directories matching *.old_* pattern
OLD_DIRS=$(find "$SCRIPT_DIR" -type d -name "*.old_*" 2>/dev/null)

if [ -n "$OLD_DIRS" ]; then
    OLD_COUNT=$(echo "$OLD_DIRS" | wc -l | tr -d ' ')
    echo -e "${YELLOW}[INFO] Found $OLD_COUNT old renamed environment(s)${NC}"
    echo "$OLD_DIRS" | while read dir; do
        echo -e "${CYAN}   - $dir${NC}"
    done
    echo ""
    read -p "   Delete these old directories? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$OLD_DIRS" | while read dir; do
            if rm -rf "$dir" 2>/dev/null; then
                echo -e "${GREEN}[OK] Deleted: $(basename $dir)${NC}"
            else
                echo -e "${YELLOW}[WARNING] Could not delete: $(basename $dir)${NC}"
            fi
        done
    fi
else
    echo -e "${GREEN}[OK] No old environments to clean up${NC}"
fi

echo ""

# ==============================================================================
# MAKE STARTUP SCRIPT EXECUTABLE
# ==============================================================================

echo -e "${CYAN}============================================================${NC}"
echo -e "${WHITE}[CONFIG] Making startup script executable${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

STARTUP_SCRIPT="$SCRIPT_DIR/start_finrag.command"
if [ -f "$STARTUP_SCRIPT" ]; then
    chmod +x "$STARTUP_SCRIPT"
    echo -e "${GREEN}[OK] Startup script is now executable${NC}"
    echo -e "${CYAN}   You can double-click start_finrag.command to launch FinRAG${NC}"
else
    echo -e "${YELLOW}[WARNING] Startup script not found: $STARTUP_SCRIPT${NC}"
fi

echo ""

# ==============================================================================
# COMPLETION
# ==============================================================================

echo -e "${GREEN}============================================================${NC}"
echo -e "${WHITE}[SUCCESS] FinRAG Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${YELLOW}Environment Locations:${NC}"
echo -e "${CYAN}   Backend:  $BACKEND_ENV${NC}"
echo -e "${CYAN}   Frontend: $FRONTEND_ENV${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "${WHITE}   1. Verify AWS credentials are configured${NC}"
echo -e "${WHITE}   2. Run './start_finrag.command' to launch FinRAG${NC}"
echo -e "${WHITE}   3. Browser will auto-open to http://localhost:8501${NC}"
echo ""
echo -e "${GREEN}[READY] Setup complete! You can now start FinRAG.${NC}"
echo ""