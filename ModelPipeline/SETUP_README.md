## Update: Better setup process!
- **Preferred Setup Method:** Please use this, a dockerized setup for proper compatibility.
- [Dockerized Setup Guide](./finrag_docker_loc_tg1/LOC_DOCKER_README.md)
- The below process is a suitable setup but it just uses quick start scripts which are based off either command files or PS1 shell files. It does achieve all the automation with a single click process but we still prefer using Docker here because it containerizes the application cleanly.

## Prerequisites
- Python 3.12 installed
- Git (to clone the repository)
- Windows: PowerShell | Mac/Linux: Terminal
- Please ensure your respective python (3.12) is cleanly installed, and your PATH variables are set correctly.

### Clone the Repository
```bash
git clone https://github.com/Finsights-MLOps/FinSights.git
cd FinSights/ModelPipeline
```

### Configure AWS Credentials
Create or update `finrag_ml_tg1/.aws_secrets/aws_credentials.env`:
```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1
```

### Make Scripts Executable (Mac/Linux Only)
- If your MAC command files are not executable in terminal, please run: 
```bash
cd ModelPipeline
chmod +x start_finrag.command
chmod +x setup_finrag.command
```

## Quick Setup (Automated)

### Windows
1. Navigate to `ModelPipeline/` folder
2. **Double-click** `setup_finrag.bat`
3. Wait 1-2 minutes for setup to complete
4. UV will be installed automatically for fast dependency resolution
5. Will automatically handle the issues of terminating Python processes and safely deleting old environment if the user wishes to delete, recreate. Graceful handling. ( Process killing, Rename Fallback, Cleanup utility. )

### Mac/Linux
1. Navigate to `ModelPipeline/` folder
2. Make script executable: `chmod +x setup_finrag.sh`
3. **Double-click** `setup_finrag.sh` (or run `./setup_finrag.sh`)
4. Wait 1-2 minutes for setup to complete
5. UV will be installed automatically for fast dependency resolution


## Starting FinRAG
1. After setup, just double-click `start_finrag.bat` (Windows) or `start_finrag.sh` (Mac/Linux).
2. Give it roughly 20s. There's an intentional 8-second backend, 6-second frontend sleep.
3. You should see like three terminals popping up and finally your browser would automatically open the streamlit interface through which communication can be done and queries sent.


### Quick check:
1. virtual environments MUST be in these exact locations:
    - `ModelPipeline/finrag_ml_tg1/venv_ml_rag/`
    - `ModelPipeline/serving/frontend/venv_frontend/`
```
ModelPipeline/
├── finrag_ml_tg1/
│   └── (no venv_ml_rag yet)
└── serving/frontend/
    └── (no venv_frontend yet)
```

2. UV auto-install - (need) 20x faster dependency installation than pip.
3. Workflow 1: "Just check if UV is installed"
4. Workflow 2: "I broke my backend environment" - Just re-run `setup_finrag` script.
5. Workflow 3: "Fresh start everything/ New PC setup" - Just re-run `setup_finrag` script! `:)` 

---


### Essence of setup work:

**Backend (FastAPI)**
- Code root: ModelPipeline/
- Entrypoint: serving/backend/api_service:app
- Command: uvicorn ... --host 0.0.0.0 --port 8000
- Python deps for serving are in finrag_ml_tg1/environments/requirements_sevalla.txt
- Needs AWS creds + LOG_LEVEL env vars.

**Frontend (Streamlit)**
- Code under serving/frontend/
- Command: streamlit run serving/frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
- Reads BACKEND_URL env var, defaulting to localhost in code.

**PowerShell / .bat helpers:**
- Pick the right venv (venv_serving, venv_frontend),
- Install a curated requirements file,
- Then run exactly those commands.

**What Nixpacks is good at:**
- Simple repos where it can see requirements.txt at the build path root and there’s a single obvious web process.
- Not for FinSights: Monorepo (FinSights) with deep sub-dirs (ModelPipeline, serving/frontend) etc. Multiple services (backend + frontend). Custom requirements file locations.


---

---


## Manual process for installing envs and running terminals:

**Setup of 2 environments**
```python
# ============================================
# ONE-TIME SETUP (WINDOWS)
# ============================================

## Backend Environment (Minimal):
cd ModelPipeline\finrag_ml_tg1
uv venv venv_backend
.\venv_backend\Scripts\Activate.ps1
uv pip install -r environments\requirements_app_backend.txt
deactivate

## Frontend Environment:
cd ModelPipeline\serving\frontend
uv venv venv_frontend
.\venv_frontend\Scripts\Activate.ps1
uv pip install -r requirements.txt
deactivate


# ============================================
# RUN APPLICATION (Two Terminals)
# ============================================

## Terminal 1 - Backend:
# Kill any process on port 8000
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force

cd ModelPipeline\finrag_ml_tg1
.\venv_backend\Scripts\Activate.ps1
cd ..\serving
uvicorn backend.api_service:app --reload --host 0.0.0.0 --port 8000

## Terminal 2 - Frontend:
# Kill any process on port 8501 (Streamlit default)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force

cd ModelPipeline\finrag_ml_tg1
..\serving\frontend\venv_frontend\Scripts\Activate.ps1
cd ..\serving
streamlit run frontend\app.py
```

-------------------------------------------------------------------------------------------------

**FOR MAC USERS** - same commands, just minimal syntax changes.

```python
# ============================================
# ONE-TIME SETUP (MAC/LINUX)
# ============================================

## Backend Environment (Minimal):
cd ModelPipeline/finrag_ml_tg1
uv venv venv_backend
source venv_backend/bin/activate
uv pip install -r environments/requirements_app_backend.txt
deactivate

## Frontend Environment:
cd ModelPipeline/serving/frontend
uv venv venv_frontend
source venv_frontend/bin/activate
uv pip install -r requirements.txt
deactivate


# ============================================
# RUN APPLICATION (Two Terminals)
# ============================================

## Terminal 1 - Backend:
# Kill any process on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

cd ModelPipeline/finrag_ml_tg1
source venv_backend/bin/activate
cd ../serving
uvicorn backend.api_service:app --reload --host 0.0.0.0 --port 8000

## Terminal 2 - Frontend:
# Kill any process on port 8501 (Streamlit default)
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

cd ModelPipeline/finrag_ml_tg1
source ../serving/frontend/venv_frontend/bin/activate
cd ../serving
streamlit run frontend/app.py
```




