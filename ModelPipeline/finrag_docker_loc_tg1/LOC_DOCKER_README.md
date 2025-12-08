## FinRAG Docker Local Setup Guide

The flow: **Clone -> Set up AWS creds -> Navigate once -> Build + Start**

**Clone repo**
- git clone https://github.com/Finsights-MLOps/FinSights.git
- cd FinSights/ModelPipeline

**Set up AWS credentials (once, after cloning the repo.)**
- Location `# Path: finrag_ml_tg1/.aws_secrets/aws_credentials.env`
- A template file is provided at `# Path: finrag_ml_tg1/.aws_secrets/example_aws_credentials.env`
- Add your AWS keys

**Ensuring you're at the path where docker-compose.yml is located**
- Ensure you're at 'FinSights/' path first. 
- cd "ModelPipeline\finrag_docker_loc_tg1"

**[Optional] Checking commands (for devs):**
- Check what Docker will send to build context (before building)
    - docker compose build --no-cache --progress=plain backend 2>&1 | Select-String "COPY"
    - docker compose build --no-cache --progress=plain backend

**Full Build + Start commands:**
   - docker compose up --build


- This builds both backend and frontend images, then starts the containers. 

---

**If necessary: Down, Removal.**
- To do basic, up and down:
  - docker compose down
  - docker compose restart  
  - docker compose up                               ( runs in foreground mode (attached) )
  - docker compose up --build                       ( does a rebuild )

- [Optional] If already running containers, stop first:
    - docker compose down
- [Optional] If you need to remove images to force fresh build
    - docker compose down -v --remove-orphans
    - docker rmi -f finrag_docker_loc_tg1-backend finrag_docker_loc_tg1-frontend
    - docker compose up --build
- [Optional] Prune
  - docker builder prune -a -f ( Removes ALL images not associated with a running container)
- If want run in detached mode (background):
   - docker compose up --build -d
- To view logs:
  -  docker compose logs -f
- If ever necessary to see *progress, full build logs*; example, for backend: 
   - docker compose --progress plain build --no-cache backend


### Docker Image Size Breakdown:
```
Base Layer (Python 3.12-slim):
├─ 87.4MB  → Debian base OS
├─ 41.3MB  → Python runtime setup
├─ 4.94MB  → Additional system setup
└─ Total: ~133MB base

Your Dockerfile Layers:
├─ 336MB   → RUN apt-get install build-essential curl  ← BUILD TOOLS
├─ 66.9MB  → COPY . .                                    ← CODE
├─ 592MB   → RUN pip install requirements               ← PYTHON PACKAGES
└─ Total: ~995MB added by your Dockerfile
─────────────────────────────────────────────────────
TOTAL: 133MB + 995MB = 1.55GB 
```

- **1 image: 1.55GB (finrag_docker_loc_tg1-backend),** 
- **2 images: 1.38GB (finrag_docker_loc_tg1-frontend)**
✅ Base OS + Python: 133MB
✅ Build tools: 336MB (needed for compiling)
✅ Code: 67MB
✅ Python packages: 592MB (ML dependencies)

--- 
### Docker Inspection & Optimization Commands

1. View all images with sizes
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

2. View only FinRAG images
docker images | Select-String "finrag"

```shell
>> 
REPOSITORY                            TAG                 SIZE      CREATED AT
finrag_docker_loc_tg1-backend         latest              1.55GB    2025-12-08 01:25:15 -0500 EST
finrag_docker_loc_tg1-frontend        latest              1.38GB    2025-12-08 01:24:55 -0500 EST
```

3. View image layer history
docker history finrag_docker_loc_tg1-backend --human --no-trunc
docker history finrag_docker_loc_tg1-frontend --human --no-trunc

4. View large layers only (over 100MB)
docker history finrag_docker_loc_tg1-backend --format "table {{.Size}}\t{{.CreatedBy}}" | Select-String -Pattern "MB|GB"

```shell
PS D:\JoelDesktop folds_24\NEU FALL2025\MLops IE7374 Project\FinSights\ModelPipeline\finrag_docker_loc_tg1> docker history finrag_docker_loc_tg1-backend --format "table {{.Size}}\t{{.CreatedBy}}" | Select-String -Pattern "MB|GB"
>>

592MB     RUN /bin/sh -c pip install --upgrade pip && ΓÇª
66.9MB    COPY . . # buildkit
336MB     RUN /bin/sh -c apt-get update && apt-get insΓÇª
41.3MB    RUN /bin/sh -c set -eux;   savedAptMark="$(aΓÇª
4.94MB    RUN /bin/sh -c set -eux;  apt-get update;  aΓÇª
87.4MB    # debian.sh --arch 'amd64' out/ 'trixie' '@1ΓÇª
```

docker system df -v

6. Freeing up in system; Do it at your own risk. Just to clear up space, unused data, build cache, stopped containers.
docker builder prune -f
docker container prune -f
docker image prune -a --dry-run 


### Safer Clean Patterns:
1. Keep containers running while cleaning
    docker compose up -d

2. Clean build cache (safe - doesn't touch images)
    docker builder prune -f

3. Clean stopped containers (safe - doesn't touch YOUR running containers)
    docker container prune -f

4. Clean old images carefully
    docker images  # Review what's there
    docker rmi <specific-old-image>  # Delete specific ones only

5. Stop services (keeps containers)
docker compose stop

6. Stop and remove containers (keeps images)
docker compose down

7. Stop, remove containers, networks, volumes (keeps images)
docker compose down -v

8. Full clean: Stop, remove containers, networks, volumes, images
docker compose down --rmi all
docker compose down -v --rmi all --remove-orphans

9. Restart services (after stopping)
docker compose restart

10. View logs, -f gives live tail
docker compose logs
docker compose logs -f
docker compose logs backend
docker compose logs frontend

11. View running containers and their status
docker compose ps
docker compose ps -a
docker stats
docker stats finrag-backend finrag-frontend

12. View all Docker images with sizes
docker images
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

13. View and prune volumes
docker volume ls
docker volume prune
docker volume rm <volume-name>

14. Remove specific images (if needed)
docker rmi finrag_docker_loc_tg1-backend
docker rmi finrag_docker_loc_tg1-frontend

--- 




## -- Extra info while debugging networking:

1. Normal Streamlit (local terminal):
   - Local URL: http://localhost:8501
   - Network URL: http://192.168.1.100:8501  ← This lets others on your WiFi access it

2. Docker Streamlit:
   -   URL: http://localhost:8501  ← Only this one

- container CAN'T see your host machine's network IP:
```
Machine (192.168.1.100)
    ↓
Docker Container (isolated network)
    ↓
Streamlit inside container sees: 172.17.0.x (Docker's internal IP)
    ↓
Container has NO IDEA host is 192.168.1.100
```
- Docker's bridge network isolates the container, so Streamlit can't auto-detect your actual network IP.


## Author
- Joel Markapudi ( markapudi.j@northeastern.edu )
- Feel free to reach out for any issues.