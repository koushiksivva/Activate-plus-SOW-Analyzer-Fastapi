#!/bin/bash
# Activate Oryx's virtual env (created during deployment)
source /antenv/bin/activate
# Optional: Ensure PYTHONPATH includes site-packages (usually not needed after activation)
export PYTHONPATH=$PYTHONPATH:/antenv/lib/python3.10/site-packages
# Change to app dir and start Gunicorn with Uvicorn worker for ASGI (FastAPI)
cd /home/site/wwwroot
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker --timeout 600 --bind 0.0.0.0:8000 --access-logfile - --error-logfile - main:app
