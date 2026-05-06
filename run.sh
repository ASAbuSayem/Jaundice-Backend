#!/bin/bash
echo "Starting NJN Jaundice API..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
