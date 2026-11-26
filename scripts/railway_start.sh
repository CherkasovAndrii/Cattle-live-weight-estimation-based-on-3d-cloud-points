#!/bin/sh
# Start command of the cloud demo image (deploy/railway.Dockerfile).
# Railway sets PORT and RAILWAY_PUBLIC_DOMAIN; set CATTLE3D_API_KEY on the service.
set -e
PUBLIC_URL="${PUBLIC_URL:-${RAILWAY_PUBLIC_DOMAIN:+https://$RAILWAY_PUBLIC_DOMAIN}}"
exec cattle3d demo --no-browser --host 0.0.0.0 --port "${PORT:-8000}" ${PUBLIC_URL:+--public-url "$PUBLIC_URL"} \
  --cowdb2 /datasets/cowdb2 --cowdb1 /datasets/cowdb1 \
  --features data/features.csv --model data/demo_model.joblib --db data/demo.sqlite3
