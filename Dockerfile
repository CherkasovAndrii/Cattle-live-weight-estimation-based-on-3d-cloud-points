# The farm service as a container, for a cloud VM or a farm server.
#   docker build -t cattle3d .
#   docker run -p 8000:8000 -e CATTLE3D_API_KEY=change-me \
#     -v /path/to/CowDatabase2:/datasets/cowdb2:ro -v cattle3d-data:/srv/data \
#     cattle3d --public-url https://your.server
FROM python:3.12-slim
# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,demo]"
COPY reports/figures/thesis ./reports/figures/thesis
VOLUME /srv/data
EXPOSE 8000
ENTRYPOINT ["cattle3d", "demo", "--no-browser", "--cowdb2", "/datasets/cowdb2", \
            "--features", "data/features.csv", "--model", "data/demo_model.joblib", "--db", "data/demo.sqlite3"]
