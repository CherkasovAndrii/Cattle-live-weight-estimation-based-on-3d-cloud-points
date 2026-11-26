# The public cloud demo (Railway): both demo farms on the public herds, ready at start.
# The build downloads the public datasets (github.com/ruchaya), computes features, trains
# the demo model, seeds the farms and keeps only the demo animals' clouds in the image.
#   service variables: CATTLE3D_API_KEY (required); Railway sets PORT and RAILWAY_PUBLIC_DOMAIN.
FROM python:3.12-slim
# LightGBM needs the OpenMP runtime; git and git-lfs fetch the datasets at build time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,demo]"
COPY reports/figures/thesis ./reports/figures/thesis
COPY scripts/railway_prepare.py scripts/railway_start.sh ./scripts/
# One layer, so the full checkouts never land in the image, only the pruned ones.
RUN git lfs install --skip-repo \
    && git clone --depth 1 https://github.com/ruchaya/CowDatabase2 /datasets/cowdb2 \
    && git clone --depth 1 https://github.com/ruchaya/CowDatabase /datasets/cowdb1 \
    && du -sh /datasets/* \
    && python scripts/railway_prepare.py /datasets/cowdb2 /datasets/cowdb1 data \
    && du -sh /datasets/* data
EXPOSE 8000
CMD ["sh", "scripts/railway_start.sh"]
