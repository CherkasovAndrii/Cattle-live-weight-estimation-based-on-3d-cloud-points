"""The web console: the React app built from ``web/`` (``npm run build``),
served by the API at ``/``. The build is committed so a farm computer needs no Node."""

from pathlib import Path

SITE_DIR = Path(__file__).parent / "static"
