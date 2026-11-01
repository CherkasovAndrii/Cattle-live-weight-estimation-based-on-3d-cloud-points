#!/usr/bin/env sh
# Start the defence demo: the farm web app on this computer and on phones.
#   sh scripts/run_demo.sh /path/to/CowDatabase2 [/path/to/CowDatabase]
set -e
cd "$(dirname "$0")/.."
if [ -z "$1" ]; then
  echo "usage: sh scripts/run_demo.sh PATH_TO_CowDatabase2 [PATH_TO_CowDatabase]" >&2
  exit 2
fi
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
fi
# Quick when nothing changed; picks up new dependencies after git pull.
.venv/bin/pip install -q -e ".[api,demo]"
set -- --cowdb2 "$1" ${2:+--cowdb1 "$2"}
# Phones on any network (mobile data too) get an internet address through a
# Cloudflare tunnel; cloudflared is downloaded on first use. NO_TUNNEL=1 skips it.
if [ "${NO_TUNNEL:-0}" != 1 ]; then set -- "$@" --tunnel; fi
exec .venv/bin/cattle3d demo "$@"
