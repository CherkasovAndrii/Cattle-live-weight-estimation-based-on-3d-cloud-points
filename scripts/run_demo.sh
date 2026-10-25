#!/usr/bin/env sh
# Start the defence demo: the farm web app on this computer and on phones in
# the same Wi-Fi network.
#   sh scripts/run_demo.sh /path/to/CowDatabase2 [/path/to/CowDatabase]
set -e
cd "$(dirname "$0")/.."
if [ -z "$1" ]; then
  echo "usage: sh scripts/run_demo.sh PATH_TO_CowDatabase2 [PATH_TO_CowDatabase]" >&2
  exit 2
fi
if [ ! -x .venv/bin/cattle3d ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -e ".[api,demo]"
fi
set -- --cowdb2 "$1" ${2:+--cowdb1 "$2"}
exec .venv/bin/cattle3d demo "$@"
