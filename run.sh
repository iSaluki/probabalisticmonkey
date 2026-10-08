#!/usr/bin/env bash
# Launch Probabilistic Monkey in a local web page.
set -e
cd "$(dirname "$0")"

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.9 or newer is required: https://www.python.org/downloads/" >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "First run: creating a virtual environment in .venv ..."
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --disable-pip-version-check -r app/requirements.txt
exec .venv/bin/python app/server.py "$@"
