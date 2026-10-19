#!/bin/bash
# One-off setup on a Lengau LOGIN node, fully OFFLINE (Lengau nodes cannot reach GitHub/PyPI).
#
# Before running this, on your LAPTOP (has internet), in PowerShell from the repo folder:
#     powershell -ExecutionPolicy Bypass -File chpc\make_bundle.ps1
# then upload the resulting chpc_bundle folder:
#     scp -r chpc_bundle spatel1@lengau.chpc.ac.za:~/lustre/cos700/
#
# Then on the CHPC:   bash chpc/setup_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
BUNDLE="$WORK/chpc_bundle"
mkdir -p "$WORK" "$RESULTS_DIR"

echo ">> checking the offline bundle in $BUNDLE"
[ -f "$BUNDLE/python312.tar.gz" ] || { echo "   MISSING $BUNDLE/python312.tar.gz -- run chpc/make_bundle.ps1 on your laptop and upload chpc_bundle"; exit 1; }
[ -d "$BUNDLE/wheels" ] || { echo "   MISSING $BUNDLE/wheels"; exit 1; }
echo "   OK ($(ls "$BUNDLE/wheels" | wc -l) wheels)"

if [ ! -x "$PY_HOME/bin/python3" ]; then
    echo ">> unpacking standalone Python 3.12 into $PY_HOME"
    tmp="$(mktemp -d "$WORK/.pyunpack.XXXX")"
    tar -xzf "$BUNDLE/python312.tar.gz" -C "$tmp"
    rm -rf "$PY_HOME"
    mv "$tmp/python" "$PY_HOME"
    rmdir "$tmp"
fi
"$PY_HOME/bin/python3" --version

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo ">> creating venv $VENV_DIR"
    "$PY_HOME/bin/python3" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo ">> installing packages offline"
pip install -q --no-index --find-links "$BUNDLE/wheels" -r "$REPO/requirements.txt"

echo ">> checking libstdc++ provides GLIBCXX_3.4.26 (needed by frams-objects.so)"
if strings "$GCC_LIBDIR/libstdc++.so.6" 2>/dev/null | grep -q GLIBCXX_3.4.26; then
    echo "   OK ($GCC_LIBDIR)"
else
    echo "   MISSING -- 'module load gcc/9.2.0' did not provide it; frams-objects.so will not load"
fi

echo ">> checking the Framsticks install"
for f in "$FRAMSPY_PATH/FramsticksLib.py" "$FRAMSPY_PATH/frams.py" "$FRAMSPY_PATH/frams_extvalue.py" \
         "$FRAMS_DIR/frams-objects.so" "$FRAMS_DIR/data/scripts/standard-eval.expdef"; do
    [ -e "$f" ] && echo "   found   $f" || echo "   MISSING $f"
done

source "$HERE/env.sh"
cd "$REPO"
echo ">> python: $(command -v python) ($(python --version))"
python scripts/check_framsticks.py --config "$EXPERIMENT_CONFIG"
echo ">> unit tests"
python -m pytest tests -q
echo
echo "SETUP OK. Next: bash chpc/submit.sh smoke"
