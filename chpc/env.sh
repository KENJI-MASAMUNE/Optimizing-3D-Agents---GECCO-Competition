#!/bin/bash
# Environment for running the DE-NSGA-II experiments on the CHPC (Lengau, PBS Pro).
# Sourced by every PBS script and by setup_env.sh. EDIT THE MARKED LINES ONCE.
#
# Expected layout on Lustre (fast scratch; home has a small quota):
#   $WORK/repo          this repository (git clone / rsync)
#   $WORK/framspy       framspy checkout (FramsticksLib.py, frams.py, frams_extvalue.py, ...)
#   $WORK/Framsticks55  Framsticks distribution: frams-objects.so + data/  (Linux .so is in the
#                       same Framsticks55 folder you use on Windows)
#   $WORK/results       all run outputs (JSON per run) -- back this up / rsync it home

# ---- EDIT: your CHPC project short name (shown in the CHPC user database), used by qsub -P
export CHPC_PROJECT="${CHPC_PROJECT:-CSCI0806}"

# ---- EDIT if you use a different location
export WORK="${WORK:-/mnt/lustre/users/spatel1/cos700}"

export REPO="${REPO:-$HOME/cos700code}"           # code lives in home (Lustre hung on small files, Oct 2026)
export FRAMSPY_PATH="$WORK/framspy"
export FRAMS_DIR="$WORK/Framsticks55"
export RESULTS_DIR="${RESULTS_DIR:-$WORK/results}"            # submit.sh comp overrides these two
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$REPO/configs/chpc_config.yaml}"
export PYTHONPATH="$REPO:$FRAMSPY_PATH:${PYTHONPATH:-}"

# One thread per process: parallelism comes from running many independent runs at once.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ---- Python environment
# Lengau has NO internet on login/compute nodes (GitHub and PyPI are unreachable), and its
# Python modules stop at 3.9 while framspy needs Python >= 3.12 (frams_extvalue.py uses 3.12
# syntax). So we use a standalone CPython 3.12 build (glibc 2.17-compatible, runs on Lengau)
# plus a venv, both installed OFFLINE from a bundle made on your laptop with
# chpc/make_bundle.ps1 -- see chpc/setup_env.sh.
export PY_HOME="$WORK/python312"
export VENV_DIR="$WORK/venv312"

# frams-objects.so needs GLIBCXX_3.4.26 (GCC >= 9); the default `gcc` module (6.1.0) and the
# system library are too old, gcc/9.2.0 is new enough.
if ! type module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh || true
fi
if type module >/dev/null 2>&1; then
    module unload gcc >/dev/null 2>&1 || true
    module load gcc/9.2.0 >/dev/null 2>&1 || echo "[env.sh] WARNING: could not load gcc/9.2.0" >&2
fi
GCC_LIBDIR=""
if command -v gcc >/dev/null 2>&1; then
    GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
    export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}"
fi
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi
