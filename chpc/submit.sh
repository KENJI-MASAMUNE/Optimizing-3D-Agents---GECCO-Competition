#!/bin/bash
# Submit a phase as a PBS job array.
#   bash chpc/submit.sh smoke
#   bash chpc/submit.sh 1 [N_CHUNKS] [WALLTIME]     e.g. bash chpc/submit.sh 1 8 12:00:00
#   bash chpc/submit.sh 2 [N_CHUNKS] [WALLTIME]
#   bash chpc/submit.sh 3 [N_CHUNKS] [WALLTIME]     (after: python -m src.experiments.cli phase3-select)
#   bash chpc/submit.sh comp [N_CHUNKS] [WALLTIME]  competition comparison (configs/competition_config.yaml,
#                                                    results in $WORK/results_competition; runs take 8-12 h)
# Check progress with:  python -m src.experiments.cli status --phase N
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PER_NODE=100                                     # ~100 short runs per 23-core node
if [ "${1:-}" = "comp" ]; then                   # competition comparison = Phase 1 manifest of another config
    export WORK="${WORK:-/mnt/lustre/users/$USER/cos700}"
    export EXPERIMENT_CONFIG="$WORK/repo/configs/competition_config.yaml"
    export RESULTS_DIR="$WORK/results_competition"
    set -- 1 "${@:2}"
    PER_NODE=23                                  # one 8-12 h run per core
fi
source "$HERE/env.sh"
cd "$REPO"
mkdir -p "$RESULTS_DIR/logs"
PHASE="${1:?usage: submit.sh smoke|1|2|3 [N_CHUNKS] [WALLTIME]}"
if [ "$PHASE" = "smoke" ]; then
    qsub -P "$CHPC_PROJECT" chpc/smoke.pbs
    exit 0
fi
N_RUNS=$(python -m src.experiments.cli list --phase "$PHASE" --show 0 2>/dev/null | head -1 | awk '{print $3}')
DEFAULT_CHUNKS=$(( (N_RUNS + PER_NODE - 1) / PER_NODE ))
N_CHUNKS="${2:-$DEFAULT_CHUNKS}"
WALLTIME="${3:-24:00:00}"
python -m src.experiments.cli status --phase "$PHASE"
echo "Submitting phase $PHASE: $N_RUNS runs as $N_CHUNKS array elements (walltime $WALLTIME)"
if [ "$N_CHUNKS" -gt 1 ]; then
    qsub -P "$CHPC_PROJECT" -N "cos700_p$PHASE" -J "0-$((N_CHUNKS - 1))" -l walltime="$WALLTIME" \
         -o "$RESULTS_DIR/logs/" -v PHASE="$PHASE",N_CHUNKS="$N_CHUNKS",EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG",RESULTS_DIR="$RESULTS_DIR" chpc/phase.pbs
else  # PBS Pro arrays need at least 2 elements
    qsub -P "$CHPC_PROJECT" -N "cos700_p$PHASE" -l walltime="$WALLTIME" \
         -o "$RESULTS_DIR/logs/" -v PHASE="$PHASE",N_CHUNKS=1,EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG",RESULTS_DIR="$RESULTS_DIR" chpc/phase.pbs
fi
