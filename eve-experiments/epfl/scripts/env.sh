# Sourced by the task entrypoints.
task_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
eve_dir=${EVE_CHECKOUT:-"$task_dir/.eve"}
python_bin=${EVE_PYTHON:-"$eve_dir/.venv/bin/python"}
export PATH="$eve_dir/.venv/bin:$PATH"
export PYTHONPATH="$eve_dir:$eve_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$eve_dir"
