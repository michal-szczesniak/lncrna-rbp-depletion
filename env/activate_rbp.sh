export PROJ=.
export CONDA_PKGS_DIRS="$PROJ/conda/pkgs"
export TMPDIR="$PROJ/tmp"; mkdir -p "$TMPDIR"
source "$PROJ/miniforge3/etc/profile.d/conda.sh"
cd "$PROJ"
conda activate "$PROJ/envs/rbp-py"
