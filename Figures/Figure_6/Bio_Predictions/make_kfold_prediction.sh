#!/bin/bash
for ((m = 0; m < 5  ; m++)); do
    export m="$m"
    export bin="$bin"
    sbatch make_prediction_python.sh "$m" "$bin"
    echo "bins: $bin -- m: $m"
done