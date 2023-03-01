#!/bin/bash
#SBATCH --job-name=SuperImportantJob
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=15:00

module purge
module load gcc/10.2.0
module load cuda
module load julia/1.6.3
make clean
make

echo "TASKFLOW_VERSION"
echo "___S12___"
julia ./src/fglt ../assets/s12.mtx
julia ./src/fglt ../assets/s12.mtx
julia ./src/fglt ../assets/s12.mtx
echo "___AUTO___"
julia ./src/fglt ../assets/auto.mtx
julia ./src/fglt ../assets/auto.mtx
julia ./src/fglt ../assets/auto.mtx
echo "___GB___"
julia ./src/fglt ../assets/great-britain_osm.mtx
julia ./src/fglt ../assets/great-britain_osm.mtx
julia ./src/fglt ../assets/great-britain_osm.mtx
echo "___DL___"
julia ./src/fglt ../assets/delaunay_n22.mtx
julia ./src/fglt ../assets/delaunay_n22.mtx
julia ./src/fglt ../assets/delaunay_n22.mtx
echo "___CYT___"
julia ./src/fglt ../assets/com-Youtube.mtx
julia ./src/fglt ../assets/com-Youtube.mtx
julia ./src/fglt ../assets/com-Youtube.mtx