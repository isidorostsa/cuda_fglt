#!/bin/bash
#SBATCH --job-name=SuperImportantJob
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00

module purge
module load gcc
module load cuda
make clean
make

ASSETS="./assets"

echo "THRUST_VERSION"
echo "___S12___"
bin/fglt $ASSETS/s12.mtx
bin/fglt $ASSETS/s12.mtx
bin/fglt $ASSETS/s12.mtx
echo "___AUTO___"
bin/fglt $ASSETS/auto.mtx
bin/fglt $ASSETS/auto.mtx
bin/fglt $ASSETS/auto.mtx
echo "___GB___"
bin/fglt $ASSETS/great-britain_osm.mtx
bin/fglt $ASSETS/great-britain_osm.mtx
bin/fglt $ASSETS/great-britain_osm.mtx
echo "___DL___"
bin/fglt $ASSETS/delaunay_n22.mtx
bin/fglt $ASSETS/delaunay_n22.mtx
bin/fglt $ASSETS/delaunay_n22.mtx
echo "___CYT___"
bin/fglt $ASSETS/com-Youtube.mtx
bin/fglt $ASSETS/com-Youtube.mtx
bin/fglt $ASSETS/com-Youtube.mtx

echo "CUDA_VERSION"
echo "___S12___"
bin/fglt_c $ASSETS/s12.mtx
bin/fglt_c $ASSETS/s12.mtx
bin/fglt_c $ASSETS/s12.mtx
echo "___AUTO___"
bin/fglt_c $ASSETS/auto.mtx
bin/fglt_c $ASSETS/auto.mtx
bin/fglt_c $ASSETS/auto.mtx
echo "___GB___"
bin/fglt_c $ASSETS/great-britain_osm.mtx
bin/fglt_c $ASSETS/great-britain_osm.mtx
bin/fglt_c $ASSETS/great-britain_osm.mtx
echo "___DL___"
bin/fglt_c $ASSETS/delaunay_n22.mtx
bin/fglt_c $ASSETS/delaunay_n22.mtx
bin/fglt_c $ASSETS/delaunay_n22.mtx
echo "___CYT___"
bin/fglt_c $ASSETS/com-Youtube.mtx
bin/fglt_c $ASSETS/com-Youtube.mtx
bin/fglt_c $ASSETS/com-Youtube.mtx
