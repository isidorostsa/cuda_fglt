#!/bin/bash
#SBATCH --job-name=SuperImportantJob
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=15:00

module purge
module load gcc
module load cuda
make clean
make

echo "THRUST_VERSION"
echo "___S12___"
bin/fglt ../assets/s12.mtx
bin/fglt ../assets/s12.mtx
bin/fglt ../assets/s12.mtx
echo "___AUTO___"
bin/fglt ../assets/auto.mtx
bin/fglt ../assets/auto.mtx
bin/fglt ../assets/auto.mtx
echo "___GB___"
bin/fglt ../assets/great-britain_osm.mtx
bin/fglt ../assets/great-britain_osm.mtx
bin/fglt ../assets/great-britain_osm.mtx
echo "___DL___"
bin/fglt ../assets/delaunay_n22.mtx
bin/fglt ../assets/delaunay_n22.mtx
bin/fglt ../assets/delaunay_n22.mtx
echo "___CYT___"
bin/fglt ../assets/com-Youtube.mtx
bin/fglt ../assets/com-Youtube.mtx
bin/fglt ../assets/com-Youtube.mtx

echo "CUDA_VERSION"
echo "___S12___"
bin/fglt_c ../assets/s12.mtx
bin/fglt_c ../assets/s12.mtx
bin/fglt_c ../assets/s12.mtx
echo "___AUTO___"
bin/fglt_c ../assets/auto.mtx
bin/fglt_c ../assets/auto.mtx
bin/fglt_c ../assets/auto.mtx
echo "___GB___"
bin/fglt_c ../assets/great-britain_osm.mtx
bin/fglt_c ../assets/great-britain_osm.mtx
bin/fglt_c ../assets/great-britain_osm.mtx
echo "___DL___"
bin/fglt_c ../assets/delaunay_n22.mtx
bin/fglt_c ../assets/delaunay_n22.mtx
bin/fglt_c ../assets/delaunay_n22.mtx
echo "___CYT___"
bin/fglt_c ../assets/com-Youtube.mtx
bin/fglt_c ../assets/com-Youtube.mtx
bin/fglt_c ../assets/com-Youtube.mtx