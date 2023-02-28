#!/bin/bash
#SBATCH --job-name=SuperImportantJob
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=2:00

module purge
module load gcc
module load cuda
make clean
make

echo "OMP_VERSION"
echo "___S12___"
bin/fglt_c_cpu ../assets/s12.mtx
bin/fglt_c_cpu ../assets/s12.mtx
bin/fglt_c_cpu ../assets/s12.mtx
echo "___AUTO___"
bin/fglt_c_cpu ../assets/auto.mtx
bin/fglt_c_cpu ../assets/auto.mtx
bin/fglt_c_cpu ../assets/auto.mtx
echo "___GB___"
bin/fglt_c_cpu ../assets/great-britain_osm.mtx
bin/fglt_c_cpu ../assets/great-britain_osm.mtx
bin/fglt_c_cpu ../assets/great-britain_osm.mtx
echo "___DL___"
bin/fglt_c_cpu ../assets/delaunay_n22.mtx
bin/fglt_c_cpu ../assets/delaunay_n22.mtx
bin/fglt_c_cpu ../assets/delaunay_n22.mtx
echo "___CYT___"
bin/fglt_c_cpu ../assets/com-Youtube.mtx
bin/fglt_c_cpu ../assets/com-Youtube.mtx
bin/fglt_c_cpu ../assets/com-Youtube.mtx
