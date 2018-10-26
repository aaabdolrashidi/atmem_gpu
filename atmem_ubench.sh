# !/bin/bash
# Usage: ./atmem_ubench.sh [list of number of elements (default = 2^20)]
# Variables
EXE=TestCUDA.exe
NUM_ELEM=(1048576)
MEM_BLOCK_SIZE=(2 4 8 16 32 64 128 256 512 1024 2048)
TB_SIZE=(1 2 4 8 16 32 64 128 256 512 1024)
MODE_ARRAY=(2 3 4 5)
OUTPUT_CYCLE=atmem_bench_cycle.csv
OUTPUT_CYCLE_MODE_FAR=atmem_far_cycle.csv
if (( $# > 0 )); then
  NUM_ELEM=$@
fi

# Initialize the output files
: > ${OUTPUT_CYCLE}
: > ${OUTPUT_CYCLE_MODE_FAR}
# CSV format:
# Table is divided among inputs in a row-wise fashion
# Columns: Varying TB sizes
# Rows: Modes

for num_elem in ${NUM_ELEM[@]}
do
  echo "* Elements = ${num_elem}"

  # Column label
  printf "TB size >" >> ${OUTPUT_CYCLE}
  for tb_size in ${TB_SIZE[@]}
  do
    printf ",${tb_size}" >> ${OUTPUT_CYCLE}
  done
  echo >> ${OUTPUT_CYCLE}

  # Start the profiling
  for mode in ${MODE_ARRAY[@]}
  do
    echo "-- Mode = ${mode}"
    # Row label
    printf "Mode ${mode}" >> ${OUTPUT_CYCLE}
    # Run the benchmark
    for tb_size in ${TB_SIZE[@]}
    do
      echo ">>> TB size = ${tb_size}"
      ./${EXE} ${num_elem} ${tb_size} 32 ${mode} 0 > LOG
      printf ",$(cat LOG | grep "Max in-SM cycles" | awk '{print $(NF-1)}')" >> ${OUTPUT_CYCLE}
    done
    # Go to next line in the output files
    echo >> ${OUTPUT_CYCLE}
  done
  # Distance mode for mode 5
  for mem_size in ${MEM_BLOCK_SIZE[@]}
  do
    ./${EXE} ${num_elem} 1024 ${mem_size} 5 0 > LOG
    printf ",$(cat LOG | grep "Max in-SM cycles" | awk '{print $(NF-1)}')" >> ${OUTPUT_CYCLE_MODE_FAR}
  done
  echo >> ${OUTPUT_CYCLE_MODE_FAR}
done
# End of profiling
rm -f LOG
echo "End of microbenchmarking."

