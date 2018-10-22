# !/bin/bash
# Usage: ./atmem_profiler.sh [list of number of elements (default = 2^20)]
# Variables
EXE=atmem_benchmark/atmem_bench
NUM_ELEM=(1048576)
MEM_BLOCK_SIZE=(2 4 8 16 32 64 128 256 512 1024 2048)
TB_SIZE=(2 4 8 16 32 64 128 256 512 1024 2048)
OUTPUT_TIME=atmem_time.csv
OUTPUT_CYCLE=atmem_cycle.csv

if (( $# > 0 )); then
  NUM_ELEM=$@
fi

# Initialize the output files
: > ${OUTPUT_TIME}
: > ${OUTPUT_CYCLE}

# CSV format:
# Table is divided among inputs in a row-wise fashion
# Columns: Varying TB sizes, in pairs of baseline and atomic modes (in order)
# Rows: Varying Memory block sizes

for num_elem in ${NUM_ELEM[@]}
do
  echo "* Elements = ${num_elem}"

  # Column label
  printf "v Mem size/${num_elem}/TB size_Mode >" >> ${OUTPUT_TIME}
  printf "v Mem size/${num_elem}/TB size_Mode >" >> ${OUTPUT_CYCLE}
  for tb_size in ${TB_SIZE[@]}
  do
    printf ",${tb_size}_BL,${tb_size}_AT" >> ${OUTPUT_TIME}
    printf ",${tb_size}_BL,${tb_size}_AT" >> ${OUTPUT_CYCLE}
  done
  echo >> ${OUTPUT_TIME}
  echo >> ${OUTPUT_CYCLE}

  # Start the profiling
  for mem_size in ${MEM_BLOCK_SIZE[@]}
  do
    echo "-- Memory block size = ${mem_size}"
    # Row label
    printf "${mem_size}" >> ${OUTPUT_TIME}
    printf "${mem_size}" >> ${OUTPUT_CYCLE}
    # Run the benchmark
    for tb_size in ${TB_SIZE[@]}
    do
      echo ">>> TB size = ${tb_size}"
      ./${EXE} ${num_elem} ${tb_size} ${mem_size} 0 0 > LOG
      printf ",$(cat LOG | grep "Total elapsed kernel time" | awk '{print $(NF-1)}')" >> ${OUTPUT_TIME}
      printf ",$(cat LOG | grep "Max in-SM cycles" | awk '{print $(NF-1)}')" >> ${OUTPUT_CYCLE}
      ./${EXE} ${num_elem} ${tb_size} ${mem_size} 1 0 > LOG
      printf ",$(cat LOG | grep "Total elapsed kernel time" | awk '{print $(NF-1)}')" >> ${OUTPUT_TIME}
      printf ",$(cat LOG | grep "Max in-SM cycles" | awk '{print $(NF-1)}')" >> ${OUTPUT_CYCLE}
    done
    # Go to next line in the output files
    echo >> ${OUTPUT_TIME}
    echo >> ${OUTPUT_CYCLE}
  done
done
# End of profiling
rm -f LOG
echo "End of profiling."

