# !/bin/bash
# Variables
EXE=atmem_benchmark/atmem_bench
NUM_ELEM=(1048576)
MEM_BLOCK_SIZE=(2 4 8 16 32 64 128)
TB_SIZE=(4 8 16 32 64 128 256 512 1024)
OUTPUT_CSV_ATM=("atmem_time_baseline.csv" "atmem_cycle_baseline.csv" "atmem_time_atomic.csv" "atmem_cycle_atomic.csv")
# Initialize the output files
for file in ${OUTPUT_CSV_ATM[@]}
do
  : > ${file}
  # Column label
  printf "v Mem size/TB size >" >> ${file}
  for tb_size in ${TB_SIZE[@]}
  do
    printf ",${tb_size}" >> ${file}
  done
  echo >> ${file}
done

# Start the profiling
for num_elem in ${NUM_ELEM[@]}
  do
  echo "* Elements = ${num_elem}"
  for mem_size in ${MEM_BLOCK_SIZE[@]}
  do
    echo "-- Memory block size = ${mem_size}"
    # Row label
    for file in ${OUTPUT_CSV_ATM[@]}
    do
      printf "${mem_size}" >> ${file}
    done
    # Run the benchmark
    for tb_size in ${TB_SIZE[@]}
    do
      echo ">>> TB size = ${tb_size}"
      ./${EXE} ${num_elem} ${tb_size} ${mem_size} 0 > LOG
      printf ",$(cat LOG | grep "Total elapsed kernel time" | awk '{print $(NF-1)}')" >> ${OUTPUT_CSV_ATM[0]}
      printf ",$(cat LOG | grep "Max in-SM cycles" | awk '{print $(NF-1)}')" >> ${OUTPUT_CSV_ATM[1]}
      ./${EXE} ${num_elem} ${tb_size} ${mem_size} 1 > LOG
      printf ",$(cat LOG | grep "Total elapsed kernel time" | awk '{print $(NF-1)}')" >> ${OUTPUT_CSV_ATM[2]}
      printf ",$(cat LOG | grep "Max in-SM cycles" | awk '{print $(NF-1)}')" >> ${OUTPUT_CSV_ATM[3]}
    done
    # Go to next line in the output files
    for file in ${OUTPUT_CSV_ATM[@]}
    do
      echo >> ${file}
    done   
    # End
  done
done
# End of profiling
rm -f LOG
echo "End of profiling."

