#!/bin/bash

for seed in {0..9}; do
	echo "Benchmark: seed $seed"
	data_path=../sim_output/benchmark-seed-$seed
	mkdir $data_path
	sim_date=$(date +%F_%H-%M-%S)
	{ time python3 run_benchmark.py benchmark_times_$sim_date.json --path=$data_path --seed=$seed 2> ../sim_output/run_benchmark_$sim_date.err; } &> ../sim_output/run_benchmark_$sim_date.out
	# For slurm change above command to this:
	# sbatch sbatch_benchmark.sh $data_path $seed
	rm -rf potjans_microcircuit_CODE
done
