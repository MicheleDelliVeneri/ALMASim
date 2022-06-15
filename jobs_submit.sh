#!/bin/bash
file="sims_parameters.txt"

INPUT="sims_param.csv"
OLDIFS=$IFS
IFS=","
main_path="$PWD"

[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read index input_dir output_dir
do
	echo "index : $index"
	echo "input dir : $input_dir"
	echo "output dir : $output_dir"
    input_dir="$main_path/$input_dir"
    output_dir="$main_path/$output_dir"
    mkdir -p "sim_$index"
    cp alma.cycle9.3.cfg sim_$index/
    cd "sim_$index"
    conda run -n casa6.5 python "$main_path/alma_simulator.py" $index $input_dir $output_dir
    cd ..
    rm -r "sim_$index"    
done < $INPUT
IFS=$OLDIFS