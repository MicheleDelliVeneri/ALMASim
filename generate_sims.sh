#!/bin/bash
i=$1
input_dir="$PWD/"$2
output_dir="$PWD/"$3
main_path="$PWD"
mkdir -p "sim_$i"
mkdir -p $output_dir
cp alma.cycle9.3.cfg sim_$i/
cd "sim_$i"
ls -l
echo "$PWD"
echo $input_dir
echo $output_dir
conda run -n casa6.5 python "$main_path/alma_simulator.py" $i $input_dir $output_dir
cd ".."
rm -r "sim_$i"