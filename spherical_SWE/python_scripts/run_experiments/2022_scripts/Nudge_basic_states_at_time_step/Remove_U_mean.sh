#!/bin/zsh

code=${PWD}/Aug_18_fixed_forcing_Q0_10_transient_Ueq_nudged_by_removing_zonal_mean_U.py

task(){
   python $code 
}

task

