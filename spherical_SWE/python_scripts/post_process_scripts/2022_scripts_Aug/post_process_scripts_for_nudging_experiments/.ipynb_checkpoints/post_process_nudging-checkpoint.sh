#!/bin/sh

code1=./Aug_19_velocity_decomposition.py
code2=./Aug_19_post_process_budget.py

task(){
   python $code1
   python $code2
}

task
