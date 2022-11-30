#!/bin/zsh

code=${PWD}/June_20_fixed_forcing_Q0_10_transient_Umax_correct_(diff_params).py

### May_16_fixed_forcing_Q0_10_transient_Umax_correct.py
### May_24_fixed_forcing_Q0_10_transient_Umax_correct_tanh_increase_in_U.py
### Feb_26_fixed_forcing_Q0_10_transient_Umax.py

task(){
   python $code 
}

task

