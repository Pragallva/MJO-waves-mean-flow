#!/bin/zsh

code=${PWD}/May16_realistic_wind_no_heat_source.py

## May17_DJF_mean_wind_jet.py
## May17_asymmetric_wind_no_heat_source.py
## May16_asymmetric_wind_stationary_heat_source.py

task(){
   python $code 
}

task

