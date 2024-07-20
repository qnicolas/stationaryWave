# Description
Stationary wave model inspired from Ting & Yu (JAS, 1998), coded in python using [Dedalus](https://github.com/DedalusProject). The model equations are explained in detail in `equations.md`. The main code is in `stationarywave.py`, and can be run in parallel using `mpiexec -n {number of cores} python stationarywave.py 0`.
It runs an idealized test case with a background state at rest, with a uniform lapse rate, and prescribed mid-tropospheric heating at the equator (i.e., a Matsuno-Gill type problem). The output is analyzed in `twoLevel.ipynb` (for a 2-level version) and `twelveLevel.ipynb` (for a 12-level version). 

The basic state, heating and topography can easily be modified, e.g. to use real data. One example of this is in `stationarywave_realgill.py`, and is analyzed in `twelveLevelGillReal.ipynb`.

A simpler, shallow-water stationary wave model is included in `stationarywave_SW.py` and analyzed un `shallowWater.ipynb`. I recommend starting with this one to get acquainted with the framework. 

Some rough utilities to open output and calculate streamfunctions / Helmholtz decompositions are included in `mydedalustools.py`.

# Planned improvements
The code is still quite rough, and, time-permitting, I will try and make it more accessible. Any suggestions are appreciated. Plans include:
 - Adding nonlinear terms
 - Making the code more modular

# Environment
A .yml file is included that contains all necessary python packages to run the code. Create a conda environment using conda env create -f dedalus3_environment.yml, then activate with conda activate dedalus3, launch a Jupyter notebook and you are hopefully all set!

# Contact
For any questions, contact qnicolas --at-- berkeley --dot-- edu
