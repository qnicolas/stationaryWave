# Description
Stationary wave model inspired from Ting & Yu (JAS, 1998), coded in python using [Dedalus](https://github.com/DedalusProject). The model equations are explained in detail in `equations.pdf`. The main code is in `stationarywave.py`, and contains a class that allows to interpolate basic states, set up the grid and problem equations, and integrate the equations. The code is MPI-parallelized, and all problem parameters, including horizontal and vertical resolution, basic state, forcings, damping rates, etc. can easily be modified.

# Example uses
Two idealized cases (Topographic stationary wave and Matsuno-Gill problem) are contained in `run_idealgill_linear.py`. They can be run in parallel using `mpiexec -n {number of cores} python run_idealgill_linear.py`.
The output from these idealized test cases is analyzed in `twoLevel.ipynb` (for 2-level idealized cases) and `twelveLevel.ipynb` (for a 12-level version of the idealized Matsuno-Gill problem).

The basic state, heating and topography can easily be modified, e.g. to use real data. One example of this is in `run_realgill_linear.py` (Matsuno-Gill problem with a realistic basic state), and is analyzed in `twelveLevelGillReal.ipynb`.

A simpler, shallow-water stationary wave model is included in `stationarywave_SW.py` and analyzed in `shallowWater.ipynb`. I recommend starting with this one to get acquainted with the framework. 

Some rough utilities to open output and calculate streamfunctions, Helmholtz decompositions, and diagnose pressure velocities, are included in `mydedalustools.py`.

# Environment
A .yml file is included that contains all necessary python packages to run the code. Create a conda environment using conda env create -f dedalus3_environment.yml, then activate with conda activate dedalus3, launch a Jupyter notebook and you are hopefully all set!

# Contact
For any questions or suggestions, contact qnicolas --at-- berkeley --dot-- edu
