# Description
Stationary wave model inspired from Ting & Yu (JAS, 1998), coded in python using [Dedalus](https://github.com/DedalusProject). The model equations are explained in detail in `equations.pdf`. The main code is in `src/stationary_wave/model.py`, and contains a class that allows to interpolate basic states, set up the grid and problem equations, and integrate the equations. The code is MPI-parallelized, and all problem parameters, including horizontal and vertical resolution, basic state, forcings, damping rates, etc. can easily be modified.

# Example uses
One idealized case (Matsuno-Gill problem) and one real-data case (from Held et al, 2002) are contained in `examples`.
Each contains a run script for the stationary wave model, which can be run in parallel using `mpiexec -n {number of cores} python run_script.py`. Output from these test cases is analyzed in jupyter notebooks contained in each example folder. 

The idealized Gill test case also has a script solving a shallow-water version of the code. It is much simpler than the full 3D code, and I recommend playing around with this shallow-water version to get acquainted with Dedalus.

Some rough utilities to open output and calculate streamfunctions, Helmholtz decompositions, and diagnose pressure velocities, are included in `src/stationary_wave/tools.py`. Use of these utilities is illustrated in the example Jupyter Notebooks.

# Environment
A .yml file is included that contains all necessary python packages to run the code. Create a conda environment using conda env create -f dedalus3_environment.yml, then activate with conda activate dedalus3 and you are hopefully all set!

# Contact
For any questions or suggestions, contact qnicolas --at-- berkeley --dot-- edu
