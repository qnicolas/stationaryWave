import xarray as xr
import numpy as np

# Find project root
import sys
import os.path as op
ROOT = op.abspath(op.join(__file__, op.pardir, op.pardir, op.pardir))
DATA_PATH = op.join(ROOT, 'data') + '/'
sys.path.insert(0, op.join(ROOT, 'src'))

from stationary_wave import StationaryWaveProblem

if __name__ == "__main__":
    case_name = "ideal_Gill_linear"
    sph_resolution = 32
    sigma_full = np.linspace(0,1,21)#(1-np.cos(np.linspace(0,1,41)*np.pi))/2 # 40-level with higher resolution near boundaries
    linear = True
    zonal_basic_state = True
    Nsigma = len(sigma_full) - 1

    held = StationaryWaveProblem(
        sph_resolution,
        sigma_full,
        linear,
        zonal_basic_state,
        DATA_PATH + "output/",
        case_name,
        hyperdiffusion_coefficient=1e17,
        rayleigh_damping_timescale=2,
        newtonian_cooling_timescale=2,
    )

    basicstate = xr.open_dataset(DATA_PATH + "input/ideal_Gill_basic_state.nc")
    held.initialize_basic_state_from_pressure_data(basicstate)

    forcings = xr.open_dataset(DATA_PATH + "input/ideal_Gill_forcings.nc")
    held.initialize_forcings_from_pressure_data(forcings)

    held.integrate(restart=False, use_CFL=False, timestep=1800, stop_sim_time=30 * 86400)
