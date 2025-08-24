import numpy as np
import xarray as xr
from stationarywave import StationaryWaveProblem

case_name = "ideal_topo_nozm"
sph_resolution = 32; linear = True; zonal_basic_state = True
dsigma = np.array([0.,0.03,0.041,0.06,0.079,0.094,0.102,0.108,0.109,0.105,0.097,0.082,0.057,0.029,0.007])
sigma_full = np.cumsum(dsigma)
Nsigma = len(sigma_full) - 1
output_dir = "data/"

held = StationaryWaveProblem(sph_resolution, sigma_full, linear, zonal_basic_state, output_dir, case_name,
                             hyperdiffusion_coefficient=1e17,
                             rayleigh_damping_timescale=2,
                             newtonian_cooling_timescale=2)

basicstate = xr.open_dataset("ideal_Gill_basic_state.nc")
basicstate['T'][:] = np.maximum(basicstate['T'].data,150)
held.initialize_basic_state_from_pressure_data(basicstate)

forcings = xr.open_dataset("ideal_topo_forcings.nc")
forcings['ZSFC'] = forcings['ZSFC'] - forcings['ZSFC'].mean('lon')
held.initialize_forcings_from_pressure_data(forcings)

held.integrate(restart=False, use_CFL=False, timestep=1800, stop_sim_time= 30 * 86400)


