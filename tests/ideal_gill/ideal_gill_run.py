import numpy as np
import xarray as xr
from stationarywave import StationaryWaveProblem

case_name = "ideal_Gill_nozm"
sph_resolution = 32
linear = True
zonal_basic_state = True
# dsigma = np.array([0.,0.03,0.041,0.06,0.079,0.094,0.102,0.108,0.109,0.105,0.097,0.082,0.057,0.029,0.007])
# sigma_full = np.cumsum(dsigma)
sigma_full = [0.00000,0.0200000,0.0300000,0.0450000,0.0600000,
              0.0800000,0.0900000,0.110000,0.130000,0.150000,
              0.170000,0.210000,0.245000,0.290000,0.340000,
              0.400000,0.470000,0.550000,0.650000,0.745000,
              0.830000,0.900000,0.955000,0.985000,1.00000,
              ]
Nsigma = len(sigma_full) - 1
output_dir = "data/"


held = StationaryWaveProblem(
    sph_resolution,
    sigma_full,
    linear,
    zonal_basic_state,
    output_dir,
    case_name,
    hyperdiffusion_coefficient=1e17,
    rayleigh_damping_timescale=2,
    newtonian_cooling_timescale=2,
)

basicstate = xr.open_dataset("ideal_Gill_basic_state.nc")
basicstate["T"][:] = np.maximum(basicstate["T"].data, 150)
held.initialize_basic_state_from_pressure_data(basicstate)

forcings = xr.open_dataset("ideal_Gill_forcings.nc")
forcings["QDIAB"] = forcings["QDIAB"] - forcings["QDIAB"].mean("lon")
held.initialize_forcings_from_pressure_data(forcings)

held.integrate(restart=False, use_CFL=False, timestep=1800, stop_sim_time=30 * 86400)
