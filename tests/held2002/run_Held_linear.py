import numpy as np
import xarray as xr
import sys
sys.path.append('../../')
from model.stationarywave import StationaryWaveProblem

# import sys
# case = int(sys.argv[1])
# assert case == 1 or case == 2
# if case==1:
#     case_name = "held2002_idealtibet_linear"
# elif case==2:
#     case_name = "held2002_smoothorog_linear"

case_name = "held2002_onlyheating_linear_dampedzonalmean_variablesigma_nozonalmeanforcing"
sph_resolution = 32; linear = True; zonal_basic_state = True
dsigma = np.array([0.,0.03,0.041,0.06,0.079,0.094,0.102,0.108,0.109,0.105,0.097,0.082,0.057,0.029,0.007])
sigma_full = np.cumsum(dsigma)
Nsigma = len(sigma_full) - 1
output_dir = "/Users/qnicolas/stationaryWave/data/"#"/net/helium/atmosdyn/qnicolas/stationarywave_snapshots/"

# #/!\/!\/!\/!\/!\/!\ for 24 levels
# rayleigh_damping_timescale = np.ones(Nsigma) * 25
# rayleigh_damping_timescale[-1] = 0.5
# rayleigh_damping_timescale[-2] = 1.
# rayleigh_damping_timescale[-3] = 6.
# rayleigh_damping_timescale[-4] = 10.

# 14 levels
rayleigh_damping_timescale = np.ones(Nsigma) * 25
rayleigh_damping_timescale[-1] = 0.3
rayleigh_damping_timescale[-2] = 0.5
rayleigh_damping_timescale[-3] = 1.0
rayleigh_damping_timescale[-4] = 8.

held = StationaryWaveProblem(sph_resolution, sigma_full, linear, zonal_basic_state, output_dir, case_name,
                             hyperdiffusion_coefficient=1e17,
                             rayleigh_damping_timescale=rayleigh_damping_timescale,
                             newtonian_cooling_timescale=15)

basicstate = xr.open_dataset("/Users/qnicolas/stationaryWave/data/inputdata/ncep_jan_basic_state.nc")
held.initialize_basic_state_from_pressure_data(basicstate)

forcings = xr.open_dataset("/Users/qnicolas/stationaryWave/data/inputdata/ncep_jan_forcings.nc")
for var in 'ZSFC','EHFD', 'EMFD_U','EMFD_V': #'QDIAB'
    forcings[var][:] = 0.
for var in 'QDIAB',:
    forcings[var] = forcings[var] - forcings[var].mean('lon')
# costrunc = lambda x,x0,sig : np.cos((x-x0)/sig * np.pi/2) * (np.abs(x-x0) < sig)
# forcings['ZSFC'] = 40 * costrunc(forcings.lat,35,10) * costrunc(forcings.lon,90,20)

#     forcings['EMFD_U'][-2:] = 0.
#     forcings['EMFD_V'][-2:] = 0.
held.initialize_forcings_from_pressure_data(forcings)

held.integrate(restart=False, use_CFL=False, timestep=1800, stop_sim_time= 30 * 86400)