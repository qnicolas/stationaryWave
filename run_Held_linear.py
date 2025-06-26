import numpy as np
import xarray as xr
from stationarywave import StationaryWaveProblem

# import sys
# case = int(sys.argv[1])
# assert case == 1 or case == 2
# if case==1:
#     case_name = "held2002_idealtibet_linear"
# elif case==2:
#     case_name = "held2002_smoothorog_linear"

case_name = "held2002_idealtibet_linear"
sph_resolution = 32; Nsigma = 12; linear = True; zonal_basic_state = True
output_dir = "/net/helium/atmosdyn/qnicolas/stationarywave_snapshots/"

# #/!\/!\/!\/!\/!\/!\ for 24 levels
# rayleigh_damping_timescale = np.ones(Nsigma) * 25
# rayleigh_damping_timescale[-1] = 0.5
# rayleigh_damping_timescale[-2] = 1.
# rayleigh_damping_timescale[-3] = 6.
# rayleigh_damping_timescale[-4] = 10.

# 12 levels
rayleigh_damping_timescale = np.ones(Nsigma) * 25
rayleigh_damping_timescale[-1] = 0.4
rayleigh_damping_timescale[-2] = 8.

held = StationaryWaveProblem(sph_resolution, Nsigma, linear, zonal_basic_state, output_dir, case_name,
                             hyperdiffusion_coefficient=1e17,
                             rayleigh_damping_timescale=rayleigh_damping_timescale,
                             newtonian_cooling_timescale=15)

basicstate = xr.open_dataset("ncep_jan_basic_state.nc")
held.initialize_basic_state_from_pressure_data(basicstate)

forcings = xr.open_dataset("ncep_jan_forcings.nc")
# forcings = forcings - forcings.mean('lon')
for var in 'QDIAB','EHFD', 'EMFD_U','EMFD_V': #'ZSFC'
    forcings[var][:] = 0.
costrunc = lambda x,x0,sig : np.cos((x-x0)/sig * np.pi/2) * (np.abs(x-x0) < sig)
forcings['ZSFC'] = 4000 * costrunc(forcings.lat,35,10) * costrunc(forcings.lon,90,20)
#     forcings['EMFD_U'][-2:] = 0.
#     forcings['EMFD_V'][-2:] = 0.
held.initialize_forcings_from_pressure_data(forcings)

held.integrate(restart=False, use_CFL=False, timestep=400, stop_sim_time= 30 * 86400)