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
    # Find case 
    case = int(sys.argv[1])
    case_names = ["held2002_all_linear",
                  "held2002_orog_linear",
                  "held2002_emfd_linear",
                  "held2002_heat_linear",
                  "held2002_all_nonlinear"]
    case_name = case_names[case]
    if case < 4:
        linear=True
        timestep = 1800
    else:
        linear=False
        timestep = 900

    sph_resolution = 32
    # Sigma levels from Held et al 2002
    dsigma = np.array([0.,0.03,0.041,0.06,0.079,0.094,0.102,0.108,0.109,0.105,0.097,0.082,0.057,0.029,0.007])
    sigma_full = np.cumsum(dsigma)
    zonal_basic_state = True

    Nsigma = len(sigma_full) - 1
    # Rayleigh damping profile from Held et al 2002
    rayleigh_damping_timescale = np.ones(Nsigma) * 25
    rayleigh_damping_timescale[-1] = 0.3
    rayleigh_damping_timescale[-2] = 0.5
    rayleigh_damping_timescale[-3] = 1.0
    rayleigh_damping_timescale[-4] = 8.

    held = StationaryWaveProblem(
        sph_resolution,
        sigma_full,
        linear,
        zonal_basic_state,
        DATA_PATH + "output/",
        case_name,
        hyperdiffusion_coefficient=1e17,
        rayleigh_damping_timescale=rayleigh_damping_timescale,
        newtonian_cooling_timescale=15,
    )

    basicstate = xr.open_dataset(DATA_PATH + "input/ncep_jan_basic_state.nc")
    held.initialize_basic_state_from_pressure_data(basicstate)

    forcings = xr.open_dataset(DATA_PATH + "input/ncep_jan_forcings.nc")
    if case == 1: # orography only
        for var in 'QDIAB','EHFD', 'EMFD_U','EMFD_V':
            forcings[var][:] = 0.
    elif case == 2: # EMFD only
        for var in 'ZSFC','QDIAB','EHFD':
            forcings[var][:] = 0.
    elif case == 3: # heating only
        for var in 'ZSFC','EMFD_U','EMFD_V':
            forcings[var][:] = 0.
    held.initialize_forcings_from_pressure_data(forcings)

    held.integrate(restart=False, use_CFL=False, timestep=timestep, stop_sim_time=50 * 86400)
