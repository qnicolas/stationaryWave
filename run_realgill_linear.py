import xarray as xr
from stationarywave_realgill import StationaryWaveProblem


idealgill_linear = StationaryWaveProblem(resolution=32, Nsigma=12, linear=True, zonal_basic_state=True)
idealgill_linear.setup_bases()
idealgill_linear.setup_problem()
#idealgill_linear.initialize_basic_state_with_zeros()
basicstate = xr.open_dataset("era5_basicstate_interp.nc").rename(latitude='lat').transpose('sigma','lat')
idealgill_linear.initialize_basic_state_from_sigma_data(basicstate)
idealgill_linear.initialize_forcings()
idealgill_linear.integrate(restart=False, use_CFL=False, timestep=400, stop_sim_time=20*86400)