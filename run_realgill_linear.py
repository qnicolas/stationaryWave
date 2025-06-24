import numpy as np
import xarray as xr
from stationarywave import StationaryWaveProblem

sph_resolution = 32; Nsigma = 12; linear = True; zonal_basic_state = True
output_dir = "/net/helium/atmosdyn/qnicolas/stationarywave_snapshots/"; case_name = "realgill"

realgill_linear = StationaryWaveProblem(sph_resolution, Nsigma, linear, zonal_basic_state, output_dir, case_name)

basicstate = xr.open_dataset("era5_basicstate.nc").rename(latitude='lat',level='pressure').transpose('pressure','lat')
realgill_linear.initialize_basic_state_from_pressure_data(basicstate)

# Define forcing
Q0 = 2.  # 2 K/day heating rate
sigma_half = (np.arange(Nsigma) + 0.5) / Nsigma
lat = np.linspace(-90,90,91)
lon = np.linspace(-180,180,181)
xr_structure = xr.DataArray(np.zeros((len(sigma_half), len(lat), len(lon))),
                            coords={'sigma_half': sigma_half, 'lat': lat, 'lon': lon},
                            dims=['sigma_half', 'lat', 'lon'])
deltalat = 10
deltalon = 35
Qdiab = Q0 * np.sin(np.pi * xr_structure.sigma_half) * np.pi/2 \
        * np.cos(np.pi * xr_structure.lat / (2 * deltalat))**2 * (np.abs(xr_structure.lat)<deltalat) \
        * np.cos(np.pi * xr_structure.lon / (2 * deltalon))**2 * (np.abs(xr_structure.lon)<deltalon)
Zsfc = 0 * xr_structure.isel(sigma_half=0)
input_forcings = xr.merge([Qdiab.rename('QDIAB'), Zsfc.rename('ZSFC')])

realgill_linear.initialize_forcings_from_sigma_data(input_forcings)
realgill_linear.integrate(restart=False, use_CFL=False, timestep=400, stop_sim_time=20*86400)