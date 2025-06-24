import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from stationarywave import StationaryWaveProblem

sph_resolution = 32; Nsigma = 12; linear = True; zonal_basic_state = True
output_dir = "/net/helium/atmosdyn/qnicolas/stationarywave_snapshots/"; case_name = "idealgill"

idealgill_linear = StationaryWaveProblem(sph_resolution, Nsigma, linear, zonal_basic_state, output_dir, case_name, hyperdiffusion_coefficient=40e15)

###################################################
############ INITIALIZE BASIC STATE ###############
###################################################
sigma_half = (np.arange(Nsigma) + 0.5) / Nsigma
lat = np.linspace(-90,90,91)
lon = np.linspace(-180,180,181)
xr_structure = xr.DataArray(np.ones((len(sigma_half), len(lat), len(lon))),
                            coords={'sigma_half': sigma_half, 'lat': lat, 'lon': lon},
                            dims=['sigma_half', 'lat', 'lon'])

# Basic-state temperature
T0 = 290. # surface temperature in Kelvin
Gamma = 7e-3 # lapse rate in Kelvin/meter
Rd = 287.; g = 9.81; Omega = 2*np.pi/86400
Tbar = T0 * xr_structure.sigma_half ** (Rd*Gamma/g) * xr_structure  # hydrostatic profile with constant lapse rate

# Basic-state V
Vbar = 0. * xr_structure

# Basic-state W
sigma_full = (np.arange(1,Nsigma)) / Nsigma
Wbar = xr.DataArray(np.zeros((len(sigma_full), len(lat), len(lon))),
                    coords={'sigma_full': sigma_full, 'lat': lat, 'lon': lon},
                    dims=['sigma_full', 'lat', 'lon'])

# Basic-state wind and log-surface pressure
# A barotropic basic-state jet can be balanced by gradients in log-surface pressure,
# following geostrophic balance: f ubar = -R T d(lnpsbar)/dy 
# Note that this must be valid at all levels, hence ubar / T must not vary in the vertical.

Ubar = 0. * xr_structure
# # Gaussian jet profile
# lat0 = 45.
# deltalat = 10.
# Ubar = 10. * np.exp(-(xr_structure.lat-lat0)**2/(2*deltalat**2)) * xr_structure

# Correct so that Ubar / T is constant in the vertical
Ubar = Ubar * Tbar / T0

# Calculate balanced log-surface pressure
f = 2 * Omega * np.sin(xr_structure.lat * np.pi / 180)
lnpsbar = cumulative_trapezoid((f * Ubar / (- Rd * Tbar) ).isel(sigma_half = 0),
                               xr_structure.lat * np.pi / 180 * 6.378e6, 
                               initial=0, 
                               axis=0,
                               ) * xr_structure.isel(sigma_half=0) 


input_data = xr.merge([Tbar.rename('T'), 
                       Ubar.rename('U'), 
                       Vbar.rename('V'),
                       Wbar.rename('W'),
                       1e5 * np.exp(lnpsbar).rename('SP')])
if zonal_basic_state:
    input_data = input_data.isel(lon=0)

idealgill_linear.initialize_basic_state_from_sigma_data(input_data)

###################################################
############## INITIALIZE FORCINGS ################
###################################################

# Topographic forcing
Zsfc = 0. * xr_structure.isel(sigma_half=0) 
# # Gaussian topography
# lon0 = 0.
# deltalon = 10.
# H0 = 500.  # peak height in meters
# Zsfc = H0 * np.exp( - (xr_structure.lat-lat0)**2/(2*deltalat**2) - (xr_structure.lon-lon0)**2/(2*deltalon**2)) * xr_structure.isel(sigma_half=0)

# Heating forcing
Q0 = 1. # 1 K/day heating rate
deltalat = 9.
deltalon = 9.
Qdiab = Q0 * np.exp(-(xr_structure.sigma_half-0.5)**2/(2*0.1**2)) * np.exp(-xr_structure.lat**2/(2*deltalat**2)) * np.exp(-xr_structure.lon**2/(2*deltalon**2))
    
idealgill_linear.initialize_forcings_from_sigma_data(xr.merge([Qdiab.rename('QDIAB'), Zsfc.rename('ZSFC')]))


###################################################
###################### RUN ########################
###################################################
idealgill_linear.integrate(restart=False, use_CFL=False, timestep=400, stop_sim_time=5*86400)