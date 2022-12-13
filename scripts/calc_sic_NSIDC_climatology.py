"""
Calculates the monthly climatology for each point of sea ice concentration from the NSIDC observations. The script uses NSIDC data, the version here is from Stephy (combined years).

Author:      Wilma Huneke
Date:        Feb 2020
Last modif.: Dec 2021 calculate climatology for 1992-2011 only to match model years
                    -> use lines under "optional" and adjust Nyear
"""

import xarray as xr
import numpy as np
import scipy.stats
import scipy.signal
from matplotlib.pyplot import *

def calc_sic_climatology():
    
    # Path
    file_path = '/g/data/gh9/wgh581/NSIDC_monthly/sic_combined.nc'
    save_path = '/g/data/gh9/wgh581/NSIDC_monthly/sic_climatology_1979_2015.nc'

    # Load data
    data = xr.open_dataset(file_path)
    sic  = data.goddard_merged_seaice_conc_monthly
    # Remove Nov, Dec 1978 for reshaping later (else dim error)
    sic  = sic[2:,...]

    # Drop years that have nans. (The detrend function can't handle nans.)
    sic = sic.where(sic['time.year']!=1987, drop=True)
    sic = sic.where(sic['time.year']!=1988, drop=True)
    # Also drop year 2016 (and 2017)  as it had minimum sea ice extent and biases the trend
    sic = sic.where(sic['time.year']!=2016, drop=True)
    sic = sic.where(sic['time.year']!=2017, drop=True)
    
    # Optional: uncomment the following 3 lines when calculating climatology for 1992-2011:
    save_path = '/g/data/gh9/wgh581/NSIDC_monthly/sic_climatology_1992_2011.nc'
    sic = sic.where(sic['time.year']>1991, drop=True)
    sic = sic.where(sic['time.year']<2012, drop=True)

    # Add coordinate for months
    months = sic['time.month']
    sic    = sic.assign_coords(month=('time', months.values))
    # (order of months is 1, 2, ..., 11, 12, 1, 2, ...)

    # Sort by month
    sic = sic.sortby('month')

    # Prepare data array
    # Number of years
    Nyear = 20 # 37: if including 2016, 35: for < 2016, 20: 1992-2011
    # Assign array
    sic_tmp = xr.DataArray(np.reshape(sic.values, (12,Nyear,332,316)),\
                           dims = ('nM','time','ygrid','xgrid'),\
                           coords = {'nM':list(set(sic.month.values)),\
                                     'nY':sic.time.dt.year[0:Nyear],\
                                     'ygrid':sic.ygrid,'xgrid':sic.xgrid})
    # Calculate the trend
    print('Calculating the climatology')
    climatology = np.zeros((12, 332, 316)) 
    for nn in range(12):
        tmp = sic_tmp.isel(nM=nn)
        climatology[nn,:,:] = tmp.mean(axis=0)

    # Assign array
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    climatology = xr.DataArray(climatology, \
                         dims = ('month', 'ygrid', 'xgrid'), \
                         coords = {'month':months, 'ygrid':sic.ygrid, 'xgrid':sic.xgrid}, \
                         name = 'sic')

    # Saving data
    print('Writing data to nc file')
    climatology.to_netcdf(save_path)


# Command-line interface
if __name__ == '__main__':

    # plot_flag = input("Save figure (1) or display in window (0)? ")
    #plot_flag = 0

    calc_sic_climatology()

