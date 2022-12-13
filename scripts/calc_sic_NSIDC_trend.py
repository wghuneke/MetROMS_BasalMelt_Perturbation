"""
Calculates the linear trend for each point of sea ice concentration from the NSIDC observations.
Script is based on a code from Stephy Libera.
Trend is calculated using the scipy.signal.detrend() option.
The script uses NSIDC data, the version here is from Stephy (combined years).

Author:      Wilma Huneke
Date:        Jan 2020
Last modif.: Feb 2020 (stop at year 2015 to exclude year 2016 as it was extreme low melt rates - bias on trend) 
             Dec 2021 calculate trend from 1992-2011 to match model years
                      -> comment/uncomment 3 lines below "optional" and adjust Nyear
"""

import xarray as xr
import numpy as np
import scipy.stats
import scipy.signal
from matplotlib.pyplot import *

def calc_sic_trend():
    
    # Path
    file_path = '/g/data/gh9/wgh581/NSIDC_monthly/sic_combined.nc'
    save_path = '/g/data/gh9/wgh581/NSIDC_monthly/sic_trend_1979_2015.nc'

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

    # Optional: calculate trend for 1992-2012 only
    save_path = '/g/data/gh9/wgh581/NSIDC_monthly/sic_trend_1992_2011.nc'
    sic = sic.where(sic['time.year']>1991, drop=True)
    sic = sic.where(sic['time.year']<2012, drop=True)

    # Add coordinate for months
    months = sic['time.month']
    sic    = sic.assign_coords(month=('time', months.values))
    # (order of months is 1, 2, ..., 11, 12, 1, 2, ...)

    # Sort by month
    sic = sic.sortby('month')
    #print(sic.time)

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
    print('Calculating the trend')
    trend = np.zeros((12, 332, 316)) 
    for nn in range(12):
        tmp = sic_tmp.isel(nM=nn)
        tmp_d = scipy.signal.detrend(tmp, axis=0)
        detrn = tmp - tmp_d
        trend[nn,:,:] = (detrn[-1,:,:] - detrn[0,:,:]) / np.size(detrn, axis=0)

    # Assign array
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    trend = xr.DataArray(trend, \
                         dims = ('month', 'ygrid', 'xgrid'), \
                         coords = {'month':months, 'ygrid':sic.ygrid, 'xgrid':sic.xgrid}, \
                         name = 'sic')

    # Saving data
    print('Writing data to nc file')
    trend.to_netcdf(save_path)


# Command-line interface
if __name__ == '__main__':

    # plot_flag = input("Save figure (1) or display in window (0)? ")
    #plot_flag = 0

    calc_sic_trend()

