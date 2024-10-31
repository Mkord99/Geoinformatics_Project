# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:39:20 2024

@author: moham
"""


import netCDF4 as nc
import numpy as np


def ncfile_matrix(file_path):
    

    data = nc.Dataset(file_path, 'r')
    z_data = data.variables['z'][:] 
    lon_range = data.variables['x_range'][:]  
    lat_range = data.variables['y_range'][:]
    dim = data.variables['dimension'][:]
    space = data.variables['spacing'][:]

    lons = np.linspace(lon_range[0]+space[0]/2, lon_range[1]-space[0]/2, num=dim[0])
    lats = np.linspace(lat_range[1]-space[1]/2, lat_range[0]+space[1]/2, num=dim[1])
    lons_grid, lats_grid = np.meshgrid(lons, lats)

 # reshape values to dim of grid
    interpols = np.reshape(z_data, (dim[1], dim[0]))
    z  = np.ma.masked_invalid(interpols);

    dic = {
        "x": lons_grid,
        "y": lats_grid,
        "z": z}

    return dic