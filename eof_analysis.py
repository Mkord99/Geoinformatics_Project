import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap


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

    # Reshape values to dimensions of the grid
    interpols = np.reshape(z_data, (dim[1], dim[0]))
    z = np.ma.masked_invalid(interpols)

    dic = {
        "long": lons_grid,
        "lat": lats_grid,
        "z": z,
        "dim": dim,
        "space": space
        
    }

    return dic



directory_path = '/home/mo/Desktop/Geoinformatics_Project/Data/dataNorthSea/rasterNorthsea'

z_dict = {}
mask_dict = {}

target_rows = 47
target_cols = 59


for filename in os.listdir(directory_path):
    if filename.endswith('.nc'):  
        file_path = os.path.join(directory_path, filename)
        
        # Reading nc files
        file_data = ncfile_matrix(file_path)
        file_key = os.path.splitext(filename)[0]        
        original_z = file_data['z']
        lats = file_data['lat']
        longs = file_data['long']
        
        # Reshapin all matrixes in same dimension and masking nan values
        reshaped_z = np.full((target_rows, target_cols), np.nan)
        original_rows, original_cols = original_z.shape
        reshaped_z[:original_rows, :original_cols] = original_z
        z_values = np.ma.masked_invalid(reshaped_z)
        z_dict[file_key] = z_values
        
        # Creating mask: 1 for data cells, 0 for NaN cells
        mask = np.where(np.isnan(reshaped_z), 0, 1)
        mask_dict[file_key] = mask

# Creating supermask, 1 for 276 month data avaiablity and 0 if some or all months are missing
masks_sum = np.sum(list(mask_dict.values()), axis=0)
super_mask = np.where(masks_sum == 276, 1, 0)

# Creating ID matrix for sea cells
id_matrix = np.full_like(super_mask, '', dtype=object)
for i in range(target_rows):
    for j in range(target_cols):
        if super_mask[i, j] == 1:
            row_id = f"{i + 1:02}"  
            col_id = f"{j + 1:02}" 
            id_matrix[i, j] = row_id + col_id

# Flattening only sea cells
fl_id_matrix = id_matrix[id_matrix != ''].reshape(1, -1)
sea_cells_dict = {}

for file_key, matrix in z_dict.items():
    sea_cells = matrix[super_mask == 1].compressed()
    sea_cells_dict[file_key] = sea_cells.reshape(1, -1)
    
# Demeaning data for covariance matrix calculation
centured_dict = {}
for file_key, sea_cells in sea_cells_dict.items():
    mean_value = sea_cells.mean()
    centured = sea_cells - mean_value
    centured_dict[file_key] = centured



F = np.vstack(list(centured_dict.values()))
N = F.shape[0]
cov_mat = np.dot(F.T, F) / (N - 1)
eigenvalues, eigenvectors = np.linalg.eig(cov_mat)


# Selecting EOFs and calculating Principal Components (PCs) and calculating the corresponding surface to each PC () 


selected_eofs = {}
pc_dict = {}
crsp_surfaces = {}

#n is the number of EOF that user want to select
n = 2
for i in range(n):  
    eof_column = eigenvectors[:, i].reshape(-1, 1)  
    selected_eofs[f'EOF{i+1}'] = eof_column

    pc_value = np.dot(F, eof_column)
    pc_dict[f'PC{i+1}'] = pc_value  
 
    crsp = np.dot(pc_value, eof_column.T)
    crsp_surfaces[f'contribution_{i+1}'] = crsp
    
 
# Reshaping EOFs into their original shape (47, 59)     
reshaped_eofs = {}

for eof_key, eof_values in selected_eofs.items():
    reshaped_eof = np.full((target_rows, target_cols), np.nan)
    eof_values_flat = eof_values.flatten()

    # Rearrenging the matrixes based on id matrix
    for idx, cell_id in enumerate(fl_id_matrix.flatten()):
        if cell_id != '':
            
            row = int(cell_id[:2]) - 1  
            col = int(cell_id[2:]) - 1
            reshaped_eof[row, col] = eof_values_flat[idx]
    reshaped_eofs[eof_key] = reshaped_eof
    

# Plotting the PCs
time_steps = pd.date_range(start="1993-01", end="2016-01", freq="ME")
for i, (key, pc_values) in enumerate(pc_dict.items(), start=1):
    plt.figure()
    plt.plot(time_steps, pc_values, label=f"PC {i}", color="blue", linewidth=1)
    plt.scatter(time_steps, pc_values, color="red", s=10, label="Month")
    years = pd.date_range(start="1993", end="2016", freq="YS")
    plt.xticks(ticks=years, labels=years.year, rotation=45)
    plt.title(f"PC{i}")
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("EOF Amplitude", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10)
    plt.show()


# Plotting EOFs    
lat_range = lats[:, 1]  
lon_range = longs[1, :]
lat_min, lat_max = np.min(lat_range) - 1, np.max(lat_range) + 1
lon_min, lon_max = np.min(lon_range) - 6, np.max(lon_range) + 6

m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max, llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='i')
for eof_key, reshaped_eof in reshaped_eofs.items():
    masked_eof = np.ma.masked_invalid(reshaped_eof)
    clevs = np.linspace(masked_eof.min(), masked_eof.max(), 21)
    plt.figure()

    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)  
    x, y = m(lon_grid, lat_grid)
    cs = m.contourf(x, y, masked_eof, levels=clevs, cmap="RdBu_r", extend="both")

    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(lat_min, lat_max, 5), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(lon_min, lon_max, 10), labels=[0, 0, 0, 1])
    
    cb = m.colorbar(cs, location='right', pad="5%")
    cb.set_label(f"{eof_key}")
    
    plt.title(f"{eof_key}", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()