import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon, Point
from numpy.linalg import lstsq
from pyproj import Proj, Transformer
from eofs.standard import Eof


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

sorted_ncfiles = sorted(
    [f for f in os.listdir(directory_path) if f.endswith('.nc')],
    key=lambda x: int(os.path.splitext(x)[0]))

z_dict = {}
mask_dict = {}

target_rows = 47
target_cols = 59


for filename in sorted_ncfiles:
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
mask_lost = np.where((2 < masks_sum) & (masks_sum  < 276), 1, 0)

# Creating ID matrix for sea cells
id_matrix = np.full_like(super_mask, '', dtype=object)
for i in range(target_rows):
    for j in range(target_cols):
        if super_mask[i, j] == 1:
            row_id = f"{i + 1:02}"  
            col_id = f"{j + 1:02}" 
            id_matrix[i, j] = row_id + col_id
            
id_coordinates = []
for i in range(target_rows):
    for j in range(target_cols):
        if id_matrix[i, j] != '':
            lat = lats[i, j]
            lon = longs[i, j]
            cell_id = id_matrix[i, j]
            id_coordinates.append({'id': cell_id, 'lat': lat, 'lon': lon})

id_coordinates_df = pd.DataFrame(id_coordinates)


# Flattening only sea cells
fl_id_matrix = id_matrix[id_matrix != ''].reshape(1, -1)
ext_id_matrix = np.tile(fl_id_matrix, (276, 1))
sea_cells_dict = {}

for file_key, matrix in z_dict.items():
    sea_cells = matrix[super_mask == 1].compressed()
    sea_cells_dict[file_key] = sea_cells.reshape(1, -1)
 
 
# Demeaning data for covariance matrix calculation
centered_dict = {}
mean_dict = {}
for file_key, sea_cells in sea_cells_dict.items():
    
    mean_value = sea_cells.mean()
    
    mean_dict[file_key] = mean_value
  
    centered = sea_cells - mean_value

    centered_dict[file_key] = centered
    


F = np.vstack(list(centered_dict.values()))

solver = Eof(F)
eofs = solver.eofs(neofs = 10)  
pcs = solver.pcs(npcs = 10)     


# Selecting EOFs and calculating Principal Components (PCs) and calculating the corresponding surface to each PC () 
crsp_surfaces = {}
org_crsp_surfs = {}
selected_eofs = {}
pc_dict = {}



for i in range(eofs.shape[0]):  
    eof_column = eofs[i, :].reshape(-1, 1)  
    pc_value = pcs [:, i].reshape(pcs.shape[0], 1)
    selected_eofs[f'EOF{i+1}'] = eof_column
    pc_dict[f'PC{i+1}'] = pc_value

    crsp = np.dot(pc_value, eof_column.T)
    org_crsp_surfs[f'crsp_surf_{i+1}'] = crsp
    crsp_surfaces[f'crsp_surf_{i+1}'] = {}

    file_keys = list(mask_dict.keys())
    for idx, file_key in enumerate(file_keys):
        row_matrix = crsp[idx, :].reshape(-1, 1)  
        crsp_surfaces[f'crsp_surf_{i+1}'][file_key] = row_matrix 

fl_id_values = fl_id_matrix.flatten().reshape(-1, 1) 

for i in range(eofs.shape[0]):  
    crsp_surface = crsp_surfaces[f'crsp_surf_{i+1}']

    for file_key, row_matrix in crsp_surface.items():
        df = pd.DataFrame(row_matrix, columns=[f'crsp_{file_key}'])
        fl_id_values_flat = fl_id_values.flatten()
        
        if len(fl_id_values_flat) == len(df):
         
            df['height'] = df[f'crsp_{file_key}']  
            df['pix_id'] = fl_id_values_flat 
            
            df = df[['height', 'pix_id']]  
            crsp_surfaces[f'crsp_surf_{i+1}'][file_key] = df

for i in range(eofs.shape[0]):
    globals()[f'crsp_surf_{i+1}'] = {}  

    for file_key, df in crsp_surfaces[f'crsp_surf_{i+1}'].items():
        globals()[f'crsp_surf_{i+1}'][file_key] = df

      
     
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
    

# %% Tide Gauges data

directory_path = '/home/mo/Desktop/Geoinformatics_Project/Data/dataNorthSea/stationsPerYear/tgDAC_seldemeanalref_proc'

sorted_csvfiles = sorted([f for f in os.listdir(directory_path) if f.endswith('.csv')], key=lambda x: int(os.path.splitext(x)[0]))

org_tide_data = {}

for filename in sorted_csvfiles:
    if filename.endswith('.csv'):  
        file_path = os.path.join(directory_path, filename)
        file_key = os.path.splitext(filename)[0]
        csv_file = pd.read_csv (file_path)
        org_tide_data [file_key] = csv_file

# LCC Projection
all_latitudes = []
all_longitudes = []

for key, df in org_tide_data.items():
    all_latitudes.extend(df['lat'])
    all_longitudes.extend(df['lon'])

lat1 = min(all_latitudes)  
lat2 = max(all_latitudes)  
lon0 = (min(all_longitudes) + max(all_longitudes)) / 2 
lat0 = (lat1 + lat2) / 2  

lcc_proj = Proj(
    proj='lcc',
    lat_1=lat1,  
    lat_2=lat2,  
    lat_0=lat0,  
    lon_0=lon0,  
    x_0=0,  
    y_0=0,  
    ellps='WGS84')

transformer = Transformer.from_proj("epsg:4326", lcc_proj, always_xy=True)


for key, df in org_tide_data.items():
    df['x'], df['y'] = transformer.transform(df['lon'].values, df['lat'].values)
    org_tide_data[key] = df
    
id_coordinates_df['x'], id_coordinates_df['y'] = transformer.transform(id_coordinates_df['lon'].values, id_coordinates_df['lat'].values)
        
    
roi_vertices = [ (-1.1759598042, 60.7411160031), (6.6149725916, 62.9700725976), (12.8227208643, 59.1162691398), (8.9064232963, 56.4498537744),
    (9.1147369967, 53.2209914178), (1.1988163806, 50.2837682418), (-4.8006181916, 57.2206144659), (-1.1759598042, 60.7411160031)]

roi_polygon = Polygon(roi_vertices)

roi_stations = {}

# checking if the stations are inside the specefied regoion of ineterset
for file_key, tide_data in org_tide_data.items():
    
    inside_roi = tide_data [ tide_data.apply(lambda row: roi_polygon.contains(Point(row['lon'], row['lat'])), axis=1) ]
    roi_stations[file_key] = inside_roi
    

# assigning each station to a pixel and assigning the pixel id based on id matrix
def find_nearest_pixel_id(station_x, station_y, id_coordinates_df):
    
    distances = np.sqrt((id_coordinates_df['y'] - station_y)**2 + (id_coordinates_df['x'] - station_x)**2)
    nearest_pix = distances.idxmin()
    return id_coordinates_df.loc[nearest_pix, 'id']



for file_key, stations in roi_stations.items():
    
    stations = stations.copy()
    stations['pix_id'] = stations.apply(lambda row: find_nearest_pixel_id(row['x'], row['y'], id_coordinates_df), axis=1)
    roi_stations[file_key] = stations

# %% Least Square solver

# Initialize variables to track the optimal results
min_squared_error = float('inf')
optimal_eofs = 0
optimal_results = {}

# Iterate over groups of EOFs (1 EOF, then 1+2, then 1+2+3, etc.)
for n_eofs in range(1, 11):  # n_eofs ranges from 1 to 10
    print(f"Processing with the first {n_eofs} EOF(s)...")
    
    # Build the design matrix for the first `n_eofs`
    design_matrix = []
    obs_vector = []

    for file_key, stations in roi_stations.items():
        for index, station in stations.iterrows():
            pix_id = station['pix_id']
            observed_height = station['height'] / 10

            crsp_values = []
            # Use only the first `n_eofs` CRSP surfaces
            for i in range(n_eofs): 
                crsp_surface = globals()[f'crsp_surf_{i+1}']
                crsp_df = crsp_surface[file_key]
                crsp_value = crsp_df.loc[crsp_df['pix_id'] == pix_id, 'height'].values[0]
                crsp_values.append(crsp_value)

            design_matrix.append(crsp_values)
            obs_vector.append(observed_height)

    A = np.array(design_matrix)
    b = np.array(obs_vector)
    coefficients, _, _, _ = lstsq(A, b, rcond=None)

    # Reconstruct the original matrix using the first `n_eofs`
    reconstruction = np.zeros_like(F)
    for i in range(n_eofs):
        reconstruction += coefficients[i] * org_crsp_surfs[f'crsp_surf_{i+1}']

    # Calculate errors
    squared_error = np.sum((F - reconstruction) ** 2)
    rmse_all = np.sqrt(np.mean((reconstruction - F) ** 2))
    mae_all = np.mean(np.abs(reconstruction - F))
    var_diff = np.var(F - reconstruction)
    EVS = 1 - (var_diff / np.var(F))
    
    print(f"  Squared Error: {squared_error}")
    print(f"  RMSE: {rmse_all}")
    print(f"  MAE: {mae_all}")
    print(f"  EVS: {EVS}")
    
    # Check if this is the minimum error
    if squared_error < min_squared_error:
        min_squared_error = squared_error
        optimal_eofs = n_eofs
        optimal_results = {
            "coefficients": coefficients,
            "reconstruction": reconstruction,
            "squared_error": squared_error,
            "rmse": rmse_all,
            "mae": mae_all,
            "evs": EVS,
        }

# After looping, print the optimal results
print("\nOptimal Results:")
print(f"  Optimal number of EOFs: {optimal_eofs}")
print(f"  Minimum Squared Error: {min_squared_error}")
print(f"  RMSE: {optimal_results['rmse']}")
print(f"  MAE: {optimal_results['mae']}")
print(f"  EVS: {optimal_results['evs']}")


optimal_coefficients = optimal_results["coefficients"]
optimal_reconstruction = optimal_results["reconstruction"]
np.save("optimal_reconstruction.npy", optimal_reconstruction)


diff = optimal_reconstruction - F
# %% plotting section

def plot_on_basemap(matrix, lats, longs, plot_label):
    
    #Plots the given matrix on a Basemap projection using default parameters.

    #Parameters:
    #- matrix: 2D numpy array (reshaped matrix) to plot
    #- lats: 2D numpy array representing latitude values
    #- longs: 2D numpy array representing longitude values
    #- plot_label: Label to display on the colorbar (default: "Matrix")
    
    # Default values for lat_range, lon_range, cmap, and contour_levels
    lat_range = lats[:, 1]  
    lon_range = longs[1, :]
    cmap = "RdBu_r"
    contour_levels = 21
    
    # Calculate the bounding latitudes and longitudes
    lat_min, lat_max = np.min(lat_range) - 1, np.max(lat_range) + 1
    lon_min, lon_max = np.min(lon_range) - 6, np.max(lon_range) + 6
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max, llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='i')
    masked_matrix = np.ma.masked_invalid(matrix)
    clevs = np.linspace(masked_matrix.min(), masked_matrix.max(), contour_levels)
    plt.figure()
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    x, y = m(lon_grid, lat_grid)
    cs = m.contourf(x, y, masked_matrix, levels=clevs, cmap=cmap, extend="both")
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(lat_min, lat_max, 5), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(lon_min, lon_max, 10), labels=[0, 0, 0, 1])
    cb = m.colorbar(cs, location='right', pad="5%")
    #cb.set_label("cm")
    plt.title(plot_label, fontsize=10)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

"""
# Plotting EOFs
for i in range(optimal_eofs):  
    eof_key = f"EOF{i+1}"
    reshaped_eof = reshaped_eofs[eof_key]
    plot_label = f"{eof_key}"
    plot_on_basemap(reshaped_eof, lats, longs, plot_label)

    
# Plotting the PCs
time_steps = pd.date_range(start="1993-01", end="2016-01", freq="ME")
for i in range(optimal_eofs):  
    pc_values = pc_dict[f'PC{i+1}'] 
    plt.figure(figsize=(25, 6))
    plt.plot(time_steps, pc_values, label=f"PC {i+1}", color="blue", linewidth=1)
    plt.scatter(time_steps, pc_values, color="red", s=10, label="Month")
    years = pd.date_range(start="1993", end="2016", freq="YS")
    plt.xticks(ticks=years, labels=years.year, rotation=45)
    plt.title(f"PC{i+1}")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("EOF Amplitude", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10)
    plt.show()



# plottig RMSE for each month
rmse_values = np.sqrt(np.mean((optimal_reconstruction - F) ** 2, axis=1))
mae_values = np.mean(np.abs(optimal_reconstruction - F), axis=1)

plt.figure(figsize=(25, 6))
plt.plot(range(1, 277), rmse_values, marker='.', linestyle='-', color='b', linewidth=0.7, label="RMSE")
plt.plot(range(1, 277), mae_values, marker='.', linestyle='-', color='r', linewidth=0.7, label="MAE")
plt.title("RMSE and MAE Values by Month (Optimal Reconstruction)") 
plt.xlabel("Month")
plt.ylabel("Values (cm)")
plt.grid(True)
plt.legend()
plt.xticks(ticks=range(1, 277, 12))  # Ticks every 12 months (to represent years)
plt.show()

"""

# Input the row number to reshape (1 to 276)
selected_row_number = 258

if 1 <= selected_row_number <= 276:
    selected_row_index = selected_row_number - 1
    reconstruction_row = optimal_reconstruction[selected_row_index]
    F_row = F[selected_row_index]
    difference_row = reconstruction_row - F_row
    sq_difference_row = (reconstruction_row - F_row)**2
    
    reshaped_reconstruction = np.full((target_rows, target_cols), np.nan)
    reshaped_F = np.full((target_rows, target_cols), np.nan)
    reshaped_difference = np.full((target_rows, target_cols), np.nan)
    reshaped_sq_difference = np.full((target_rows, target_cols), np.nan) 
    
    reconstruction_row_flat = reconstruction_row.flatten()
    F_row_flat = F_row.flatten()
    difference_row_flat = difference_row.flatten()
    sq_difference_row_flat = sq_difference_row.flatten()
    
    for idx, cell_id in enumerate(fl_id_matrix.flatten()):
        if cell_id != '':
            # Extract the row and column indices from the ID matrix
            row = int(cell_id[:2]) - 1
            col = int(cell_id[2:]) - 1
            reshaped_reconstruction[row, col] = reconstruction_row_flat[idx]
            reshaped_F[row, col] = F_row_flat[idx]
            reshaped_difference[row, col] = difference_row_flat[idx]
            reshaped_sq_difference[row, col] = sq_difference_row_flat[idx]
    
    
else:
    print("Invalid row number. Please enter a number between 1 and 276.")
    
error_variances = []

for epoch in range(F.shape[0]):
    error = optimal_reconstruction[epoch, :] - F[epoch, :]
    var_error = np.var(error, ddof=1) 
    error_variances.append(var_error)

# Average variance of reconstruction errors across all epochs
average_variance = np.mean(error_variances)

print("Average Variance of Reconstruction Errors:", average_variance)

plot_on_basemap(reshaped_reconstruction, lats, longs, plot_label="Reconstruction Surafce of the Selected Month [cm]")
plot_on_basemap(reshaped_F, lats, longs, plot_label="Original Surface of the Selected Month [cm]")
plot_on_basemap(reshaped_difference, lats, longs, plot_label="Misfits surface of the Selected Month [cm]")
plot_on_basemap(reshaped_sq_difference, lats, longs, plot_label="Squared misfits surface of the Selected Month [cm2]")




