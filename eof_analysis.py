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
for file_key, sea_cells in sea_cells_dict.items():
    mean_value = sea_cells.mean()
    centered = sea_cells - mean_value
    centered_dict[file_key] = centered

F = np.vstack(list(centered_dict.values()))

solver = Eof(F)
eofs = solver.eofs(neofs = 3)  
pcs = solver.pcs(npcs = 3)     


# Selecting EOFs and calculating Principal Components (PCs) and calculating the corresponding surface to each PC () 
crsp_surfaces = {}
org_crsp_surfs = {}
selected_eofs = {}
pc_dict = {}

#n is the number of EOF that user want to select

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

"""        
# Shows us eash eof represent what persent of all EOFs   
eof_percent = {}
for i in range (eofs.shape[0]):
    cntrb = (eigenvalues[i] / (np.sum(eigenvalues))) * 100
    eof_percent[f'eof{i+1}_percent'] = cntrb
"""    
     
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
design_matrix = []
obs_vector = []

for file_key, stations in roi_stations.items():
    for index, station in stations.iterrows():
        pix_id = station['pix_id']
        observed_height = station['height']  

        
        crsp_values = []
        for i in range(eofs.shape[0]): 
            crsp_surface = globals()[f'crsp_surf_{i+1}']
            crsp_df = crsp_surface[file_key]
            crsp_value = crsp_df.loc[crsp_df['pix_id'] == pix_id, 'height'].values[0]
            crsp_values.append(crsp_value)

        design_matrix.append(crsp_values)
        obs_vector.append(observed_height)

A = np.array(design_matrix)
b = np.array(obs_vector)
coefficients, residuals, rank, singular_values = lstsq(A, b, rcond= None)


reconstruction = np.zeros_like(next(iter(org_crsp_surfs.values())))
for i in range(eofs.shape[0]):
        reconstruction += coefficients[i] * org_crsp_surfs[f'crsp_surf_{i+1}']


# %% plotting section

# Plotting the PCs
time_steps = pd.date_range(start="1993-01", end="2016-01", freq="ME")
for i, (key, pc_values) in enumerate(pc_dict.items(), start=1):
    plt.figure()
    plt.plot(time_steps, pc_values, label=f"PC {i}", color="blue", linewidth=1)
    plt.scatter(time_steps, pc_values, color="red", s=10, label="Month")
    years = pd.date_range(start="1993", end="2016", freq="YS")
    plt.xticks(ticks=years, labels=years.year, rotation=45)
    plt.title(f"PC{i}")
    plt.xlabel("Epochs", fontsize=12)
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





















