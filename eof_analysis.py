import os
import netCDF4 as nc
from nc_reader import ncfile_matrix
import numpy as np

# Directory path for the files
directory_path = '/home/mo/Desktop/Geoinformatics_Project/Data/dataNorthSea/rasterNorthsea'

# Initialize dictionaries
z_dict = {}
mask_dict = {}

target_rows = 47
target_cols = 59

# Process each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.nc'):  
        file_path = os.path.join(directory_path, filename)
        
        # Reading the file
        file_data = ncfile_matrix(file_path)
        
        # Key based on file name without extension
        file_key = os.path.splitext(filename)[0]  
        
        original_z = file_data['z']
        
        # Make the reshaped matrix with NaN for empty cells
        reshaped_z = np.full((target_rows, target_cols), np.nan)

        # Copy original data to the reshaped matrix
        original_rows, original_cols = original_z.shape
        reshaped_z[:original_rows, :original_cols] = original_z

        # Mask NaN values in the reshaped matrix
        z_values = np.ma.masked_invalid(reshaped_z)
        z_dict[file_key] = z_values
        
        # Create mask: 1 for data cells, 0 for NaN cells
        mask = np.where(np.isnan(reshaped_z), 0, 1)
        mask_dict[file_key] = mask

# Sum of all masks to check each cell's data availability
masks_sum = np.sum(list(mask_dict.values()), axis=0)
super_mask = np.where(masks_sum == 276, 1, 0)

# Create an ID matrix for sea cells
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
    
# Demeaning the sea cells for covariance calculation
demean_sea_dict = {}
for file_key, sea_cells in sea_cells_dict.items():
    mean_value = sea_cells.mean()
    demean_sea = sea_cells - mean_value
    demean_sea_dict[file_key] = demean_sea

# Construct the matrix F
F = np.vstack(list(demean_sea_dict.values()))
N = F.shape[1]
cov_mat = np.dot(F.T, F) / (N - 1)
eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

# Selecting EOFs based on eigenvalues > 100
selected_eofs = {}
pc_dict = {}
contributions = {}

for i, eigenvalue in enumerate(eigenvalues):
    if eigenvalue > 100:
        eof_column = eigenvectors[:, i].reshape(-1, 1)  # Reshape EOF for matrix compatibility
        selected_eofs[f'eof_{i+1}'] = eof_column
        
        # Calculate the principal component
        pc_value = np.dot(F, eof_column).reshape(-1, 1)
        pc_dict[f'pc_{i+1}'] = pc_value  
        
        # Calculate contribution matrix
        ctb = np.dot(pc_value, eof_column.T)
        contributions[f'contribution_{i+1}'] = ctb

# Sum all contribution matrices to create the reconstructed sea matrix
reconstructed_sea = sum(contributions.values())








