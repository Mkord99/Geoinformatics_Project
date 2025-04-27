"""
Basic usage example for EOFtoolkit.
This script demonstrates how to use EOFtoolkit to:
1. Process a directory of NetCDF files
2. Perform SVD analysis
3. Reconstruct data
4. Visualize results
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import eoftoolkit as eof

# Set the directory containing NetCDF files
data_dir = "/home/mo/Desktop/EOFtoolkit/dataNorthSea/rasterNorthsea"  # Change this to your data directory

# Initialize the processor
processor = eof.EOFProcessor(verbose=True)

# Process the directory of NetCDF files
result = processor.process_directory(
    directory_path=data_dir,
    file_extension='.nc'
)

print(f"\nProcessed {len(processor.file_keys)} files")
print(f"Target dimensions: {processor.target_dims}")

# Configure projection for the North Sea region
# Lambert Conformal Conic is good for mid-latitude regions
lats = processor.latitude
lons = processor.longitude
lat_min = np.nanmin(lats)
lat_max = np.nanmax(lats)
lon_min = np.nanmin(lons)
lon_max = np.nanmax(lons)

processor.configure_projection(
    projection='lcc',  # Lambert Conformal Conic
    projection_params={
        'lat_1': lat_min,  # Lower standard parallel
        'lat_2': lat_max,  # Upper standard parallel
        'lat_0': (lat_min + lat_max) / 2,  # Central latitude
        'lon_0': (lon_min + lon_max) / 2   # Central longitude
    }
)

# Perform SVD analysis with 10 modes
svd_results = processor.perform_svd(num_modes=10, compute_surfaces=True)

# Get the explained variance for each mode
explained_variance = svd_results['explained_variance']
cumulative_variance = svd_results['cumulative_variance']

# Plot the explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='b', label='Individual')
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', label='Cumulative')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('EOF Mode')
plt.ylabel('Variance Explained (%)')
plt.title('Variance Explained by EOF Modes')
plt.legend()
plt.tight_layout()
plt.show()

# Perform reconstruction using up to all available modes
recon_results = processor.reconstruct(max_modes=None, metric='rmse')

optimal_mode_count = recon_results['optimal_mode_count']
print(f"\nOptimal reconstruction uses {optimal_mode_count} modes")

# Visualize the first three EOFs
plt.figure(figsize=(15, 5))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    processor.visualize_eof(mode_number=i, fig=plt.gcf(), ax=plt.gca())
#plt.tight_layout()
plt.show()

# Visualize the first three PCs with proper date formatting
plt.figure(figsize=(15, 12))
for i in range(1, 4):
    plt.subplot(3, 1, i)
    processor.visualize_pc(mode_number=i, date_format='%Y%m', fig=plt.gcf(), ax=plt.gca())
plt.tight_layout()
plt.show()

# Compare original and reconstructed data for the first timestamp
processor.visualize_comparison(timestamp_index=0, title_prefix="First Timestamp")
plt.show()

# Plot reconstruction error metrics
processor.visualize_reconstruction_error(metric='rmse')
plt.show()

# Visualize multiple error metrics at once
from eoftoolkit.visualization.error_plots import plot_multiple_error_metrics
plot_multiple_error_metrics(recon_results['error_metrics'], metrics=['rmse', 'mae', 'r2'])
plt.show()

# Save results
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

saved_files = processor.save_results(output_dir)
print("\nSaved results:")
for key, path in saved_files.items():
    print(f"  {key}: {path}")