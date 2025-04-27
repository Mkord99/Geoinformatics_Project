# EOFtoolkit

A comprehensive toolkit for EOF (Empirical Orthogonal Function) analysis of NetCDF data.

## Overview

EOFtoolkit is a Python library designed to process geographic/spatial-temporal data stored in NetCDF (.nc) files, perform SVD analysis to extract Empirical Orthogonal Functions (EOFs) and Principal Components (PCs), create optimal data reconstructions, and visualize results on maps.

The library provides tools for:

- Reading and processing chronologically ordered NetCDF files
- Standardizing dimensions across multiple files
- Creating binary masks and super masks to identify cells with valid data
- Generating ID matrices for spatial reference
- Performing SVD analysis to extract EOFs and PCs
- Reconstructing data using various numbers of modes
- Visualizing results on geographic maps

## Installation

You can install EOFtoolkit using pip:

```bash
pip install eoftoolkit
```

Or install from source:

```bash
git clone https://github.com/yourusername/eoftoolkit.git
cd eoftoolkit
pip install -e .
```

## Dependencies

- **Core dependencies**: numpy, scipy, xarray, netCDF4, matplotlib, basemap, pyproj, pandas
- **Optional dependencies**: dask, tqdm

## Quick Start

```python
import eoftoolkit as eof
import matplotlib.pyplot as plt

# Initialize the processor
processor = eof.EOFProcessor()

# Process a directory of NetCDF files
result = processor.process_directory(
    directory_path="/path/to/nc/files",
    file_extension='.nc'
)

# Perform SVD analysis with 10 modes
svd_results = processor.perform_svd(num_modes=10)

# Reconstruct data
recon_results = processor.reconstruct(max_modes=10)

# Visualize the first EOF
processor.visualize_eof(mode_number=1)
plt.show()

# Visualize the first PC
processor.visualize_pc(mode_number=1)
plt.show()

# Compare original and reconstructed data
processor.visualize_comparison(timestamp_index=0)
plt.show()
```

## Usage Examples

See the `examples` directory for more comprehensive examples:

- `basic_usage.py`: Basic example of EOF analysis on NetCDF data
- (more examples to be added)

## Processing Pipeline

The library follows this processing pipeline:

1. **File Handling**: Read and sort NetCDF files chronologically
2. **Dimension Standardization**: Ensure all matrices have consistent dimensions
3. **Masking**: Create binary masks and super mask to identify valid data cells
4. **ID Matrix Generation**: Create spatial reference system
5. **Matrix Flattening**: Convert 2D spatial matrices to 1D vectors
6. **Super Matrix Creation**: Stack all flattened matrices to create a time-space matrix
7. **SVD Analysis**: Extract EOFs and PCs from the super matrix
8. **Data Reconstruction**: Generate reconstructions using various numbers of modes
9. **Visualization**: Plot results on maps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.