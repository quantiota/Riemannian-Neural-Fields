import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import numpy as np
import os
os.makedirs("/home/coder/density_xd_results", exist_ok=True)
# Try to import Simplex noise, with fallback to a simpler random implementation

from noise import snoise3

def generate_3d_density_field(grid_size=(15, 15, 15), base_density=100000, 
                             density_variation=0.5, scale=0.1, 
                             octaves=4, persistence=0.5, lacunarity=2.0):
    """
    Generate a 3D neuron density field using Simplex noise.

    Args:
        grid_size: Tuple (nx, ny, nz) defining the 3D grid dimensions
        base_density: Base neuron density (neurons per cubic mm)
        density_variation: Amplitude of density variation around base (0-1)
        scale: Physical size of each grid cell in mm
        octaves: Number of octaves for Simplex noise
        persistence: Persistence for Simplex noise
        lacunarity: Lacunarity for Simplex noise

    Returns:
        torch.Tensor: 3D tensor of neuron counts per grid cell
    """
    nx, ny, nz = grid_size
    density = torch.zeros(grid_size)
    
    # Create spatial coordinates
    x = torch.linspace(0, nx * scale, nx)
    y = torch.linspace(0, ny * scale, ny)
    z = torch.linspace(0, nz * scale, nz)
    
    # Generate density field
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                noise_val = snoise3(x[i].item(), y[j].item(), z[k].item(),
                                   octaves=octaves,
                                   persistence=persistence,
                                   lacunarity=lacunarity)
                density[i, j, k] = base_density * (1 + density_variation * noise_val)
    
    # Calculate volume of each grid cell (in cubic mm)
    cell_volume = scale ** 3
    
    # Convert density to neurons per cell
    neurons_per_cell = density * cell_volume
    
    return neurons_per_cell

def visualize_3d_density(neurons_per_cell, grid_size, scale, base_density=100000):
    """
    Visualize the 3D neuron density field with a 2D slice, histogram, and 3D scatter plot.

    Args:
        neurons_per_cell: 3D tensor of neuron counts
        grid_size: Tuple (nx, ny, nz) of grid dimensions
        scale: Physical size of each grid cell in mm
        base_density: Base neuron density (neurons per cubic mm) for statistics display
    """
    nx, ny, nz = grid_size
    volume_mm3 = nx * ny * nz * (scale ** 3)
    

    
    # 2D Slice Visualization with Histogram (using the provided template)
    slice_z = nz // 2
    plt.figure(figsize=(12, 10))
    
    # Main heatmap
    ax = plt.subplot(111)
    im = ax.imshow(neurons_per_cell[:, :, slice_z].detach().cpu().numpy(), cmap='plasma', 
                   interpolation='bilinear',
                   norm=Normalize(vmin=neurons_per_cell.detach().cpu().numpy().min(), 
                                  vmax=neurons_per_cell.detach().cpu().numpy().max()))
    ax.set_xlabel('X position (grid points)', fontsize=12)
    ax.set_ylabel('Y position (grid points)', fontsize=12, labelpad=20)
    ax.yaxis.set_label_position("left")
    
    # Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Neurons per grid cell', fontsize=12)
    cbar.formatter = ticker.FuncFormatter(lambda x, pos: f"{int(x)}")
    cbar.update_ticks()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set title with grid index
    ax.set_title(f'Neuron Density at z = {slice_z} (3D)', fontsize=14, pad=15, y=1.02)
    
    # Statistics text
    min_neurons = torch.min(neurons_per_cell).item()
    max_neurons = torch.max(neurons_per_cell).item()
    avg_neurons = torch.mean(neurons_per_cell).item()
    total_neurons = torch.sum(neurons_per_cell).item()
    
    stats_text = (
        f"Grid size: {nx}×{ny}×{nz} cells\n"
        f"Physical volume: {volume_mm3:.1f} mm³\n"
        f"Cell size: {scale}×{scale}×{scale} mm\n"
        f"Base density: {base_density} neurons/mm³\n"
        f"Neurons per grid cell:\n"
        f"  Min: {min_neurons:.1f}\n"
        f"  Max: {max_neurons:.1f}\n"
        f"  Average: {avg_neurons:.1f}\n"
        f"Total neurons: {total_neurons/1e6:.2f} million"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.25, 0.88, stats_text, fontsize=12, 
            bbox=props, 
            verticalalignment='top',
            horizontalalignment='left')
    
    # Add a histogram of neuron distribution
    ax2 = divider.append_axes("bottom", size="20%", pad=0.6)
    hist_data = neurons_per_cell.flatten().detach().cpu().numpy()
    ax2.hist(hist_data, bins=50, color='navy', alpha=0.7)
    ax2.set_xlabel('Neurons per grid cell', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('density_xd_results/neuron_density_3D.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3D Scatter Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid coordinates
    x, y, z = torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing='ij')
    x = x.flatten().detach().cpu().numpy()
    y = y.flatten().detach().cpu().numpy()
    z = z.flatten().detach().cpu().numpy()
    density_values = neurons_per_cell.flatten().detach().cpu().numpy()
    
    # Plot scatter with density_values directly as the color
    scatter = ax.scatter(x, y, z, c=density_values, cmap='plasma', s=50, alpha=0.6)
    
    # Add colorbar with actual neuron counts
    cbar = plt.colorbar(scatter, ax=ax, label='Neurons per cell', pad=0.1)
    
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Z position')
    ax.set_title('3D Neuron Density Field')
    
    plt.tight_layout()
    plt.savefig('density_xd_results/density_3d_scatter.png')
    plt.close()

def main():
    # Parameters (matching original script)
    grid_size = (15, 15, 15)
    base_density = 100000
    density_variation = 0.5
    scale = 0.1
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0
    
    # Generate density field
    neurons_per_cell = generate_3d_density_field(
        grid_size=grid_size,
        base_density=base_density,
        density_variation=density_variation,
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity
    )
    
    # Visualize the density field
    visualize_3d_density(neurons_per_cell, grid_size, scale, base_density=base_density)
    print("Density field generated and visualizations saved in 'density_xd_results' directory.")

if __name__ == "__main__":
    main()