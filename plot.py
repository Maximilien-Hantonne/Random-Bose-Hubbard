import matplotlib
matplotlib.use('TkAgg')

import os
import shutil
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def wrap_title(title, width=30):
    return "\n".join(textwrap.wrap(title, width))

def load_phase_data():
    with open('phase.txt', 'r') as file:
        fixed_param_line = file.readline().strip().split()
        fixed_param = fixed_param_line[0]
        fixed_value = float(fixed_param_line[1])
        metadata_line = file.readline().strip().split()
        m = int(metadata_line[1])
        n = int(metadata_line[3])
        R = int(metadata_line[5])
        # Read scale if present (default to 'log' for backward compatibility)
        scale = 'log'
        if len(metadata_line) >= 8 and metadata_line[6] == 'scale':
            scale = metadata_line[7]
        data = np.loadtxt(file)
    return fixed_param, fixed_value, m, n, R, scale, data

def get_parameter_labels(fixed_param):
    if fixed_param == "T":
        return 'Interaction Strength (U)', 'Chemical Potential ($\\mu$)', 'U', '$\\mu$'
    elif fixed_param == "U":
        return 'Hopping Parameter (t)', 'Chemical Potential ($\\mu$)', 't', '$\\mu$'
    elif fixed_param == "u":
        return 'Hopping Parameter (t)', 'Interaction Strength (U)', 't', 'U'
    else:
        raise ValueError("Invalid fixed parameter in phase.txt")

def load_eigenvalues():
    if not os.path.exists('eigenvalues_diagonal.txt'):
        return None, None, None
    with open('eigenvalues_diagonal.txt', 'r') as f:
        header = f.readline().strip()
        eigen_data = np.loadtxt(f)
    if eigen_data.size == 0:
        return None, None, None
    if eigen_data.ndim == 1:
        eigen_data = eigen_data.reshape(1, -1)
    ratios = eigen_data[:, 0]
    eigenvalues = eigen_data[:, 1:]
    sort_idx = np.argsort(ratios)
    return ratios[sort_idx], eigenvalues[sort_idx], header

def plot_phase_map(x_grid, y_grid, data_grid, label, x_label, y_label, output_dir, filename, title, scale='log', note=None):
    plt.figure(figsize=(10, 6))
    plt.contourf(x_grid, y_grid, data_grid, levels=50, cmap='viridis')
    cbar = plt.colorbar(label=label)
    if label == 'Gap Ratio':
        cbar.ax.axhline(y=0.39, color='red', linestyle='solid', linewidth=3)
        cbar.ax.axhline(y=0.53, color='red', linestyle='solid', linewidth=3)
    if scale == 'log':
        plt.xscale('log')
        plt.yscale('log')
    plt.xlim(x_grid.min(), x_grid.max())
    plt.ylim(y_grid.min(), y_grid.max())
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(wrap_title(title), fontsize=12)
    if note:
        plt.figtext(0.5, 0.01, note, ha='center', fontsize=9, color='red')
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()
    plt.close()

def plot_eigenvalues(ratios, eigenvalues, header, output_dir):
    plt.figure(figsize=(12, 8))
    for i in range(min(10, eigenvalues.shape[1])):
        plt.plot(ratios, eigenvalues[:, i], '_', markersize=10, markeredgewidth=2)
        plt.xscale('log')
        plt.yscale('symlog', linthresh=10)
    y_min = np.min(eigenvalues)
    y_max = np.max(eigenvalues)
    plt.ylim(y_min, y_max)
    xlabel = header.split()[2].replace('T', 't')
    plt.xlabel(xlabel)
    plt.ylabel('Energy')
    plt.title('First Eigenvalues Evolution')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eigenvalues_evolution.svg'), bbox_inches='tight')
    plt.show()
    plt.close()

# Load data
ratios, eigenvalues, header = load_eigenvalues()
fixed_param, fixed_value, m, n, R, scale, data = load_phase_data()
x_label, y_label, non_fixed_param1, non_fixed_param2 = get_parameter_labels(fixed_param)

# Extract data columns
x_values = data[:, 0]
y_values = data[:, 1]
gap_ratio = data[:, 2]
condensate_fraction = data[:, 3]
fluctuations = data[:, 4]
qEA = data[:, 5]

# Normalize by fixed parameter
if abs(fixed_value) > 1e-10:  
    x_values = x_values / fixed_value
    y_values = y_values / fixed_value
    x_label = f'{x_label}/{fixed_param}'
    y_label = f'{y_label}/{fixed_param}'

# Calculate parameter ranges
param1_min = np.min(x_values)
param1_max = np.max(x_values)
param2_min = np.min(y_values)
param2_max = np.max(y_values)

# Create grids
x_unique = np.unique(x_values)
y_unique = np.unique(y_values)
x_grid, y_grid = np.meshgrid(x_unique, y_unique)

# Reshape data to grid (data is stored with param1 outer loop, param2 inner loop)
gap_ratio_grid = gap_ratio.reshape(len(x_unique), len(y_unique)).T
condensate_fraction_grid = condensate_fraction.reshape(len(x_unique), len(y_unique)).T
fluctuations_grid = fluctuations.reshape(len(x_unique), len(y_unique)).T
qEA_grid = qEA.reshape(len(x_unique), len(y_unique)).T

# Apply smoothing
sigma = 2
gap_ratio_blurred = gaussian_filter(gap_ratio_grid, sigma=sigma)
condensate_fraction_blurred = gaussian_filter(condensate_fraction_grid, sigma=sigma)
fluctuations_blurred = gaussian_filter(fluctuations_grid, sigma=sigma)
qEA_blurred = gaussian_filter(qEA_grid, sigma=sigma)

# Create output directory
output_dir = f'figures/m{m}_n{n}_{fixed_param}_{fixed_value}_{non_fixed_param1}_{param1_min}-{param1_max}_{non_fixed_param2}_{param2_min}-{param2_max}_R{R}'
os.makedirs(output_dir, exist_ok=True)

# Copy data files to output directory
if os.path.exists('phase.txt'):
    shutil.copy2('phase.txt', os.path.join(output_dir, 'phase.txt'))
if os.path.exists('eigenvalues_diagonal.txt'):
    shutil.copy2('eigenvalues_diagonal.txt', os.path.join(output_dir, 'eigenvalues_diagonal.txt'))

# Generate phase map plots
plot_eigenvalues(ratios, eigenvalues, header, output_dir)

plot_phase_map(x_grid, y_grid, gap_ratio_blurred, 'Gap Ratio', x_label, y_label, output_dir, 
               'gap_ratio_plot.svg', f'Gap Ratio with respect to {x_label} and {y_label}', scale,
               'Note: 0.39 is for a Poissonnian distribution and 0.53 is for a Gaussian orthogonal ensemble (GOE)')

plot_phase_map(x_grid, y_grid, condensate_fraction_blurred, 'Condensate Fraction', x_label, y_label, output_dir,
               'condensate_fraction_plot.svg', f'Condensate fraction with respect to {x_label} and {y_label}', scale)

plot_phase_map(x_grid, y_grid, fluctuations_blurred, 'Fluctuations in Boson Number', x_label, y_label, output_dir,
               'fluctuations_plot.svg', f'Fluctuations with respect to {x_label} and {y_label}', scale)

plot_phase_map(x_grid, y_grid, qEA_blurred, 'Edwards-Anderson Order Parameter ($q_{EA}$)', x_label, y_label, output_dir,
               'qEA_plot.svg', f'Edwards-Anderson Parameter with respect to {x_label} and {y_label}', scale,
               f'Note: Averaged over {R} disorder realizations')
