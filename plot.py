import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid blocking

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
        scale = 'log'
        if len(metadata_line) >= 8 and metadata_line[6] == 'scale':
            scale = metadata_line[7]
        
        # Read disorder information (new format: tD value tp dist UD value Up dist uD value up dist)
        disorder_line = file.readline().strip().split()
        disorder_info = {}
        if len(disorder_line) > 1:
            i = 1
            while i < len(disorder_line):
                key = disorder_line[i]
                if key == 'none':
                    disorder_info = {'none': True}
                    break
                elif key in ['tD', 'UD', 'uD']:
                    # Disorder strength
                    if i + 1 < len(disorder_line):
                        disorder_info[key] = float(disorder_line[i + 1])
                        i += 2
                    else:
                        i += 1
                elif key in ['tp', 'Up', 'up']:
                    # Distribution type
                    if i + 1 < len(disorder_line):
                        disorder_info[key] = disorder_line[i + 1]
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
        
        data = np.loadtxt(file)
    return fixed_param, fixed_value, m, n, R, scale, disorder_info, data

def get_parameter_labels(fixed_param):
    if fixed_param == "t":
        return 'U', '$\\mu$', 'U', '$\\mu$'
    elif fixed_param == "U":
        return 't', '$\\mu$', 't', '$\\mu$'
    elif fixed_param == "u":
        return 't', 'U', 't', 'U'
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
    plt.close()

def plot_eigenvalues(ratios, eigenvalues, header, output_dir, x_label):
    plt.figure(figsize=(12, 8))
    for i in range(min(10, eigenvalues.shape[1])):
        valid_mask = ~np.isnan(eigenvalues[:, i])
        if np.any(valid_mask):
            plt.plot(ratios[valid_mask], eigenvalues[valid_mask, i], '_', markersize=10, markeredgewidth=2)
    plt.xscale('log')
    plt.yscale('symlog', linthresh=10)
    valid_eigenvalues = eigenvalues[~np.isnan(eigenvalues)]
    if valid_eigenvalues.size > 0:
        y_min = np.min(valid_eigenvalues)
        y_max = np.max(valid_eigenvalues)
        plt.ylim(y_min, y_max)
    if header and header.startswith('# Ratio'):
        xlabel_text = header.replace('# Ratio ', '').split(' Eigenvalues')[0]
    else:
        xlabel_text = 'Ratio'
    plt.xlabel(xlabel_text)
    plt.ylabel('Energy')
    plt.title('First Eigenvalues Evolution')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eigenvalues_evolution.svg'), bbox_inches='tight')
    plt.close()

# Load data
ratios, eigenvalues, header = load_eigenvalues()
fixed_param, fixed_value, m, n, R, scale, disorder_info, data = load_phase_data()
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

# Apply smoothing with NaN handling
sigma = 2

def smooth_with_nan(data, sigma):
    """Apply Gaussian smoothing while properly handling NaN values"""
    # Create a mask for valid (non-NaN) data
    valid_mask = ~np.isnan(data)
    
    # If all values are NaN, return the original data
    if not np.any(valid_mask):
        return data
    
    # Replace NaN with 0 for filtering
    data_filled = np.where(valid_mask, data, 0)
    
    # Apply Gaussian filter to both data and mask
    smoothed_data = gaussian_filter(data_filled, sigma=sigma)
    smoothed_mask = gaussian_filter(valid_mask.astype(float), sigma=sigma)
    
    # Normalize by the smoothed mask to account for missing values
    # Set threshold to avoid division by very small numbers
    result = np.where(smoothed_mask > 0.01, smoothed_data / smoothed_mask, np.nan)
    
    return result

gap_ratio_blurred = smooth_with_nan(gap_ratio_grid, sigma)
condensate_fraction_blurred = smooth_with_nan(condensate_fraction_grid, sigma)
fluctuations_blurred = smooth_with_nan(fluctuations_grid, sigma)
qEA_blurred = smooth_with_nan(qEA_grid, sigma)

# Create output directory
# Build disorder string for folder name using new naming convention
disorder_str = ''
if 'none' not in disorder_info and disorder_info:
    disorder_parts = []
    # Process in order: tD/tp, UD/Up, uD/up
    for param in ['t', 'U', 'u']:
        strength_key = f'{param}D'
        dist_key = f'{param}p'
        if strength_key in disorder_info:
            disorder_parts.append(f'{strength_key}_{disorder_info[strength_key]}')
            if dist_key in disorder_info:
                disorder_parts.append(f'{dist_key}_{disorder_info[dist_key]}')
    if disorder_parts:
        disorder_str = '_' + '_'.join(disorder_parts)

output_dir = f'figures/m_{m}_n_{n}_{fixed_param}_{fixed_value}_{non_fixed_param1}_{param1_min}-{param1_max}_{non_fixed_param2}_{param2_min}-{param2_max}_R_{R}{disorder_str}'
os.makedirs(output_dir, exist_ok=True)

# Copy data files to output directory
if os.path.exists('phase.txt'):
    shutil.copy2('phase.txt', os.path.join(output_dir, 'phase.txt'))
if os.path.exists('eigenvalues_diagonal.txt'):
    shutil.copy2('eigenvalues_diagonal.txt', os.path.join(output_dir, 'eigenvalues_diagonal.txt'))

# Generate phase map plots
plot_eigenvalues(ratios, eigenvalues, header, output_dir, x_label)

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
