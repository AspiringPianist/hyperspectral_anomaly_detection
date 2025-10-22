import scipy.io
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# --- 1. Load the .mat file ---
# Replace 'hjg' with the actual path to your .mat file
dataset = 'abu_airport/abu-airport-1.mat'
ground_truth = 'abu_airport/abu-airport-1.mat'

try:
    mat_data = scipy.io.loadmat(dataset)
    print(f"Successfully loaded: {dataset}")
except FileNotFoundError:
    print(f"Error: File not found at {dataset}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

try:
    gt_data = scipy.io.loadmat(ground_truth)
    print(f"Successfully loaded: {ground_truth}")
except FileNotFoundError:
    print(f"Error: File not found at {ground_truth}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# --- 2. View the variable names inside the file ---
# .mat files are loaded as Python dictionaries. The keys are the variable names.
print("\nVariables stored in the .mat files 1:")
print(list(mat_data.keys()))
print("\nVariables stored in the .mat files 2:")
print(list(gt_data.keys()))

print("\nVariables stored in the .mat files 1, data:")
print(list(mat_data['data']))
print("\nVariables stored in the .mat files 2, map:")
print(list(gt_data['map']))

# --- 3. Access and inspect specific variables ---
# Common variable names for Salinas: 'salinas_corrected' or 'salinas' for the data
# and 'salinas_gt' for the ground truth.
# Replace these with the actual variable names printed above if they differ.

data_var_name = 'data' # Or 'salinas', check the keys output
gt_var_name = 'map'

if data_var_name in mat_data:
    hyperspectral_data = mat_data[data_var_name]
    print(f"\nShape of hyperspectral data ('{data_var_name}'): {hyperspectral_data.shape}")
    # Shape is typically (rows, columns, bands)
else:
    print(f"\nWarning: Variable '{data_var_name}' not found in the .mat file.")
    hyperspectral_data = None

if gt_var_name in gt_data:
    ground_truth = gt_data[gt_var_name]
    print(f"Shape of ground truth ('{gt_var_name}'): {ground_truth.shape}")
    # Shape is typically (rows, columns)
else:
    print(f"\nWarning: Variable '{gt_var_name}' not found in the .mat file.")
    ground_truth = None

# --- 4. (Optional) Visualize the Ground Truth and a Sample Band ---
if ground_truth is not None:
    plt.figure(figsize=(8, 6))
    plt.imshow(ground_truth, cmap='jet') # 'jet' or 'nipy_spectral' are common colormaps
    plt.title(f'Ground Truth Map ({gt_var_name})')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.colorbar(label='Class Label')
    plt.show()

if hyperspectral_data is not None:
    # Check data dimensions
    if not (hyperspectral_data.ndim == 3 and hyperspectral_data.shape[2] > 0):
        print("\nCannot display bands. Data is not a 3D cube or has no bands.")
        exit()

    num_bands = hyperspectral_data.shape[2]
    initial_band = 0 # Start with the first band

    # Create the figure and axes for the image
    # Adjust bottom margin to make space for the slider
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.25)

    # Display the initial band
    img_display = ax.imshow(hyperspectral_data[:, :, initial_band], cmap='gray')
    ax.set_title(f'Band {initial_band}')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    fig.colorbar(img_display, ax=ax, label='Reflectance/Radiance (scaled)')

    # --- Create the Slider Axes ---
    # Position: [left, bottom, width, height]
    ax_slider = plt.axes([0.20, 0.1, 0.65, 0.03])

    # --- Create the Slider ---
    band_slider = Slider(
        ax=ax_slider,
        label='Band Index',
        valmin=0, # Minimum band index
        valmax=num_bands - 1, # Maximum band index
        valinit=initial_band, # Initial band
        valstep=1, # Step size (integer bands)
        valfmt='%d' # Format label as integer
    )

    # --- Define the function to update the plot ---
    def update(val):
        band_index = int(band_slider.val) # Get integer band index from slider
        img_display.set_data(hyperspectral_data[:, :, band_index]) # Update image data
        ax.set_title(f'Band {band_index}') # Update title
        fig.canvas.draw_idle() # Redraw the figure

    # --- Register the update function with the slider ---
    band_slider.on_changed(update)

    # --- Show the plot ---
    plt.show()