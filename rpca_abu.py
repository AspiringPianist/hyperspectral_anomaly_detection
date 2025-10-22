# Necessary Imports
import numpy as np
import scipy.io
import scipy.linalg # For SVD
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time
import os # Added for directory creation

# --- Create Output Directory ---
output_dir = 'rpca_anomaly_detection_results'
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved in: '{output_dir}/'")

# --- Load Your Data ---
# Replace with the actual paths to your .mat files
dataset_path = 'abu_airport/abu-airport-1.mat'
ground_truth_path = 'abu_airport/abu-airport-1.mat'
dataset_basename = os.path.splitext(os.path.basename(dataset_path))[0] # Get base filename for saving

try:
    mat_data = scipy.io.loadmat(dataset_path)
    print(f"Successfully loaded data: {dataset_path}")
    # --- Access hyperspectral data ---
    data_var_name = 'data' # Adjust if your variable name is different
    if data_var_name in mat_data:
        hyperspectral_data = mat_data[data_var_name].astype(np.float64) # Ensure float type
        H, W, D = hyperspectral_data.shape
        N = H * W
        print(f"Shape of hyperspectral data ('{data_var_name}'): {hyperspectral_data.shape}")
    else:
        print(f"Error: Variable '{data_var_name}' not found in {dataset_path}.")
        exit()
except FileNotFoundError:
    print(f"Error: Data file not found at {dataset_path}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data file: {e}")
    exit()

try:
    gt_data = scipy.io.loadmat(ground_truth_path)
    print(f"Successfully loaded ground truth: {ground_truth_path}")
    # --- Access ground truth data ---
    gt_var_name = 'map' # Adjust if your variable name is different
    if gt_var_name in gt_data:
        ground_truth = gt_data[gt_var_name].astype(int) # Ensure integer type
        print(f"Shape of ground truth ('{gt_var_name}'): {ground_truth.shape}")
        if ground_truth.shape != (H, W):
             print(f"Error: Ground truth shape {ground_truth.shape} mismatch with data shape {(H, W)}")
             exit()
    else:
        print(f"Error: Variable '{gt_var_name}' not found in {ground_truth_path}.")
        exit()
except FileNotFoundError:
    print(f"Error: Ground truth file not found at {ground_truth_path}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the ground truth file: {e}")
    exit()


# --- 1. Data Preparation ---
X = hyperspectral_data.reshape((N, D)) # Reshape to (N_pixels, N_bands)
print(f"Reshaped data matrix X shape: {X.shape}")

# Flatten ground truth for metrics calculation
gt_flat = ground_truth.flatten() # Shape (N,)
n_anomaly_gt = np.sum(gt_flat)
n_background_gt = N - n_anomaly_gt
print(f"Ground Truth: {n_anomaly_gt} anomaly pixels, {n_background_gt} background pixels.")
if n_anomaly_gt == 0 or n_background_gt == 0:
    print("Warning: Ground truth contains only one class. ROC/AUC metrics will be invalid.")


# --- Utility Functions ---

def soft_threshold(A, tau):
    """Applies element-wise soft thresholding."""
    return np.sign(A) * np.maximum(np.abs(A) - tau, 0.)

def svt(A, tau):
    """Applies Singular Value Thresholding."""
    try:
        U, s, Vh = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesvd')
    except (np.linalg.LinAlgError, ValueError): # Catch potential convergence errors or NaN/Inf issues
         print("SVT Warning: SVD(gesvd) did not converge or encountered invalid values. Trying gesdd...")
         try:
            U, s, Vh = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesdd')
         except (np.linalg.LinAlgError, ValueError) as e:
            print(f"SVT Error: SVD(gesdd) also failed: {e}. Returning zeros.")
            return np.zeros_like(A) # Return zeros as a fallback
    s_thresh = soft_threshold(s, tau)
    return U @ np.diag(s_thresh) @ Vh

def hard_threshold(A, k):
    """Keeps the k largest magnitude elements of A, sets others to 0."""
    S = np.zeros_like(A)
    if k > 0 and A.size > 0 : # Ensure k is positive and A is not empty
        # Flatten A to find the kth largest value efficiently
        flat_abs_A = np.abs(A).flatten()
        if k >= len(flat_abs_A): # If k is larger than or equal to total elements
             S = A # Keep everything
        else:
             # Find the threshold value: the k-th largest absolute value
             threshold_val = np.partition(flat_abs_A, -k)[-k]
             # Keep elements where abs(value) >= threshold
             # Handle potential duplicates correctly if threshold_val is 0
             if threshold_val == 0:
                 # If threshold is 0, just keep non-zeros up to k elements
                 non_zero_indices = np.nonzero(A)
                 actual_k = min(k, len(non_zero_indices[0]))
                 if actual_k > 0:
                     abs_vals_nonzero = np.abs(A[non_zero_indices])
                     # Get indices relative to the non-zero elements
                     indices_to_keep_rel = np.argsort(abs_vals_nonzero)[-actual_k:]
                     # Convert relative indices back to original indices
                     original_indices = tuple(idx[indices_to_keep_rel] for idx in non_zero_indices)
                     S[original_indices] = A[original_indices]
             else:
                # Standard case: Keep elements >= threshold
                indices_to_keep = np.where(np.abs(A) >= threshold_val)
                S[indices_to_keep] = A[indices_to_keep]
                # If we kept too many due to duplicate threshold values, remove smallest extras
                num_kept = len(indices_to_keep[0])
                if num_kept > k:
                    # Find the indices of the elements exactly equal to the threshold
                    thresh_indices = np.where(np.abs(A) == threshold_val)
                    num_at_thresh = len(thresh_indices[0])
                    num_over_thresh = num_kept - num_at_thresh
                    num_to_remove = num_kept - k
                    # Get the indices of threshold values, sort them by original value (not abs)
                    thresh_coords = list(zip(*thresh_indices))
                    thresh_vals_orig = [A[coord] for coord in thresh_coords]
                    sorted_thresh_indices_orig = np.argsort(np.abs(thresh_vals_orig)) # Sort by abs value first
                    # Indices to set to zero (smallest absolute values at threshold)
                    indices_to_zero_coords = [thresh_coords[i] for i in sorted_thresh_indices_orig[:num_to_remove]]
                    for coord in indices_to_zero_coords:
                         S[coord] = 0

    return S


# --- 2. Option A: Principal Component Pursuit (PCP) via ALM ---
def pcp_alm(X, lambda_val, mu=None, tol=1e-7, max_iter=500, verbose=True):
    """
    Solves the PCP problem using the Augmented Lagrange Multiplier method.
    min ||L||_* + lambda ||S||_1 subj to L + S = X
    Based on Candès et al. (2011) [cite_start][cite: 1475]
    Returns L, S, elapsed_time
    """
    N, D = X.shape
    norm_X_fro = np.linalg.norm(X, 'fro')
    if norm_X_fro == 0: norm_X_fro = 1.0

    if mu is None:
        abs_X_sum = np.sum(np.abs(X))
        if abs_X_sum == 0: abs_X_sum = 1.0 # Avoid division by zero
        mu = N * D / (4 * abs_X_sum)
        if verbose: print(f"PCP-ALM: Calculated initial mu: {mu:.4f}")
    mu_max = mu * 1e6 # Set a max mu to prevent potential overflow
    rho = 1.6

    L_k = np.zeros_like(X)
    S_k = np.zeros_like(X)
    Y_k = np.zeros_like(X)

    if verbose: print("\nStarting PCP-ALM Algorithm...")
    start_time = time.time()
    iter_count = 0
    for k in range(max_iter):
        iter_count += 1
        # [cite_start]Update L [cite: 2329]
        L_k_plus_1 = svt(X - S_k + (1/mu) * Y_k, 1/mu)
        # [cite_start]Update S [cite: 2325]
        S_k_plus_1 = soft_threshold(X - L_k_plus_1 + (1/mu) * Y_k, lambda_val/mu)
        # Update Y
        Y_k_plus_1 = Y_k + mu * (X - L_k_plus_1 - S_k_plus_1)

        stop_criterion = np.linalg.norm(X - L_k_plus_1 - S_k_plus_1, 'fro') / norm_X_fro

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            print(f"PCP-ALM Iter {k+1}/{max_iter}, Stop Criterion: {stop_criterion:.6f}")

        if stop_criterion < tol:
            if verbose: print(f"PCP-ALM Converged at iteration {k+1}.")
            break

        # Update mu
        mu = min(rho * mu, mu_max)
        L_k, S_k, Y_k = L_k_plus_1, S_k_plus_1, Y_k_plus_1
    else:
        if verbose: print("PCP-ALM Reached maximum iterations.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose: print(f"PCP-ALM finished in {elapsed_time:.2f} seconds.")
    return L_k_plus_1, S_k_plus_1, elapsed_time

# --- 3. Option B: LRaSMD via GoDec ---
def lrasmd_godec(X, rank_r, sparsity_k_ratio, tol=1e-7, max_iter=100, verbose=True):
    """
    Solves the LRaSMD problem using the GoDec algorithm.
    min ||X - B - S||_F^2 subj to rank(B)<=r, card(S)<=k*N*D
    Based on Sun et al. (2014) [cite_start][cite: 2623] which uses GoDec from Zhou & Tao (2011).
    Returns B, S, elapsed_time
    """
    N, D = X.shape
    k_elements = int(sparsity_k_ratio * N * D)
    norm_X_fro = np.linalg.norm(X, 'fro')
    if norm_X_fro == 0: norm_X_fro = 1.0

    if verbose: print(f"\nStarting LRaSMD-GoDec Algorithm...")
    if verbose: print(f"Target rank r = {rank_r}, Target sparsity k = {k_elements} ({sparsity_k_ratio*100:.2f}% elements)")

    B_t = X.copy()
    S_t = np.zeros_like(X)
    prev_error = np.inf
    iter_count = 0

    start_time = time.time()
    for t in range(max_iter):
        iter_count += 1
        # [cite_start]Update B (Keep top r singular values/vectors of X - S) [cite: 2809]
        try:
             U, s, Vh = scipy.linalg.svd(X - S_t, full_matrices=False, lapack_driver='gesvd')
        except (np.linalg.LinAlgError, ValueError):
             print("GoDec SVD Warning: gesvd failed. Trying gesdd...")
             try:
                 U, s, Vh = scipy.linalg.svd(X - S_t, full_matrices=False, lapack_driver='gesdd')
             except (np.linalg.LinAlgError, ValueError) as e:
                 print(f"GoDec SVD Error: gesdd also failed: {e}. Using previous B.")
                 # If SVD fails completely, keep the previous B and continue
                 B_t_plus_1 = B_t # Fallback
                 # Still need to update s for rank calculation if possible
                 s = scipy.linalg.svdvals(X-S_t) # Calculate only singular values if possible
                 if len(s)==0: s=np.array([0]) # Handle empty case

        rank_r_actual = min(rank_r, len(s))
        B_t_plus_1 = U[:, :rank_r_actual] @ np.diag(s[:rank_r_actual]) @ Vh[:rank_r_actual, :]

        # [cite_start]Update S (Keep k largest magnitude elements of X - B) [cite: 2810]
        S_t_plus_1 = hard_threshold(X - B_t_plus_1, k_elements)

        current_error = np.linalg.norm(X - B_t_plus_1 - S_t_plus_1, 'fro')
        # Check if current_error is NaN or Inf
        if not np.isfinite(current_error):
             print(f"GoDec Error: Non-finite error encountered at iteration {t+1}. Stopping.")
             break # Stop iteration if error becomes non-finite

        error_change = abs(prev_error - current_error) / norm_X_fro if np.isfinite(prev_error) else np.inf


        if verbose and (t % 20 == 0 or t == max_iter - 1):
             print(f"GoDec Iter {t+1}/{max_iter}, Error Change: {error_change:.6f}, Current Error Norm: {current_error:.4f}")

        if error_change < tol and t > 0:
            if verbose: print(f"GoDec Converged at iteration {t+1}.")
            break

        B_t, S_t = B_t_plus_1, S_t_plus_1
        prev_error = current_error
    else:
         if verbose: print("GoDec Reached maximum iterations.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose: print(f"LRaSMD-GoDec finished in {elapsed_time:.2f} seconds.")
    # Ensure final matrices are returned even if loop breaks due to error
    return B_t, S_t, elapsed_time

# --- 4. Anomaly Score Calculation ---
def calculate_anomaly_scores(S, H, W):
    """
    Calculates anomaly score for each pixel based on the sparse matrix S.
    Uses Euclidean distance from the mean row vector.
    Based on Sun et al. (2014)[cite_start], Equation 8[cite: 2823].
    """
    N, D = S.shape
    if N == 0 or D == 0: return np.zeros((H,W)) # Handle empty S
    S_mean = np.mean(S, axis=0)
    scores = np.linalg.norm(S - S_mean, axis=1) # Efficient Euclidean distance
    return scores.reshape((H, W))

# --- 5. Metrics Calculation Functions ---

def calculate_ser(scores_normalized_flat, gt_flat):
    """
    Calculates the Square Error Ratio (SER).
    [cite_start]Based on Hyperspectral Anomaly Detection survey, Eq 12[cite: 590].
    Assumes scores_normalized_flat are scaled between 0 and 1.
    """
    if len(scores_normalized_flat) != len(gt_flat):
        print("SER Error: Score and GT lengths differ.")
        return np.nan

    anomaly_indices = np.where(gt_flat == 1)[0]
    background_indices = np.where(gt_flat == 0)[0]

    anomaly_error = np.sum((scores_normalized_flat[anomaly_indices] - 1)**2) if len(anomaly_indices) > 0 else 0
    bck_error = np.sum((scores_normalized_flat[background_indices] - 0)**2) if len(background_indices) > 0 else 0

    # Ensure errors are finite
    if not (np.isfinite(anomaly_error) and np.isfinite(bck_error)):
        print("SER Warning: Non-finite error components.")
        return np.nan

    ser = (anomaly_error + bck_error) / len(gt_flat) * 100 # Original paper multiplies by 100 [cite: 596]
    return ser

def calculate_aer(scores_flat, gt_flat):
    """
    Calculates the Area Error Ratio (AER).
    [cite_start]Based on Hyperspectral Anomaly Detection survey, Eq 13[cite: 652].
    Requires calculation of areas under Pd vs Threshold and Pfa vs Threshold.
    """
    thresholds = np.unique(scores_flat) # Get unique score values as potential thresholds
    thresholds = np.append(thresholds, thresholds.max()+1) # Add a threshold above max
    thresholds = np.insert(thresholds, 0, thresholds.min()-1) # Add a threshold below min
    thresholds = np.sort(thresholds)[::-1] # Sort decreasing

    num_thresholds = len(thresholds)
    pd = np.zeros(num_thresholds)
    pfa = np.zeros(num_thresholds)

    n_anomaly = np.sum(gt_flat == 1)
    n_background = len(gt_flat) - n_anomaly

    if n_anomaly == 0 or n_background == 0:
        print("Warning: Cannot calculate AER with zero anomalies or zero background pixels.")
        return np.nan

    for i, thresh in enumerate(thresholds):
        detected_as_anomaly = scores_flat >= thresh
        true_positives = np.sum((detected_as_anomaly == 1) & (gt_flat == 1))
        false_positives = np.sum((detected_as_anomaly == 1) & (gt_flat == 0))

        pd[i] = true_positives / n_anomaly
        pfa[i] = false_positives / n_background

    # Normalize thresholds for area calculation (0 to 1 range)
    min_score, max_score = np.min(scores_flat), np.max(scores_flat)
    if max_score == min_score:
        norm_thresholds = np.array([1.0, 0.0]) # Effectively two points
        # Use PFA/PD corresponding to detecting all (if score > -inf) or none
        if np.isfinite(scores_flat[0]):
            idx = np.where(thresholds >= scores_flat[0])[0][-1] # Index for the score itself
            pd = np.array([pd[idx], pd[idx]])
            pfa = np.array([pfa[idx], pfa[idx]])
        else:
            pd = np.array([0., 0.])
            pfa = np.array([0., 0.])
    else:
        norm_thresholds = (thresholds - min_score) / (max_score - min_score)

    # Calculate area using trapezoidal rule - AUC expects increasing x
    # We sort by normalized threshold (ascending)
    sort_idx = np.argsort(norm_thresholds)
    norm_thresholds_sorted = norm_thresholds[sort_idx]
    pd_sorted = pd[sort_idx]
    pfa_sorted = pfa[sort_idx]

    ap_d = auc(norm_thresholds_sorted, pd_sorted)
    ap_fa = auc(norm_thresholds_sorted, pfa_sorted)

    if np.isnan(ap_d) or np.isnan(ap_fa): aer = np.nan
    else: aer = ap_d - ap_fa # Formula AP_D - AP_FA [cite: 1464]

    return aer


def normalize_scores(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        # Avoid division by zero, return 0.5 as neutral? or based on min_val?
        return np.ones_like(scores) * 0.5 if min_val != 0 else np.zeros_like(scores)
    # Check for NaN/Inf before normalization
    if not (np.isfinite(min_val) and np.isfinite(max_val)):
         print("Normalize Warning: Non-finite values in scores. Returning original.")
         return scores # Or handle differently, e.g., return NaNs
    return (scores - min_val) / (max_val - min_val)

# --- 6. Run the Algorithms and Collect Metrics ---

metrics_results = {}

# Parameters - **TUNE THESE**
lambda_param = 1.0 / np.sqrt(max(N, D)) # Default lambda for PCP [cite: 2096]
target_rank = 5 # Example for GoDec: Needs tuning based on background complexity [cite: 2977]
sparsity_ratio = 0.05 # Example for GoDec: Needs tuning, often > true anomaly ratio [cite: 3095, 3101]

# --- Option A: PCP-ALM ---
print("-" * 30)
model_name_a = "PCP_ALM"
try:
    L_pcp, S_pcp, time_pcp = pcp_alm(X, lambda_val=lambda_param, verbose=True, max_iter=500, tol=1e-6)
    scores_pcp_img = calculate_anomaly_scores(S_pcp, H, W)
    scores_pcp_flat = scores_pcp_img.flatten()

    # Normalize scores *before* calculating SER/AER if needed by their formula
    scores_pcp_norm_flat = normalize_scores(scores_pcp_flat)

    # Check for valid scores before metric calculation
    if np.all(np.isfinite(scores_pcp_flat)) and n_anomaly_gt > 0 and n_background_gt > 0:
        auc_pcp = roc_auc_score(gt_flat, scores_pcp_flat)
        ser_pcp = calculate_ser(scores_pcp_norm_flat, gt_flat)
        aer_pcp = calculate_aer(scores_pcp_flat, gt_flat) # Use original scores for AER thresholds
        fpr_pcp, tpr_pcp, thresholds_pcp = roc_curve(gt_flat, scores_pcp_flat)
    else:
        auc_pcp, ser_pcp, aer_pcp = np.nan, np.nan, np.nan
        fpr_pcp, tpr_pcp, thresholds_pcp = np.array([0]), np.array([0]), np.array([0])
        print(f"{model_name_a} Warning: Metrics calculation skipped due to invalid scores or ground truth.")

    metrics_results[model_name_a] = {
        'AUC': auc_pcp, 'SER': ser_pcp, 'AER': aer_pcp, 'Time': time_pcp,
        'Scores': scores_pcp_img, 'FPR': fpr_pcp, 'TPR': tpr_pcp, 'Thresholds': thresholds_pcp
    }
    print(f"{model_name_a} Metrics: AUC={auc_pcp:.4f}, SER={ser_pcp:.4f}, AER={aer_pcp:.4f}, Time={time_pcp:.2f}s")
except Exception as e:
    print(f"Error running {model_name_a}: {e}")
    metrics_results[model_name_a] = {'AUC': np.nan, 'SER': np.nan, 'AER': np.nan, 'Time': np.nan, 'Scores': np.zeros((H,W)), 'FPR':[], 'TPR':[], 'Thresholds':[]}


# --- Option B: LRaSMD-GoDec ---
print("-" * 30)
model_name_b = f"LRaSMD_GoDec_r{target_rank}_k{sparsity_ratio:.2f}"
try:
    B_godec, S_godec, time_godec = lrasmd_godec(X, rank_r=target_rank, sparsity_k_ratio=sparsity_ratio, verbose=True, max_iter=100, tol=1e-6)
    scores_godec_img = calculate_anomaly_scores(S_godec, H, W)
    scores_godec_flat = scores_godec_img.flatten()
    scores_godec_norm_flat = normalize_scores(scores_godec_flat)

    if np.all(np.isfinite(scores_godec_flat)) and n_anomaly_gt > 0 and n_background_gt > 0:
        auc_godec = roc_auc_score(gt_flat, scores_godec_flat)
        ser_godec = calculate_ser(scores_godec_norm_flat, gt_flat)
        aer_godec = calculate_aer(scores_godec_flat, gt_flat)
        fpr_godec, tpr_godec, thresholds_godec = roc_curve(gt_flat, scores_godec_flat)
    else:
        auc_godec, ser_godec, aer_godec = np.nan, np.nan, np.nan
        fpr_godec, tpr_godec, thresholds_godec = np.array([0]), np.array([0]), np.array([0])
        print(f"{model_name_b} Warning: Metrics calculation skipped due to invalid scores or ground truth.")

    metrics_results[model_name_b] = {
        'AUC': auc_godec, 'SER': ser_godec, 'AER': aer_godec, 'Time': time_godec,
        'Scores': scores_godec_img, 'FPR': fpr_godec, 'TPR': tpr_godec, 'Thresholds': thresholds_godec
    }
    print(f"{model_name_b} Metrics: AUC={auc_godec:.4f}, SER={ser_godec:.4f}, AER={aer_godec:.4f}, Time={time_godec:.2f}s")
except Exception as e:
    print(f"Error running {model_name_b}: {e}")
    metrics_results[model_name_b] = {'AUC': np.nan, 'SER': np.nan, 'AER': np.nan, 'Time': np.nan, 'Scores': np.zeros((H,W)), 'FPR':[], 'TPR':[], 'Thresholds':[]}

print("-" * 30)


# --- 7. Visualization & Saving Plots ---

# Define plot saving function
def save_plot(fig, filename_base, model_name, plot_type, metrics_dict=None):
    """Saves the figure with a descriptive filename."""
    metric_str = ""
    if metrics_dict and 'AUC' in metrics_dict and not np.isnan(metrics_dict['AUC']):
        metric_str = f"_AUC{metrics_dict['AUC']:.3f}"
    # Add GoDec params if applicable
    if "GoDec" in model_name:
         params = model_name.split('_')[-2:] # Extracts rX and kX.XX
         metric_str += f"_{params[0]}_{params[1]}"

    filename = f"{filename_base}_{model_name}_{plot_type}{metric_str}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        fig.savefig(filepath, bbox_inches='tight', dpi=150)
        print(f"Saved plot: {filepath}")
    except Exception as e:
        print(f"Error saving plot {filepath}: {e}")
    plt.close(fig) # Close the figure to free memory

# Plot 1: Anomaly Score Maps
fig_maps, axes_maps = plt.subplots(1, len(metrics_results) + 1, figsize=(6 * (len(metrics_results) + 1), 6))
# Ensure axes_maps is always an array, even if only one model is run
if len(metrics_results) == 0:
     axes_maps = [axes_maps] # Make it a list/array of one subplot
else:
     axes_maps = axes_maps.flatten()


im0 = axes_maps[0].imshow(ground_truth, cmap='gray')
axes_maps[0].set_title('Ground Truth')
axes_maps[0].set_xticks([])
axes_maps[0].set_yticks([])
fig_maps.colorbar(im0, ax=axes_maps[0], fraction=0.046, pad=0.04)

for i, (model_name, results) in enumerate(metrics_results.items()):
    ax_idx = i + 1
    scores_img = results.get('Scores', np.zeros((H,W))) # Use get with default
    auc_val = results.get('AUC', np.nan)
    auc_title_str = f"(AUC={auc_val:.3f})" if not np.isnan(auc_val) else "(AUC: N/A)"
    im = axes_maps[ax_idx].imshow(scores_img, cmap='hot')
    axes_maps[ax_idx].set_title(f"{model_name} Scores {auc_title_str}")
    axes_maps[ax_idx].set_xticks([])
    axes_maps[ax_idx].set_yticks([])
    fig_maps.colorbar(im, ax=axes_maps[ax_idx], fraction=0.046, pad=0.04)

plt.tight_layout()
save_plot(fig_maps, dataset_basename, "Combined", "ScoreMaps")

# Plot 2: ROC Curves
fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
added_chance_label = False
for model_name, results in metrics_results.items():
    fpr = results.get('FPR', None)
    tpr = results.get('TPR', None)
    roc_auc = results.get('AUC', np.nan)

    if fpr is not None and tpr is not None and len(fpr)>1 and not np.isnan(roc_auc): # Check if ROC data is valid
        ax_roc.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    else:
         # Plot a dummy point or skip if metrics are invalid
         ax_roc.plot([0],[0], marker='x', linestyle='None', label=f"{model_name} (AUC = N/A)")
         print(f"Skipping ROC plot for {model_name} due to invalid metrics or insufficient data.")

ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')

ax_roc.set_xlim([-0.01, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate (FPR) [0-1]') # Clarified label
ax_roc.set_ylabel('True Positive Rate (TPR) [0-1]')
ax_roc.set_title(f'ROC Curves - {dataset_basename}')
ax_roc.legend(loc="lower right")
ax_roc.grid(True)
save_plot(fig_roc, dataset_basename, "Combined", "ROC")


# --- 8. Threshold Selection & Saving Prediction Maps ---
optimal_thresholds = {}
num_models_valid_roc = sum(1 for res in metrics_results.values() if res.get('FPR') is not None and len(res['FPR']) > 1)

if num_models_valid_roc > 0:
    fig_preds, axes_preds = plt.subplots(1, num_models_valid_roc + 1, figsize=(6 * (num_models_valid_roc + 1), 6))
    if num_models_valid_roc == 0:
         axes_preds = [axes_preds] # Make it a list/array of one subplot
    else:
         axes_preds = axes_preds.flatten()

    axes_preds[0].imshow(ground_truth, cmap='gray')
    axes_preds[0].set_title('Ground Truth')
    axes_preds[0].set_xticks([])
    axes_preds[0].set_yticks([])

    plot_idx = 1
    for model_name, results in metrics_results.items():
        tpr = results.get('TPR', None)
        fpr = results.get('FPR', None)
        thresholds = results.get('Thresholds', None)
        scores_img = results.get('Scores', None)

        if tpr is not None and fpr is not None and thresholds is not None and scores_img is not None and len(tpr) > 1:
            # Find optimal threshold using Youden's J statistic
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            optimal_thresholds[model_name] = optimal_threshold
            print(f"\nOptimal Threshold for {model_name} (closest to top-left): {optimal_threshold:.4f}")
            print(f"  -> TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
            pred_map = (scores_img > optimal_threshold).astype(int)

            # Display prediction map
            axes_preds[plot_idx].imshow(pred_map, cmap='gray')
            axes_preds[plot_idx].set_title(f'{model_name} Pred (Thresh={optimal_threshold:.2f})')
            axes_preds[plot_idx].set_xticks([])
            axes_preds[plot_idx].set_yticks([])
            plot_idx += 1
        else:
             print(f"Skipping prediction map for {model_name} due to invalid ROC data.")
             optimal_thresholds[model_name] = np.nan # Store NaN if not calculable

    plt.tight_layout()
    save_plot(fig_preds, dataset_basename, "Combined", "PredictionMaps")
else:
    print("\nSkipping Prediction Maps generation as no valid ROC data was found for any model.")


# --- 9. Save Metrics Summary ---
summary_filepath = os.path.join(output_dir, f"{dataset_basename}_results_summary.txt")
with open(summary_filepath, 'w') as f:
    f.write(f"Anomaly Detection Results Summary for Dataset: {dataset_basename}\n")
    f.write(f"Timestamp: {time.ctime()}\n")
    f.write("="*40 + "\n\n")

    for model_name, metrics in metrics_results.items():
        f.write(f"Model: {model_name}\n")
        # Include parameters if available
        if "PCP_ALM" in model_name:
             f.write(f"  Lambda (λ): {lambda_param:.4e}\n")
        elif "LRaSMD_GoDec" in model_name:
             f.write(f"  Target Rank (r): {target_rank}\n")
             f.write(f"  Sparsity Ratio (k): {sparsity_ratio:.4f}\n")

        f.write(f"  AUC: {metrics.get('AUC', 'N/A'):.4f}\n")
        f.write(f"  SER: {metrics.get('SER', 'N/A'):.4f}\n")
        f.write(f"  AER: {metrics.get('AER', 'N/A'):.4f}\n")
        f.write(f"  Time (s): {metrics.get('Time', 'N/A'):.2f}\n")
        f.write(f"  Optimal Threshold: {optimal_thresholds.get(model_name, 'N/A'):.4f}\n")
        f.write("-" * 20 + "\n")

print(f"\nMetrics summary saved to: {summary_filepath}")
print("Script finished.")