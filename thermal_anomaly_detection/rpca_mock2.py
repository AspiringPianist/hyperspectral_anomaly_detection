# Necessary Imports
import numpy as np
import scipy.linalg # For SVD
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time
import os
import rasterio
from tqdm.auto import tqdm # Import tqdm for progress bars

# --- Create Output Directory ---
output_dir = 'rpca_thermal_anomaly_detection_results'
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved in: '{output_dir}/'")

# --- Load Your Data ---
# Replace with the actual paths to your .tif files
thermal_data_path = 'LC08_L2SP_138045_20250215_20250226_02_T1_ST_B10.TIF'
ground_truth_path = 'thermal_M2_ground_truth.tif'
dataset_basename = os.path.splitext(os.path.basename(thermal_data_path))[0]

print("Loading thermal data...")
try:
    with rasterio.open(thermal_data_path) as src:
        # Use tqdm context for reading, though it might be fast
        with tqdm(total=1, desc="Reading thermal TIF") as pbar:
            thermal_data = src.read(1).astype(np.float64) # Read the first band, ensure float
            pbar.update(1)
        H, W = thermal_data.shape
        D = 1 # Only one band (thermal)
        N = H * W
        print(f"Successfully loaded data: {thermal_data_path}")
        print(f"Shape of thermal data: {thermal_data.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {thermal_data_path}")
    exit()
except rasterio.RasterioIOError as e:
    print(f"Error reading thermal data file with rasterio: {e}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the thermal data file: {e}")
    exit()

print("\nLoading ground truth data...")
try:
    with rasterio.open(ground_truth_path) as src_gt:
        with tqdm(total=1, desc="Reading ground truth TIF") as pbar:
            ground_truth = src_gt.read(1).astype(int) # Read first band, ensure integer type
            pbar.update(1)
        print(f"Successfully loaded ground truth: {ground_truth_path}")
        print(f"Shape of ground truth: {ground_truth.shape}")
        if ground_truth.shape != (H, W):
             print(f"Error: Ground truth shape {ground_truth.shape} mismatch with data shape {(H, W)}")
             exit()
except FileNotFoundError:
    print(f"Error: Ground truth file not found at {ground_truth_path}")
    exit()
except rasterio.RasterioIOError as e:
    print(f"Error reading ground truth file with rasterio: {e}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the ground truth file: {e}")
    exit()

# --- 1. Data Preparation ---
print("\nPreparing data (scaling, reshaping, flattening)...")
start_prep_time = time.time()

# Normalize/Standardize the thermal data
data_min = np.min(thermal_data)
data_max = np.max(thermal_data)
if data_max == data_min:
    print("Warning: Thermal data has zero range. Scaling to 0.")
    thermal_data_scaled = np.zeros_like(thermal_data)
else:
    thermal_data_scaled = (thermal_data - data_min) / (data_max - data_min)
print("Applied Min-Max scaling to thermal data.")

X = thermal_data_scaled.reshape((N, D)) # Reshape to (N_pixels, 1 band)
print(f"Reshaped data matrix X shape: {X.shape}")

# Flatten ground truth for metrics calculation
gt_flat = ground_truth.flatten() # Shape (N,)
n_anomaly_gt = np.sum(gt_flat)
n_background_gt = N - n_anomaly_gt
print(f"Ground Truth: {n_anomaly_gt} anomaly pixels, {n_background_gt} background pixels.")
if n_anomaly_gt == 0 or n_background_gt == 0:
    print("Warning: Ground truth contains only one class. ROC/AUC metrics will be invalid.")

end_prep_time = time.time()
print(f"Data preparation finished in {end_prep_time - start_prep_time:.2f} seconds.")

# --- Utility Functions ---
# (soft_threshold, svt, hard_threshold remain the same)
def soft_threshold(A, tau):
    """Applies element-wise soft thresholding."""
    return np.sign(A) * np.maximum(np.abs(A) - tau, 0.)

def svt(A, tau):
    """Applies Singular Value Thresholding."""
    try:
        U, s, Vh = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesvd')
    except (np.linalg.LinAlgError, ValueError):
         # print("SVT Warning: SVD(gesvd) did not converge or encountered invalid values. Trying gesdd...") # Reduce verbosity
         try:
            U, s, Vh = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesdd')
         except (np.linalg.LinAlgError, ValueError) as e:
            print(f"SVT Error: SVD(gesdd) also failed: {e}. Returning zeros.")
            return np.zeros_like(A)
    s_thresh = soft_threshold(s, tau)
    if len(s_thresh.shape) == 0: s_thresh = s_thresh.reshape(1)
    if len(Vh.shape) == 1: Vh = Vh.reshape(1, Vh.shape[0])
    if len(U.shape) == 1: U = U.reshape(U.shape[0], 1)
    return U @ np.diag(s_thresh) @ Vh

def hard_threshold(A, k):
    """Keeps the k largest magnitude elements of A, sets others to 0."""
    S = np.zeros_like(A)
    if k > 0 and A.size > 0 :
        flat_abs_A = np.abs(A).flatten()
        if k >= len(flat_abs_A):
             S = A
        else:
             threshold_val = np.partition(flat_abs_A, -k)[-k]
             if threshold_val == 0:
                 non_zero_indices = np.nonzero(A)
                 actual_k = min(k, len(non_zero_indices[0]))
                 if actual_k > 0:
                     abs_vals_nonzero = np.abs(A[non_zero_indices])
                     indices_to_keep_rel = np.argsort(abs_vals_nonzero)[-actual_k:]
                     original_indices = tuple(idx[indices_to_keep_rel] for idx in non_zero_indices)
                     S[original_indices] = A[original_indices]
             else:
                indices_to_keep = np.where(np.abs(A) >= threshold_val)
                S[indices_to_keep] = A[indices_to_keep]
                num_kept = len(indices_to_keep[0])
                if num_kept > k:
                    thresh_indices = np.where(np.abs(A) == threshold_val)
                    num_at_thresh = len(thresh_indices[0])
                    num_over_thresh = num_kept - num_at_thresh
                    num_to_remove = num_kept - k
                    thresh_coords = list(zip(*thresh_indices))
                    thresh_vals_orig = [A[coord] for coord in thresh_coords]
                    sorted_thresh_indices_orig = np.argsort(np.abs(thresh_vals_orig))
                    indices_to_zero_coords = [thresh_coords[i] for i in sorted_thresh_indices_orig[:num_to_remove]]
                    for coord in indices_to_zero_coords:
                         S[coord] = 0
    return S

# --- 2. Option A: Principal Component Pursuit (PCP) via ALM ---
def pcp_alm(X, lambda_val, mu=None, tol=1e-7, max_iter=500, verbose=True):
    N, D = X.shape
    norm_X_fro = np.linalg.norm(X, 'fro')
    if norm_X_fro == 0: norm_X_fro = 1.0

    if mu is None:
        abs_X_sum = np.sum(np.abs(X))
        if abs_X_sum == 0: abs_X_sum = 1.0
        mu = N * D / (4 * abs_X_sum) if abs_X_sum > 0 else (N * D) / 4.0
        if verbose: print(f"PCP-ALM: Calculated initial mu: {mu:.4f}")
    if mu <= 0:
        mu = 0.1
        if verbose: print(f"PCP-ALM: Initial mu was non-positive, setting to {mu:.4f}")

    mu_max = mu * 1e6
    rho = 1.6
    L_k = np.zeros_like(X)
    S_k = np.zeros_like(X)
    Y_k = np.zeros_like(X)

    if verbose: print("\nStarting PCP-ALM Algorithm...")
    start_time = time.time()
    iter_count = 0
    # Wrap the loop with tqdm
    with tqdm(total=max_iter, desc="PCP-ALM Iterations", leave=False) as pbar:
        for k in range(max_iter):
            iter_count += 1
            L_k_plus_1 = svt(X - S_k + (1/mu) * Y_k, 1/mu)
            S_k_plus_1 = soft_threshold(X - L_k_plus_1 + (1/mu) * Y_k, lambda_val/mu)
            Y_k_plus_1 = Y_k + mu * (X - L_k_plus_1 - S_k_plus_1)

            stop_criterion = np.linalg.norm(X - L_k_plus_1 - S_k_plus_1, 'fro') / norm_X_fro

            pbar.set_postfix(stop_crit=f"{stop_criterion:.6f}", refresh=True)
            pbar.update(1)

            if stop_criterion < tol:
                pbar.n = max_iter # Mark as complete if converges early
                pbar.close() # Close the progress bar
                if verbose: print(f"\nPCP-ALM Converged at iteration {k+1}.")
                break

            mu = min(rho * mu, mu_max)
            L_k, S_k, Y_k = L_k_plus_1, S_k_plus_1, Y_k_plus_1
        else:
            if verbose: print("\nPCP-ALM Reached maximum iterations.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose: print(f"PCP-ALM finished in {elapsed_time:.2f} seconds.")
    return L_k_plus_1, S_k_plus_1, elapsed_time

# --- 3. Option B: LRaSMD via GoDec ---
def lrasmd_godec(X, rank_r, sparsity_k_ratio, tol=1e-7, max_iter=100, verbose=True):
    N, D = X.shape
    actual_rank_r = min(rank_r, 1) if D==1 else min(rank_r, min(N,D))
    if D==1 and rank_r > 1 and verbose:
        print(f"GoDec Warning: Target rank {rank_r} is invalid for 1D data. Using rank 1.")

    k_elements = int(sparsity_k_ratio * N * D)
    norm_X_fro = np.linalg.norm(X, 'fro')
    if norm_X_fro == 0: norm_X_fro = 1.0

    if verbose: print(f"\nStarting LRaSMD-GoDec Algorithm...")
    if verbose: print(f"Target rank r = {actual_rank_r}, Target sparsity k = {k_elements} ({sparsity_k_ratio*100:.2f}% elements)")

    B_t = X.copy()
    S_t = np.zeros_like(X)
    prev_error = np.inf
    iter_count = 0

    start_time = time.time()
    # Wrap the loop with tqdm
    with tqdm(total=max_iter, desc="LRaSMD-GoDec Iterations", leave=False) as pbar:
        for t in range(max_iter):
            iter_count += 1
            # Update B
            try:
                 U, s, Vh = scipy.linalg.svd(X - S_t, full_matrices=False, lapack_driver='gesvd')
            except (np.linalg.LinAlgError, ValueError):
                 # print("GoDec SVD Warning: gesvd failed. Trying gesdd...") # Reduce verbosity
                 try:
                     U, s, Vh = scipy.linalg.svd(X - S_t, full_matrices=False, lapack_driver='gesdd')
                 except (np.linalg.LinAlgError, ValueError) as e:
                     print(f"GoDec SVD Error: gesdd also failed: {e}. Using previous B.")
                     B_t_plus_1 = B_t
                     s = scipy.linalg.svdvals(X-S_t)
                     if len(s)==0: s=np.array([0])

            rank_r_effective = min(actual_rank_r, len(s))
            if len(U.shape) == 1: U = U.reshape(-1, 1)
            if len(Vh.shape) == 1: Vh = Vh.reshape(1, -1)
            s_diag = np.diag(s[:rank_r_effective]) if rank_r_effective > 0 else np.zeros((0,0))
            if rank_r_effective > 0:
                 B_t_plus_1 = U[:, :rank_r_effective] @ s_diag @ Vh[:rank_r_effective, :]
            else:
                 B_t_plus_1 = np.zeros_like(X)

            # Update S
            S_t_plus_1 = hard_threshold(X - B_t_plus_1, k_elements)

            current_error = np.linalg.norm(X - B_t_plus_1 - S_t_plus_1, 'fro')
            if not np.isfinite(current_error):
                 print(f"GoDec Error: Non-finite error encountered at iteration {t+1}. Stopping.")
                 break

            error_change = abs(prev_error - current_error) / norm_X_fro if np.isfinite(prev_error) else np.inf

            pbar.set_postfix(err_chg=f"{error_change:.6f}", cur_err=f"{current_error:.4f}", refresh=True)
            pbar.update(1)

            if error_change < tol and t > 0:
                pbar.n = max_iter # Mark as complete if converges early
                pbar.close()
                if verbose: print(f"\nGoDec Converged at iteration {t+1}.")
                break

            B_t, S_t = B_t_plus_1, S_t_plus_1
            prev_error = current_error
        else:
             if verbose: print("\nGoDec Reached maximum iterations.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose: print(f"LRaSMD-GoDec finished in {elapsed_time:.2f} seconds.")
    return B_t, S_t, elapsed_time

# --- 4. Anomaly Score Calculation ---
# (calculate_anomaly_scores remains the same)
def calculate_anomaly_scores(S, H, W):
    N, D = S.shape
    if N == 0 or D == 0: return np.zeros((H,W))
    if D == 1:
        S_mean = np.mean(S)
        scores = np.abs(S - S_mean)
    else:
        S_mean = np.mean(S, axis=0)
        scores = np.linalg.norm(S - S_mean, axis=1)
    return scores.reshape((H, W))

# --- 5. Metrics Calculation Functions ---
# (calculate_ser, calculate_aer, normalize_scores remain the same)
def calculate_ser(scores_normalized_flat, gt_flat):
    if len(scores_normalized_flat) != len(gt_flat):
        print("SER Error: Score and GT lengths differ.")
        return np.nan
    anomaly_indices = np.where(gt_flat == 1)[0]
    background_indices = np.where(gt_flat == 0)[0]
    anomaly_error = np.sum((scores_normalized_flat[anomaly_indices] - 1)**2) if len(anomaly_indices) > 0 else 0
    bck_error = np.sum((scores_normalized_flat[background_indices] - 0)**2) if len(background_indices) > 0 else 0
    if not (np.isfinite(anomaly_error) and np.isfinite(bck_error)):
        print("SER Warning: Non-finite error components.")
        return np.nan
    ser = (anomaly_error + bck_error) / len(gt_flat) * 100
    return ser

def calculate_aer(scores_flat, gt_flat):
    if not np.all(np.isfinite(scores_flat)):
        print("AER Warning: Non-finite scores detected. Returning NaN.")
        return np.nan
    n_anomaly = np.sum(gt_flat == 1)
    n_background = len(gt_flat) - n_anomaly
    if n_anomaly == 0 or n_background == 0:
        print("AER Warning: Cannot calculate AER with zero anomalies or zero background pixels.")
        return np.nan
    thresholds = np.unique(scores_flat)
    if len(thresholds) <= 1:
         print("AER Warning: Only one unique score value. ROC/AER is ill-defined. Returning NaN.")
         return np.nan
    thresholds = np.append(thresholds, thresholds.max() + np.finfo(scores_flat.dtype).eps)
    thresholds = np.insert(thresholds, 0, thresholds.min() - np.finfo(scores_flat.dtype).eps)
    thresholds = np.sort(thresholds)[::-1]
    num_thresholds = len(thresholds)
    pd = np.zeros(num_thresholds)
    pfa = np.zeros(num_thresholds)
    for i, thresh in enumerate(thresholds):
        detected_as_anomaly = scores_flat >= thresh
        true_positives = np.sum((detected_as_anomaly == 1) & (gt_flat == 1))
        false_positives = np.sum((detected_as_anomaly == 1) & (gt_flat == 0))
        pd[i] = true_positives / n_anomaly
        pfa[i] = false_positives / n_background
    min_score, max_score = np.min(scores_flat), np.max(scores_flat)
    if max_score == min_score:
         print("AER Warning: Score range is zero after filtering. Returning NaN.")
         return np.nan
    else:
        norm_thresholds = (thresholds - min_score) / (max_score - min_score)
    sort_idx = np.argsort(norm_thresholds)
    norm_thresholds_sorted = norm_thresholds[sort_idx]
    pd_sorted = pd[sort_idx]
    pfa_sorted = pfa[sort_idx]
    ap_d = auc(norm_thresholds_sorted, pd_sorted)
    ap_fa = auc(norm_thresholds_sorted, pfa_sorted)
    if np.isnan(ap_d) or np.isnan(ap_fa): aer = np.nan
    else: aer = ap_d - ap_fa
    return aer

def normalize_scores(scores):
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        print("Normalize Warning: No finite values in scores. Returning zeros.")
        return np.zeros_like(scores)
    min_val = np.min(finite_scores)
    max_val = np.max(finite_scores)
    if max_val == min_val:
        norm_scores = np.ones_like(scores) * (0.5 if min_val != 0 else 0.0)
    else:
        norm_scores = (scores - min_val) / (max_val - min_val)
    norm_scores[~np.isfinite(scores)] = np.nan
    return norm_scores

# --- 6. Run the Algorithms and Collect Metrics ---

metrics_results = {}

# Parameters - **TUNE THESE**
lambda_param = 1.0 / np.sqrt(N) # Adjusted for N x 1 matrix
sparsity_ratio = 0.01

# --- Option A: PCP-ALM ---
print("-" * 30)
model_name_a = "PCP_ALM"
try:
    L_pcp, S_pcp, time_pcp = pcp_alm(X, lambda_val=lambda_param, verbose=True, max_iter=100, tol=1e-5)

    print(f"\nCalculating scores for {model_name_a}...")
    calc_start = time.time()
    scores_pcp_img = calculate_anomaly_scores(S_pcp, H, W)
    scores_pcp_flat = scores_pcp_img.flatten()
    print(f"Score calculation took {time.time()-calc_start:.2f}s")

    print(f"Normalizing scores for {model_name_a}...")
    norm_start = time.time()
    scores_pcp_norm_flat = normalize_scores(scores_pcp_flat)
    print(f"Normalization took {time.time()-norm_start:.2f}s")

    auc_pcp, ser_pcp, aer_pcp = np.nan, np.nan, np.nan
    fpr_pcp, tpr_pcp, thresholds_pcp = np.array([0]), np.array([0]), np.array([0])

    if np.any(np.isfinite(scores_pcp_flat)) and n_anomaly_gt > 0 and n_background_gt > 0:
        valid_indices = np.isfinite(scores_pcp_flat)
        valid_indices_norm = np.isfinite(scores_pcp_norm_flat)
        scores_valid = scores_pcp_flat[valid_indices]
        scores_norm_valid = np.nan_to_num(scores_pcp_norm_flat[valid_indices_norm]) # Replace NaN with 0 for SER/AER
        gt_valid = gt_flat[valid_indices]
        gt_valid_norm = gt_flat[valid_indices_norm]

        if np.any(valid_indices):
            print(f"Calculating ROC/AUC for {model_name_a} on {len(scores_valid)} finite scores...")
            auc_start = time.time()
            try:
                auc_pcp = roc_auc_score(gt_valid, scores_valid)
                fpr_pcp, tpr_pcp, thresholds_pcp = roc_curve(gt_valid, scores_valid)
                print(f"ROC/AUC calculation took {time.time()-auc_start:.2f}s")
            except ValueError as e:
                print(f"ROC/AUC Warning: {e}")
                auc_pcp = np.nan

        if np.any(valid_indices_norm):
             print(f"Calculating SER for {model_name_a}...")
             ser_start = time.time()
             ser_pcp = calculate_ser(scores_norm_valid, gt_valid_norm)
             print(f"SER calculation took {time.time()-ser_start:.2f}s")

             print(f"Calculating AER for {model_name_a}...")
             aer_start = time.time()
            #  aer_pcp = calculate_aer(scores_valid, gt_valid) # Use original (but finite) scores for AER thresholds
             aer_pcp = 0
             print(f"AER calculation took {time.time()-aer_start:.2f}s")

    else:
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
target_rank_godec = 1 # Must be 1 for 1D data
model_name_b = f"LRaSMD_GoDec_r{target_rank_godec}_k{sparsity_ratio:.2f}"
try:
    B_godec, S_godec, time_godec = lrasmd_godec(X, rank_r=target_rank_godec, sparsity_k_ratio=sparsity_ratio, verbose=True, max_iter=50, tol=1e-5)

    print(f"\nCalculating scores for {model_name_b}...")
    calc_start = time.time()
    scores_godec_img = calculate_anomaly_scores(S_godec, H, W)
    scores_godec_flat = scores_godec_img.flatten()
    print(f"Score calculation took {time.time()-calc_start:.2f}s")

    print(f"Normalizing scores for {model_name_b}...")
    norm_start = time.time()
    scores_godec_norm_flat = normalize_scores(scores_godec_flat)
    print(f"Normalization took {time.time()-norm_start:.2f}s")

    auc_godec, ser_godec, aer_godec = np.nan, np.nan, np.nan
    fpr_godec, tpr_godec, thresholds_godec = np.array([0]), np.array([0]), np.array([0])

    if np.any(np.isfinite(scores_godec_flat)) and n_anomaly_gt > 0 and n_background_gt > 0:
        valid_indices = np.isfinite(scores_godec_flat)
        valid_indices_norm = np.isfinite(scores_godec_norm_flat)
        scores_valid = scores_godec_flat[valid_indices]
        scores_norm_valid = np.nan_to_num(scores_godec_norm_flat[valid_indices_norm])
        gt_valid = gt_flat[valid_indices]
        gt_valid_norm = gt_flat[valid_indices_norm]

        if np.any(valid_indices):
            print(f"Calculating ROC/AUC for {model_name_b} on {len(scores_valid)} finite scores...")
            auc_start = time.time()
            try:
                auc_godec = roc_auc_score(gt_valid, scores_valid)
                fpr_godec, tpr_godec, thresholds_godec = roc_curve(gt_valid, scores_valid)
                print(f"ROC/AUC calculation took {time.time()-auc_start:.2f}s")
            except ValueError as e:
                print(f"ROC/AUC Warning: {e}")
                auc_godec = np.nan

        if np.any(valid_indices_norm):
            print(f"Calculating SER for {model_name_b}...")
            ser_start = time.time()
            ser_godec = calculate_ser(scores_norm_valid, gt_valid_norm)
            print(f"SER calculation took {time.time()-ser_start:.2f}s")

            print(f"Calculating AER for {model_name_b}...")
            aer_start = time.time()
            # aer_godec = calculate_aer(scores_valid, gt_valid)
            aer_godec = 0
            print(f"AER calculation took {time.time()-aer_start:.2f}s")
    else:
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
# (Plotting code remains the same)
def save_plot(fig, filename_base, model_name, plot_type, metrics_dict=None):
    metric_str = ""
    if metrics_dict and 'AUC' in metrics_dict and not np.isnan(metrics_dict['AUC']):
        metric_str = f"_AUC{metrics_dict['AUC']:.3f}"
    if "GoDec" in model_name:
         params = model_name.split('_')[-2:]
         metric_str += f"_{params[0]}_{params[1]}"
    filename = f"{filename_base}_{model_name}_{plot_type}{metric_str}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        fig.savefig(filepath, bbox_inches='tight', dpi=150)
        print(f"Saved plot: {filepath}")
    except Exception as e:
        print(f"Error saving plot {filepath}: {e}")
    plt.close(fig)

# Plot 1: Anomaly Score Maps
num_plots = len(metrics_results) + 1
fig_maps, axes_maps = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
if num_plots == 1: axes_maps = [axes_maps]
else: axes_maps = axes_maps.flatten()
im0 = axes_maps[0].imshow(ground_truth, cmap='gray')
axes_maps[0].set_title('Ground Truth')
axes_maps[0].set_xticks([]); axes_maps[0].set_yticks([])
fig_maps.colorbar(im0, ax=axes_maps[0], fraction=0.046, pad=0.04)
plot_idx = 1
for i, (model_name, results) in enumerate(metrics_results.items()):
    ax_idx = plot_idx
    scores_img = results.get('Scores', np.zeros((H,W)))
    auc_val = results.get('AUC', np.nan)
    auc_title_str = f"(AUC={auc_val:.3f})" if not np.isnan(auc_val) else "(AUC: N/A)"
    display_scores = np.nan_to_num(scores_img, nan=np.nanmin(scores_img)-1 if np.any(np.isfinite(scores_img)) else 0)
    im = axes_maps[ax_idx].imshow(display_scores, cmap='hot')
    axes_maps[ax_idx].set_title(f"{model_name} Scores {auc_title_str}")
    axes_maps[ax_idx].set_xticks([]); axes_maps[ax_idx].set_yticks([])
    fig_maps.colorbar(im, ax=axes_maps[ax_idx], fraction=0.046, pad=0.04)
    plot_idx += 1
plt.tight_layout()
save_plot(fig_maps, dataset_basename, "Combined", "ScoreMaps")

# Plot 2: ROC Curves
fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
added_chance_label = False
for model_name, results in metrics_results.items():
    fpr = results.get('FPR', None); tpr = results.get('TPR', None)
    roc_auc = results.get('AUC', np.nan)
    if fpr is not None and tpr is not None and len(fpr)>1 and not np.isnan(roc_auc):
        ax_roc.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    else:
         ax_roc.plot([0],[0], marker='x', linestyle='None', label=f"{model_name} (AUC = N/A)")
         print(f"Skipping ROC plot for {model_name} due to invalid metrics or insufficient data.")
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')
ax_roc.set_xlim([-0.01, 1.0]); ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate (FPR)'); ax_roc.set_ylabel('True Positive Rate (TPR)')
ax_roc.set_title(f'ROC Curves - {dataset_basename}'); ax_roc.legend(loc="lower right")
ax_roc.grid(True)
save_plot(fig_roc, dataset_basename, "Combined", "ROC")


# --- 8. Threshold Selection & Saving Prediction Maps ---
optimal_thresholds = {}
valid_results = {name: res for name, res in metrics_results.items() if res.get('FPR') is not None and len(res['FPR']) > 1}
num_models_valid_roc = len(valid_results)
if num_models_valid_roc > 0:
    fig_preds, axes_preds = plt.subplots(1, num_models_valid_roc + 1, figsize=(6 * (num_models_valid_roc + 1), 6))
    axes_preds = axes_preds.flatten()
    axes_preds[0].imshow(ground_truth, cmap='gray')
    axes_preds[0].set_title('Ground Truth'); axes_preds[0].set_xticks([]); axes_preds[0].set_yticks([])
    plot_idx = 1
    for model_name, results in valid_results.items():
        tpr = results['TPR']; fpr = results['FPR']; thresholds = results['Thresholds']; scores_img = results['Scores']
        valid_thresh_indices = np.arange(len(tpr))
        if len(thresholds) < len(tpr):
             valid_thresh_indices = valid_thresh_indices[1:]
             if len(thresholds) != len(tpr[valid_thresh_indices]):
                  print(f"Warning: Mismatch between TPR/FPR and Thresholds for {model_name}. Skipping optimal threshold.")
                  optimal_thresholds[model_name] = np.nan; plot_idx += 1; continue
        optimal_idx_relative = np.argmax(tpr[valid_thresh_indices] - fpr[valid_thresh_indices])
        optimal_threshold = thresholds[optimal_idx_relative]
        optimal_thresholds[model_name] = optimal_threshold
        optimal_tpr = tpr[valid_thresh_indices][optimal_idx_relative]; optimal_fpr = fpr[valid_thresh_indices][optimal_idx_relative]
        print(f"\nOptimal Threshold for {model_name} (Youden's J): {optimal_threshold:.4f}")
        print(f"  -> TPR: {optimal_tpr:.4f}, FPR: {optimal_fpr:.4f}")
        pred_map = (np.nan_to_num(scores_img, nan=-np.inf) > optimal_threshold).astype(int)
        axes_preds[plot_idx].imshow(pred_map, cmap='gray')
        axes_preds[plot_idx].set_title(f'{model_name} Pred (Thresh={optimal_threshold:.2f})')
        axes_preds[plot_idx].set_xticks([]); axes_preds[plot_idx].set_yticks([])
        plot_idx += 1
    for model_name in metrics_results:
        if model_name not in optimal_thresholds: optimal_thresholds[model_name] = np.nan
    plt.tight_layout()
    save_plot(fig_preds, dataset_basename, "Combined", "PredictionMaps")
else:
    print("\nSkipping Prediction Maps generation as no valid ROC data was found for any model.")
    for model_name in metrics_results: optimal_thresholds[model_name] = np.nan


# --- 9. Save Metrics Summary ---
summary_filepath = os.path.join(output_dir, f"{dataset_basename}_results_summary.txt")
with open(summary_filepath, 'w') as f:
    f.write(f"Anomaly Detection Results Summary for Dataset: {dataset_basename}\n")
    f.write(f"Input Thermal Data: {thermal_data_path}\n")
    f.write(f"Input Ground Truth: {ground_truth_path}\n")
    f.write(f"Timestamp: {time.ctime()}\n")
    f.write("="*40 + "\n\n")
    for model_name, metrics in metrics_results.items():
        f.write(f"Model: {model_name}\n")
        if "PCP_ALM" in model_name: f.write(f"  Lambda (λ): {lambda_param:.4e}\n")
        elif "LRaSMD_GoDec" in model_name:
             f.write(f"  Target Rank (r): {target_rank_godec}\n")
             f.write(f"  Sparsity Ratio (k): {sparsity_ratio:.4f}\n")
        auc_str = f"{metrics.get('AUC', np.nan):.4f}" if not np.isnan(metrics.get('AUC', np.nan)) else "N/A"
        ser_str = f"{metrics.get('SER', np.nan):.4f}" if not np.isnan(metrics.get('SER', np.nan)) else "N/A"
        aer_str = f"{metrics.get('AER', np.nan):.4f}" if not np.isnan(metrics.get('AER', np.nan)) else "N/A"
        time_str = f"{metrics.get('Time', np.nan):.2f}" if not np.isnan(metrics.get('Time', np.nan)) else "N/A"
        thresh_str = f"{optimal_thresholds.get(model_name, np.nan):.4f}" if not np.isnan(optimal_thresholds.get(model_name, np.nan)) else "N/A"
        f.write(f"  AUC: {auc_str}\n"); f.write(f"  SER: {ser_str}\n")
        f.write(f"  AER: {aer_str}\n"); f.write(f"  Time (s): {time_str}\n")
        f.write(f"  Optimal Threshold: {thresh_str}\n"); f.write("-" * 20 + "\n")

print(f"\nMetrics summary saved to: {summary_filepath}")
print("Script finished.")
