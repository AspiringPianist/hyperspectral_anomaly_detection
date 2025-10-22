# Necessary Imports
import numpy as np
import scipy.io
import scipy.sparse as sp # For sparse matrices (Laplacian)
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv # Using PyG's implementation for Graph Attention
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.special import comb # For Beta function calculation
import matplotlib.pyplot as plt

# --- Configuration & Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

output_dir = 'gan_bwgnn_had_results'
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved in: '{output_dir}/'")

# --- Load Your Data ---
dataset_path = 'Salinas/Salians_120_sj25_syn.mat'
ground_truth_path = 'Salinas/Salians_120_sj25_gt.mat'
dataset_basename = os.path.splitext(os.path.basename(dataset_path))[0]

try:
    mat_data = scipy.io.loadmat(dataset_path)
    data_var_name = 'hsi'
    if data_var_name in mat_data:
        hyperspectral_data = mat_data[data_var_name].astype(np.float64)
        H, W, D = hyperspectral_data.shape
        N = H * W
        print(f"Data shape: {hyperspectral_data.shape}")
    else:
        raise KeyError(f"Variable '{data_var_name}' not found.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

try:
    gt_data = scipy.io.loadmat(ground_truth_path)
    gt_var_name = 'hsi_gt'
    if gt_var_name in gt_data:
        ground_truth = gt_data[gt_var_name].astype(int)
        print(f"Ground truth shape: {ground_truth.shape}")
        if ground_truth.shape != (H, W):
             raise ValueError("Shape mismatch between data and ground truth.")
    else:
         raise KeyError(f"Variable '{gt_var_name}' not found.")
except Exception as e:
    print(f"Error loading ground truth: {e}")
    exit()

# --- 1. Data Preparation & Graph Construction ---
print("\n--- Preparing Data and Building Graph ---")
# Flatten data and ground truth
X_flat = hyperspectral_data.reshape((N, D))
y_flat = ground_truth.flatten() # Labels for training loss (Eq. 22)

# --- Added: Flattened GT and simple evaluation helpers (fix undefined names) ---
gt_flat = ground_truth.flatten()
n_anomaly_gt = int((gt_flat == 1).sum())
n_background_gt = int((gt_flat == 0).sum())

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_flat, dtype=torch.float32).to(device) # BCEWithLogitsLoss expects float

# --- KNN Graph Construction ---
k_neighbors = 10 # Parameter K from paper [cite: 1976, 2173]
print(f"Building KNN graph with k={k_neighbors}...")
start_knn = time.time()
# Calculate KNN graph based on spectral features (pixel values)
# mode='distance' gives weights, 'connectivity' gives binary adjacency
adj_matrix_sparse = kneighbors_graph(X_flat, k_neighbors, mode='connectivity', include_self=False)
end_knn = time.time()
print(f"KNN graph calculation took {end_knn - start_knn:.2f} seconds.")

# Convert sparse adjacency matrix to PyTorch Geometric edge_index format
coo = adj_matrix_sparse.tocoo()
edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long).to(device)
print(f"Graph created with {edge_index.shape[1]} edges.")

# --- Calculate Normalized Laplacian (for BWGNN) ---
# Need symmetric normalization L_sys = I - D^{-1/2} A D^{-1/2} [cite: 1997]
# Or just L = D - A for the polynomial filter W_{a,b} = beta*(L) [cite: 2106]
# Let's use L = D - A as implied by Eq. 14
print("Calculating Laplacian matrix L = D - A...")
start_lap = time.time()
# Ensure adjacency is symmetric for undirected graph Laplacian
adj_matrix_sparse = adj_matrix_sparse + adj_matrix_sparse.T
adj_matrix_sparse.data = np.ones_like(adj_matrix_sparse.data) # Ensure binary

degree_matrix_sparse = sp.diags(adj_matrix_sparse.sum(axis=1).A1)
laplacian_sparse = degree_matrix_sparse - adj_matrix_sparse

# Check if graph is connected (optional but good)
num_components = sp.csgraph.connected_components(laplacian_sparse, directed=False, return_labels=False)
print(f"Graph has {num_components} connected components.")

# Convert sparse Laplacian to PyTorch sparse tensor
laplacian_coo = laplacian_sparse.tocoo()
indices = torch.tensor(np.vstack((laplacian_coo.row, laplacian_coo.col)), dtype=torch.long)
values = torch.tensor(laplacian_coo.data, dtype=torch.float32)
shape = torch.Size(laplacian_coo.shape)
L_sparse_tensor = torch.sparse_coo_tensor(indices, values, shape).to(device)
end_lap = time.time()
print(f"Laplacian calculation took {end_lap - start_lap:.2f} seconds.")


# Create PyG Data object
graph_data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor).to(device)

# --- Helper evaluation functions (previously undefined) ---
def normalize_scores(scores):
    """Min-max normalize a 1D numpy array to [0,1]."""
    scores = np.asarray(scores, dtype=np.float64)
    mn = scores.min()
    mx = scores.max()
    if mx - mn < 1e-12:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)

def calculate_ser(scores_norm, gt_flat_local):
    """Simple SER proxy: ratio of mean normalized score on anomalies to background."""
    gt_arr = np.asarray(gt_flat_local).flatten()
    scores_norm = np.asarray(scores_norm).flatten()
    eps = 1e-12
    if (gt_arr == 1).sum() == 0 or (gt_arr == 0).sum() == 0:
        return np.nan
    mean_anom = scores_norm[gt_arr == 1].mean()
    mean_bg = scores_norm[gt_arr == 0].mean()
    return float((mean_anom) / (mean_bg + eps))

from sklearn.metrics import roc_curve
def calculate_aer(scores, gt_flat_local):
    """Estimate Equal Error Rate (EER)-like AER: threshold where FPR ~= 1-TPR."""
    gt_arr = np.asarray(gt_flat_local).flatten()
    scores = np.asarray(scores).flatten()
    if (gt_arr == 1).sum() == 0 or (gt_arr == 0).sum() == 0:
        return np.nan
    fpr, tpr, thresholds = roc_curve(gt_arr, scores)
    # Find threshold where abs(fpr - (1 - tpr)) is minimal
    diff = np.abs(fpr - (1 - tpr))
    idx = np.argmin(diff)
    eer = (fpr[idx] + (1 - tpr[idx])) / 2.0
    return float(eer)

# --- Moved/Defined: save_plot before its first use (was previously defined later) ---
def save_plot(fig, filename_base, model_name, plot_type, metrics_dict=None):
    """Saves the figure with a descriptive filename."""
    metric_str = ""
    params_str = ""
    if metrics_dict and 'Params' in metrics_dict:
        params = metrics_dict['Params']
        params_str = f"_C{params.get('C','')}_k{params.get('k','')}"
    if metrics_dict and 'AUC' in metrics_dict and not np.isnan(metrics_dict['AUC']):
        metric_str = f"_AUC{metrics_dict['AUC']:.3f}"

    filename = f"{filename_base}_{model_name}_{plot_type}{params_str}{metric_str}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        fig.savefig(filepath, bbox_inches='tight', dpi=150)
        print(f"Saved plot: {filepath}")
    except Exception as e:
        print(f"Error saving plot {filepath}: {e}")
    plt.close(fig) # Close the figure


# --- Helper for Beta Wavelet Polynomial ---
# Using precomputed factorials might be faster for small C
# or use torch.lgamma for larger numbers if needed.
factorials = [1.0] * 21 # Precompute up to 20!
for i in range(2, 21):
    factorials[i] = factorials[i-1] * i

def beta_function_val(a_plus_1, b_plus_1):
    """ Calculates B(a+1, b+1) = a! * b! / (a+b+1)! """
    a, b = a_plus_1 - 1, b_plus_1 - 1
    if a < 0 or b < 0 or (a + b + 1) >= len(factorials):
        # Fallback for large values or invalid input (requires torch.lgamma)
        # return torch.exp(torch.lgamma(torch.tensor(a+1)) + torch.lgamma(torch.tensor(b+1)) - torch.lgamma(torch.tensor(a+b+2)))
         raise ValueError(f"Invalid a={a} or b={b} for precomputed factorials.")
    return (factorials[int(a)] * factorials[int(b)]) / factorials[int(a + b + 1)]

def sparse_matrix_power(sparse_mat, power):
    """ Calculates sparse_mat^power using repeated sparse matrix multiplication """
    if power == 0:
        # Return sparse identity matrix
        n = sparse_mat.shape[0]
        indices = torch.arange(n).unsqueeze(0).repeat(2, 1).to(sparse_mat.device)
        values = torch.ones(n).to(sparse_mat.device)
        return torch.sparse_coo_tensor(indices, values, sparse_mat.shape)
    if power == 1:
        return sparse_mat
    if power % 2 == 0:
        half_pow = sparse_matrix_power(sparse_mat, power // 2)
        return torch.sparse.mm(half_pow, half_pow)
    else:
        return torch.sparse.mm(sparse_mat, sparse_matrix_power(sparse_mat, power - 1))

# --- 2. Model Definition ---

class GAN_BWGNN_HAD(nn.Module):
    def __init__(self, in_channels, hidden_channels, C_order, num_nodes, heads=1):
        super(GAN_BWGNN_HAD, self).__init__()
        self.C_order = C_order
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels

        # Graph Attention Layer (Eq 1-5)
        # Using PyG's GATConv for simplicity. Output dim = hidden_channels
        self.gat_conv = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.0) # Concat=False if heads=1 or avg

        # MLP after GAT (used in Eq 19)
        self.mlp1 = nn.Linear(hidden_channels, hidden_channels)

        # Beta Wavelet Filters (Eq 14, 15)
        self.beta_wavelets = nn.ModuleList()
        self.beta_constants = []
        identity_sparse = self._create_sparse_identity(num_nodes)
        for i in range(C_order + 1):
            a = i
            b = C_order - i
            # Calculate constant 1 / (2 * B(a+1, b+1))
            beta_const = 1.0 / (2.0 * beta_function_val(a + 1, b + 1))
            self.beta_constants.append(beta_const)
            # Store a, b for polynomial calculation in forward pass
            self.beta_wavelets.append(nn.Identity()) # Placeholder, calculation is in forward

        # Aggregation and Final MLP (Eq 20, 21)
        # Aggregation is concatenation, so input dim = hidden_channels * (C_order + 1)
        self.mlp2 = nn.Linear(hidden_channels * (C_order + 1), 1) # Output is 1 (anomaly probability)

    def _create_sparse_identity(self, size):
        indices = torch.arange(size).unsqueeze(0).repeat(2, 1)
        values = torch.ones(size)
        return torch.sparse_coo_tensor(indices, values, (size, size)).to(device)

    def calculate_beta_polynomial(self, L_sparse_tensor, a, b, I_sparse):
        """ Calculates (L/2)^a * (I - L/2)^b efficiently """
        # Note: Need sparse matrix multiplication: torch.sparse.mm
        L_half = L_sparse_tensor * 0.5
        I_minus_L_half = I_sparse - L_half

        term1 = sparse_matrix_power(L_half, a)
        term2 = sparse_matrix_power(I_minus_L_half, b)

        # Element-wise multiplication of sparse matrices isn't directly supported.
        # A common GNN approach is to apply the polynomial filter to features: P(L)X
        # Instead of calculating the matrix W_ab, we calculate W_ab @ Features
        # This avoids forming the potentially dense polynomial matrix.
        # However, Eq 19 applies MLP *before* the wavelet.
        # If we interpret W_ab(MLP(h')) as applying the filter *operator* to the result
        # of MLP(h'), we can do this without forming W_ab explicitly.

        # Let's try calculating the operator application P(L)X = (L/2)^a (I-L/2)^b X

        # Efficient calculation P(L)X using repeated mat-vec products
        # Calculate (I - L/2)^b X first
        term2_X = self.mlp_output # Start with MLP output
        for _ in range(b):
            term2_X = torch.sparse.mm(I_minus_L_half, term2_X)

        # Then calculate (L/2)^a [ (I - L/2)^b X ]
        term1_term2_X = term2_X
        for _ in range(a):
            term1_term2_X = torch.sparse.mm(L_half, term1_term2_X)

        return term1_term2_X


    def forward(self, data, L_sparse_tensor):
        x, edge_index = data.x, data.edge_index

        # 1. Spatial Processing (GAN)
        # GATConv applies linear transform, attention, aggregation, activation
        h_prime = self.gat_conv(x, edge_index) # Output shape (N, hidden_channels)
        h_prime = F.relu(h_prime) # Apply activation after GAT

        # 2. MLP before Wavelets
        self.mlp_output = self.mlp1(h_prime) # Shape (N, hidden_channels)
        self.mlp_output = F.relu(self.mlp_output)

        # 3. Frequency Processing (BWGNN - Parallel Filters)
        Z_list = []
        I_sparse = self._create_sparse_identity(self.num_nodes)
        for i in range(self.C_order + 1):
            a = i
            b = self.C_order - i
            beta_const = self.beta_constants[i]

            # Calculate W_ab(MLP(h')) = beta_const * P(L) * MLP(h')
            # P(L)X calculation
            filtered_features = self.calculate_beta_polynomial(L_sparse_tensor, a, b, I_sparse)

            Z_i = beta_const * filtered_features # Apply constant scaler
            Z_list.append(Z_i)

        # 4. Aggregation (Concatenation) Eq 20
        S = torch.cat(Z_list, dim=1) # Shape (N, hidden_channels * (C_order + 1))

        # 5. Final MLP & Output Eq 21
        # Output logits for BCEWithLogitsLoss
        logits = self.mlp2(S).squeeze(-1) # Shape (N,)

        return logits # Return logits, sigmoid applied in loss or evaluation

# --- 3. Weighted Loss Function ---
def calculate_weights(labels):
    """ Calculate weights for BCE loss based on class imbalance. """
    n_total = len(labels)
    n_pos = labels.sum()
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return None # No weighting needed if only one class
    # Weight for positive class (anomalies) - inversely proportional to frequency
    # pos_weight = n_neg / n_pos # Standard way
    # Paper uses gamma = ratio of anomaly to normal? Let's use standard inverse freq
    pos_weight = torch.tensor(n_neg / n_pos, device=labels.device, dtype=torch.float32)
    print(f"Calculated pos_weight for BCE: {pos_weight.item():.4f}")
    return pos_weight

# --- 4. Training Function ---
def train(model, data, L_sparse, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    logits = model(data, L_sparse)
    loss = criterion(logits, data.y) # BCEWithLogitsLoss
    loss.backward()
    optimizer.step()
    return loss.item()

# --- 5. Inference Function ---
@torch.no_grad() # Disable gradient calculations for inference
def predict(model, data, L_sparse):
    model.eval()
    logits = model(data, L_sparse)
    probabilities = torch.sigmoid(logits) # Apply sigmoid to get probabilities
    return probabilities.cpu().numpy() # Return probabilities as numpy array

# --- 6. Main Execution ---
print("\n--- Model Initialization & Training ---")

# Hyperparameters
hidden_dim = 128 # From paper [cite: 2172]
beta_order_C = 2 # From paper (C=2 or C=3 optimal) [cite: 2173, 3773]
learning_rate = 0.01 # From paper [cite: 2171]
epochs = 200 # From paper [cite: 2172]
# Training ratio from paper (60%) [cite: 2172] - here we use all data as GNNs usually train on the full graph
# weight_decay = 5e-4 # Common for GNNs, not mentioned in paper

# Initialize Model
model = GAN_BWGNN_HAD(
    in_channels=D,
    hidden_channels=hidden_dim,
    C_order=beta_order_C,
    num_nodes=N
).to(device)

print(model)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # weight_decay=weight_decay)

# Loss Criterion (using pos_weight for imbalance as in Eq 22 )
pos_w = calculate_weights(graph_data.y)
# Use BCEWithLogitsLoss which includes sigmoid and is numerically stable
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

# --- Training Loop ---
start_train = time.time()
print("Starting training...")
losses = []
for epoch in range(1, epochs + 1):
    loss = train(model, graph_data, L_sparse_tensor, optimizer, criterion)
    losses.append(loss)
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}')
end_train = time.time()
training_time = end_train - start_train
print(f"Training finished in {training_time:.2f} seconds.")

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'GAN-BWGNN Training Loss ({dataset_basename})')
plt.legend()
plt.grid(True)
save_plot(plt.gcf(), dataset_basename, "GAN-BWGNN", f"TrainingLoss_C{beta_order_C}_k{k_neighbors}")


# --- 7. Inference and Evaluation ---
print("\n--- Inference and Evaluation ---")
start_infer = time.time()
anomaly_probabilities_flat = predict(model, graph_data, L_sparse_tensor)
end_infer = time.time()
inference_time = end_infer - start_infer
print(f"Inference finished in {inference_time:.2f} seconds.")

anomaly_map = anomaly_probabilities_flat.reshape((H, W))

# Calculate Metrics (reuse functions from previous RPCA code)
auc_score = roc_auc_score(gt_flat, anomaly_probabilities_flat) if n_anomaly_gt > 0 and n_background_gt > 0 else np.nan
scores_norm_flat = normalize_scores(anomaly_probabilities_flat) # Normalize for SER/AER
ser_score = calculate_ser(scores_norm_flat, gt_flat)
aer_score = calculate_aer(anomaly_probabilities_flat, gt_flat) # Use original scores for AER thresholds

metrics_results = {
    'GAN-BWGNN': {
        'AUC': auc_score, 'SER': ser_score, 'AER': aer_score,
        'Time_Train': training_time, 'Time_Infer': inference_time,
        'Scores': anomaly_map, 'Params': {'C': beta_order_C, 'k': k_neighbors, 'hidden': hidden_dim}
    }
}
print(f"GAN-BWGNN Metrics: AUC={auc_score:.4f}, SER={ser_score:.4f}, AER={aer_score:.4f}, Train Time={training_time:.2f}s, Infer Time={inference_time:.2f}s")


# --- 8. Save Plots & Summary ---

# Anomaly Score Map
fig_map, ax_map = plt.subplots(1, 2, figsize=(12, 6))
im0 = ax_map[0].imshow(ground_truth, cmap='gray')
ax_map[0].set_title('Ground Truth')
ax_map[0].set_xticks([])
ax_map[0].set_yticks([])
fig_map.colorbar(im0, ax=ax_map[0], fraction=0.046, pad=0.04)

im1 = ax_map[1].imshow(anomaly_map, cmap='hot')
ax_map[1].set_title(f"GAN-BWGNN Probabilities (AUC={auc_score:.4f})")
ax_map[1].set_xticks([])
ax_map[1].set_yticks([])
fig_map.colorbar(im1, ax=ax_map[1], fraction=0.046, pad=0.04)
plt.tight_layout()
save_plot(fig_map, dataset_basename, "GAN-BWGNN", "ScoreMap", metrics_results['GAN-BWGNN'])

# ROC Curve
fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
if n_anomaly_gt > 0 and n_background_gt > 0:
    fpr, tpr, thresholds_roc = roc_curve(gt_flat, anomaly_probabilities_flat)
    ax_roc.plot(fpr, tpr, lw=2, label=f"GAN-BWGNN (AUC = {auc_score:.4f})")
else:
    ax_roc.plot([0],[0], marker='x', linestyle='None', label=f"GAN-BWGNN (AUC = N/A)")

ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')
ax_roc.set_xlim([-0.01, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate (FPR) [0-1]')
ax_roc.set_ylabel('True Positive Rate (TPR) [0-1]')
ax_roc.set_title(f'ROC Curve - {dataset_basename}')
ax_roc.legend(loc="lower right")
ax_roc.grid(True)
save_plot(fig_roc, dataset_basename, "GAN-BWGNN", "ROC", metrics_results['GAN-BWGNN'])


if n_anomaly_gt > 0 and n_background_gt > 0 and len(tpr) > 1:
    youden_j = tpr - fpr # Calculate Youden's J statistic
    optimal_idx = np.argmax(youden_j) # Find index of max J
    optimal_threshold = thresholds_roc[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]

    print(f"\nOptimal Threshold (Max Youden's J) for GAN-BWGNN: {optimal_threshold:.4f}")
    print(f"  -> Corresponding TPR: {optimal_tpr:.4f}")
    print(f"  -> Corresponding FPR: {optimal_fpr:.4f}")

    # Generate prediction map using this threshold
    pred_map_optimal = (anomaly_map > optimal_threshold).astype(int)

    # Plot and Save Prediction Map
    fig_pred, ax_pred = plt.subplots(1, 2, figsize=(12, 6))
    im0 = ax_pred[0].imshow(ground_truth, cmap='gray')
    ax_pred[0].set_title('Ground Truth')
    ax_pred[0].set_xticks([])
    ax_pred[0].set_yticks([])
    fig_pred.colorbar(im0, ax=ax_pred[0], fraction=0.046, pad=0.04)

    im1 = ax_pred[1].imshow(pred_map_optimal, cmap='gray')
    ax_pred[1].set_title(f'GAN-BWGNN Prediction (Thresh={optimal_threshold:.2f})')
    ax_pred[1].set_xticks([])
    ax_pred[1].set_yticks([])
    # No colorbar needed for binary map
    plt.tight_layout()
    # Pass the metrics dict to include AUC in filename if desired
    save_plot(fig_pred, dataset_basename, "GAN-BWGNN", "PredictionMap_YoudenJ", metrics_results['GAN-BWGNN'])

else:
    print("\nSkipping Optimal Threshold calculation and Prediction Map due to invalid ROC data.")


# --- Save Metrics Summary ---
summary_filepath = os.path.join(output_dir, f"{dataset_basename}_GAN-BWGNN_summary.txt")
with open(summary_filepath, 'w') as f:
    f.write(f"GAN-BWGNN HAD Results Summary for Dataset: {dataset_basename}\n")
    f.write(f"Timestamp: {time.ctime()}\n")
    f.write("="*40 + "\n\n")
    metrics = metrics_results['GAN-BWGNN']
    params = metrics['Params']
    f.write(f"Model: GAN-BWGNN\n")
    f.write(f"  Parameters:\n")
    f.write(f"    Beta Wavelet Order (C): {params['C']}\n")
    f.write(f"    KNN Neighbors (k): {params['k']}\n")
    f.write(f"    Hidden Dimension: {params['hidden']}\n")
    f.write(f"    Learning Rate: {learning_rate}\n")
    f.write(f"    Epochs: {epochs}\n")
    f.write(f"  Metrics:\n")
    f.write(f"    AUC: {metrics.get('AUC', 'N/A'):.4f}\n")
    f.write(f"    SER: {metrics.get('SER', 'N/A'):.4f}\n")
    f.write(f"    AER: {metrics.get('AER', 'N/A'):.4f}\n")
    f.write(f"    Training Time (s): {metrics.get('Time_Train', 'N/A'):.2f}\n")
    f.write(f"    Inference Time (s): {metrics.get('Time_Infer', 'N/A'):.2f}\n")
    # --- MODIFIED LINE ---
    f.write(f"  Optimal Threshold (Max Youden's J): {optimal_threshold:.4f} (TPR={optimal_tpr:.4f}, FPR={optimal_fpr:.4f})\n" if not np.isnan(optimal_threshold) else "  Optimal Threshold (Max Youden's J): N/A\n")
    f.write("-" * 20 + "\n")

print(f"\nMetrics summary saved to: {summary_filepath}")
print("Script finished.")