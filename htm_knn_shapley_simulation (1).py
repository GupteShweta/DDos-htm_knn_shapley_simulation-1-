import numpy as np
import itertools
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# 1. Feature Definitions & Baseline Setup
FEATURE_NAMES = [
    'Src_IP_Entropy',
    'Dst_IP_Entropy',
    'Src_Port_Entropy',
    'Dst_Port_Entropy',
    'Pkt_Len_Entropy',
    'Avg_Pkt_Len',
    'Pkt_Count',
    'Protocol_Entropy'
]

N_FEATURES = len(FEATURE_NAMES)

# Normal/Benign baseline feature vector (normalized 0.0 to 1.0)
BENIGN_BASELINE = np.array([0.75, 0.75, 0.70, 0.70, 0.60, 0.40, 0.20, 0.50])

# Attack Profiles based on APSIPA 2021 HTM-KNN entropy trends
# SYN Flood: IP & Pkt_Len entropies decrease (-), Port entropies & Pkt_Count increase (+), Protocol entropy decreases (-)
SYN_FLOOD_PROFILE = np.array([0.20, 0.20, 0.90, 0.90, 0.15, 0.10, 0.95, 0.10])

# LDAP Flood: IP entropies decrease (-), Port & Pkt_Len entropies, Avg Pkt Len, Pkt Count, Protocol entropy all increase (+)
LDAP_FLOOD_PROFILE = np.array([0.25, 0.25, 0.85, 0.85, 0.85, 0.80, 0.90, 0.90])


# 2. HTM Semantic Encoder & Spatial Pooler Simulation
class HTMEncoder:
    """Encodes continuous feature vectors into Sparse Distributed Representations (SDR)."""
    def __init__(self, n_bits_per_feature=20, w=4):
        self.n_bits = n_bits_per_feature
        self.w = w  # active bits

    def encode_scalar(self, val):
        """Map scalar val in [0, 1] to a binary SDR chunk of length n_bits with w active bits."""
        val = np.clip(val, 0.0, 1.0)
        max_start = self.n_bits - self.w
        start_idx = int(round(val * max_start))
        sdr = np.zeros(self.n_bits, dtype=int)
        sdr[start_idx : start_idx + self.w] = 1
        return sdr

    def encode_vector(self, vec):
        """Concatenate SDR chunks for each feature into a full SDR."""
        sdr_chunks = [self.encode_scalar(v) for v in vec]
        return np.concatenate(sdr_chunks)


# 3. HTM-KNN Classifier Simulation
class HTMKNNClassifier:
    def __init__(self, encoder, tau=5.0):
        self.encoder = encoder
        self.tau = tau
        self.prototypes = {}

    def register_prototype(self, class_name, feature_vector):
        sdr = self.encoder.encode_vector(feature_vector)
        self.prototypes[class_name] = sdr

    def predict_proba(self, feature_vector, target_class):
        """Calculate KNN classification probability using SDR Hamming distance."""
        sdr_obs = self.encoder.encode_vector(feature_vector)
        distances = {}
        for c_name, sdr_proto in self.prototypes.items():
            # Raw overlap distance: XOR bit differences
            dist = np.sum(sdr_obs != sdr_proto)
            distances[c_name] = dist
        
        # Softmax over negative distances (KNN prototype similarity)
        classes = list(self.prototypes.keys())
        logits = np.array([-distances[c] / self.tau for c in classes])
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        target_idx = classes.index(target_class)
        return probs[target_idx]


# 4. Exact Shapley Value Calculation
def compute_shapley_values(classifier, input_vector, baseline_vector, target_class):
    """
    Computes exact Shapley values for all features.
    f(S) represents prediction confidence when features in S are active (input_vector)
    and features outside S are masked with baseline_vector.
    """
    n = len(input_vector)
    shapley_values = np.zeros(n)
    
    # Pre-calculate factorial weights for subset sizes
    fact_cache = {i: math.factorial(i) for i in range(n + 1)}
    
    # Iterate over all features
    for i in range(n):
        other_features = [j for j in range(n) if j != i]
        phi_i = 0.0
        
        # Iterate over all coalitions of other features
        for k in range(len(other_features) + 1):
            weight = (fact_cache[k] * fact_cache[n - k - 1]) / fact_cache[n]
            for subset in itertools.combinations(other_features, k):
                # Coalitions without feature i
                vec_without = baseline_vector.copy()
                for idx in subset:
                    vec_without[idx] = input_vector[idx]
                
                # Coalition with feature i
                vec_with = vec_without.copy()
                vec_with[i] = input_vector[i]
                
                # Marginal contribution
                val_with = classifier.predict_proba(vec_with, target_class)
                val_without = classifier.predict_proba(vec_without, target_class)
                
                phi_i += weight * (val_with - val_without)
                
        shapley_values[i] = phi_i
        
    return shapley_values


# 5. Main Simulation Execution
def run_simulation():
    encoder = HTMEncoder(n_bits_per_feature=20, w=4)
    model = HTMKNNClassifier(encoder, tau=4.0)
    
    # Register learned prototype patterns
    model.register_prototype('Benign', BENIGN_BASELINE)
    model.register_prototype('SYN Flood', SYN_FLOOD_PROFILE)
    model.register_prototype('LDAP Flood', LDAP_FLOOD_PROFILE)
    
    print("=== HTM-KNN SHAPLEY VALUE XAI SIMULATION ===")
    
    # Test Sample: Observed SYN Flood traffic sample
    sample_syn = np.array([0.18, 0.22, 0.92, 0.88, 0.12, 0.12, 0.98, 0.15])
    prob_syn = model.predict_proba(sample_syn, 'SYN Flood')
    shap_syn = compute_shapley_values(model, sample_syn, BENIGN_BASELINE, 'SYN Flood')
    
    print(f"\n[Sample 1] SYN Flood Detection Confidence: {prob_syn * 100:.2f}%")
    df_syn = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Observed Value': sample_syn,
        'Benign Baseline': BENIGN_BASELINE,
        'Shapley Value (phi)': shap_syn
    }).sort_values(by='Shapley Value (phi)', ascending=False)
    print(df_syn.to_string(index=False))
    
    # Test Sample: Observed LDAP Flood traffic sample
    sample_ldap = np.array([0.22, 0.28, 0.88, 0.82, 0.88, 0.82, 0.92, 0.92])
    prob_ldap = model.predict_proba(sample_ldap, 'LDAP Flood')
    shap_ldap = compute_shapley_values(model, sample_ldap, BENIGN_BASELINE, 'LDAP Flood')
    
    print(f"\n[Sample 2] LDAP Flood Detection Confidence: {prob_ldap * 100:.2f}%")
    df_ldap = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Observed Value': sample_ldap,
        'Benign Baseline': BENIGN_BASELINE,
        'Shapley Value (phi)': shap_ldap
    }).sort_values(by='Shapley Value (phi)', ascending=False)
    print(df_ldap.to_string(index=False))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # SYN Plot
    colors_syn = ['#d9534f' if v > 0 else '#5bc0de' for v in shap_syn]
    axes[0].barh(FEATURE_NAMES, shap_syn, color=colors_syn, edgecolor='black', alpha=0.85)
    axes[0].axvline(0, color='grey', linestyle='--', linewidth=1)
    axes[0].set_title(f'SYN Flood Attribution (Confidence: {prob_syn*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Shapley Marginal Value (phi_i)', fontsize=11)
    axes[0].grid(axis='x', linestyle=':', alpha=0.6)
    
    # LDAP Plot
    colors_ldap = ['#d9534f' if v > 0 else '#5bc0de' for v in shap_ldap]
    axes[1].barh(FEATURE_NAMES, shap_ldap, color=colors_ldap, edgecolor='black', alpha=0.85)
    axes[1].axvline(0, color='grey', linestyle='--', linewidth=1)
    axes[1].set_title(f'LDAP Flood Attribution (Confidence: {prob_ldap*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Shapley Marginal Value (phi_i)', fontsize=11)
    axes[1].grid(axis='x', linestyle=':', alpha=0.6)
    
    plt.suptitle('HTM-KNN Threat Detection: Real-Time Shapley Feature Attribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspace/scratch/htm_knn_shapley/shapley_attribution_plot.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to /workspace/scratch/htm_knn_shapley/shapley_attribution_plot.png")

if __name__ == '__main__':
    run_simulation()
