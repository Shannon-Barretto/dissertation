import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14})

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])

# 1. First subplot: Conceptual explanation with KL divergence
ax1 = plt.subplot(gs[0, 0])

# Create two distributions
x = np.linspace(-5, 10, 1000)
p = stats.norm.pdf(x, 0, 1)  # Distribution P
q = stats.norm.pdf(x, 2, 1.5)  # Distribution Q

# Plot the distributions
ax1.plot(x, p, 'b-', linewidth=2, label='P: Original Data')
ax1.plot(x, q, 'r-', linewidth=2, label='Q: Synthetic Data')
ax1.set_ylim(0, 0.45)
ax1.set_xlim(-5, 10)
ax1.legend(loc='upper right')
ax1.set_title('KL Divergence: Measuring Distribution Differences')
ax1.set_ylabel('Probability Density')
ax1.set_xlabel('Data Value')

# Add explanatory text
kl_text = "KL(P||Q) measures how much information is lost\nwhen approximating P with Q"
ax1.text(4, 0.3, kl_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Add KL area visualization
for i in range(len(x)-1):
    if p[i] > q[i]:
        ax1.fill_between(x[i:i+2], p[i:i+2], q[i:i+2], alpha=0.3, color='blue')

# 2. Second subplot: JS divergence explanation
ax2 = plt.subplot(gs[0, 1])

# Create mixture distribution
m = (p + q) / 2

# Plot the distributions
ax2.plot(x, p, 'b-', linewidth=2, label='P: Original Data')
ax2.plot(x, q, 'r-', linewidth=2, label='Q: Synthetic Data')
ax2.plot(x, m, 'g--', linewidth=2, label='M: Mixture (P+Q)/2')
ax2.set_ylim(0, 0.45)
ax2.set_xlim(-5, 10)
ax2.legend(loc='upper right')
ax2.set_title('JS Divergence: Symmetric Alternative to KL')
ax2.set_ylabel('Probability Density')
ax2.set_xlabel('Data Value')

# Add explanatory text
js_text = "JS(P||Q) = ½KL(P||M) + ½KL(Q||M)\nwhere M = (P+Q)/2"
ax2.text(4, 0.3, js_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 3. Third subplot: Visualizing JS divergence values
ax3 = plt.subplot(gs[1, 0])

# Create a range of synthetic distributions with varying means and stds
means = np.linspace(0, 3, 5)
stds = np.linspace(1, 2, 5)

js_values = np.zeros((len(means), len(stds)))

# Calculate JS divergence
for i, mean in enumerate(means):
    for j, std in enumerate(stds):
        q_temp = stats.norm.pdf(x, mean, std)
        m_temp = (p + q_temp) / 2
        kl_p_m = np.sum(np.where(p > 0, p * np.log(p / m_temp), 0)) * (x[1] - x[0])
        kl_q_m = np.sum(np.where(q_temp > 0, q_temp * np.log(q_temp / m_temp), 0)) * (x[1] - x[0])
        js_values[i, j] = 0.5 * kl_p_m + 0.5 * kl_q_m

# Create heatmap
im = ax3.imshow(js_values, cmap='viridis', origin='lower')
plt.colorbar(im, ax=ax3, label='JS Divergence')
ax3.set_xticks(np.arange(len(stds)))
ax3.set_yticks(np.arange(len(means)))
ax3.set_xticklabels([f'σ={std:.2f}' for std in stds])
ax3.set_yticklabels([f'μ={mean:.2f}' for mean in means])
ax3.set_title('JS Divergence Values for Different Synthetic Distributions')
ax3.set_xlabel('Standard Deviation of Synthetic Data')
ax3.set_ylabel('Mean of Synthetic Data')

# 4. Fourth subplot: Real data example
ax4 = plt.subplot(gs[1, 1])

# Simulate binned data (like histograms)
np.random.seed(42)
bins = np.arange(10)
original_counts = np.random.poisson(lam=30, size=10)
original_probs = original_counts / original_counts.sum()

# Create three different synthetic datasets
synthetic1 = np.random.poisson(lam=30, size=10)
synthetic1_probs = synthetic1 / synthetic1.sum()

synthetic2 = np.random.poisson(lam=20, size=10)
synthetic2_probs = synthetic2 / synthetic2.sum()

synthetic3 = np.random.poisson(lam=10, size=10)
synthetic3_probs = synthetic3 / synthetic3.sum()

# Calculate JS divergence for each synthetic dataset
def calculate_js(p, q):
    m = (p + q) / 2
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    m = m + epsilon
    
    # Normalize again
    p = p / p.sum()
    q = q / q.sum()
    m = m / m.sum()
    
    kl_p_m = np.sum(p * np.log(p / m))
    kl_q_m = np.sum(q * np.log(q / m))
    return 0.5 * kl_p_m + 0.5 * kl_q_m

js1 = calculate_js(original_probs, synthetic1_probs)
js2 = calculate_js(original_probs, synthetic2_probs)
js3 = calculate_js(original_probs, synthetic3_probs)

# Bar chart
width = 0.2
ax4.bar(bins - width, original_probs, width, label='Original Data', color='blue')
ax4.bar(bins, synthetic1_probs, width, label=f'Synthetic 1 (JS={js1:.4f})', color='green')
ax4.bar(bins + width, synthetic2_probs, width, label=f'Synthetic 2 (JS={js2:.4f})', color='orange')
ax4.bar(bins + 2*width, synthetic3_probs, width, label=f'Synthetic 3 (JS={js3:.4f})', color='red')
ax4.set_xlabel('Feature Value (bin)')
ax4.set_ylabel('Probability')
ax4.set_title('Comparing Original vs. Synthetic Data Distributions')
ax4.legend(loc='upper right')

# 5. Fifth subplot (spans two columns): JS Divergence in the context of synthetic data evaluation
ax5 = plt.subplot(gs[2, :])

# Create a comparison of different metrics
methods = ['Gaussian', 'CTGAN', 'Copula', 'VAE', 'GAN']
js_scores = [0.08, 0.05, 0.03, 0.09, 0.07]
other_metric = [0.7, 0.82, 0.9, 0.65, 0.75]  # Could be some other quality metric

# Twin axes for comparing metrics
ax5a = ax5
ax5a.set_xlabel('Synthetic Data Generation Method')
ax5a.set_ylabel('JS Divergence (lower is better)')
ax5a.bar(methods, js_scores, color='blue', alpha=0.6, label='JS Divergence')
ax5a.set_ylim(0, 0.2)

ax5b = ax5a.twinx()
ax5b.set_ylabel('Data Utility Score (higher is better)')
ax5b.plot(methods, other_metric, 'ro-', linewidth=2, label='Data Utility')
ax5b.set_ylim(0, 1.0)

# Add annotations
for i, method in enumerate(methods):
    ax5a.annotate(f"{js_scores[i]:.3f}", 
                 xy=(i, js_scores[i] + 0.01), 
                 ha='center', va='bottom', 
                 color='blue', fontweight='bold')
    
    ax5b.annotate(f"{other_metric[i]:.2f}", 
                 xy=(i, other_metric[i] + 0.03), 
                 ha='center', va='bottom', 
                 color='red', fontweight='bold')

# Add explanation
explanation = """
JS Divergence for Synthetic Data Evaluation:
- Lower values (closer to 0) indicate synthetic distributions closer to original data
- Unlike KL divergence, JS is symmetric and bounded between 0 and 1
- JS = 0 means identical distributions
- Often used with other metrics to evaluate overall synthetic data quality
"""
ax5.text(0.5, -0.35, explanation, transform=ax5.transAxes, 
        ha='center', va='center', fontsize=12,
        bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.5'))

# Combine legends
lines1, labels1 = ax5a.get_legend_handles_labels()
lines2, labels2 = ax5b.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax5a.set_title('JS Divergence in Synthetic Data Generation Methods')

# Adjust layout
plt.tight_layout()
fig.suptitle('Understanding Jensen-Shannon (JS) Divergence for Synthetic Data Evaluation', 
             fontsize=18, y=0.98)
fig.subplots_adjust(top=0.94)

# Save the figure
plt.savefig('js_divergence_explanation.png', dpi=300, bbox_inches='tight')
plt.show()
