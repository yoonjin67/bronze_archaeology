import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('bronze.csv')
print("Original DataFrame shape:", df.shape) # Original DataFrame shape:

# Select all chemical element columns and 'GROUP' label for analysis
chemical_elements = ["Cu", "Sn", "Pb", "Zn", "Au", "Ag", "As", "Sb"]
df_selected = df[chemical_elements + ["GROUP"]].copy()
print("Shape of DataFrame with selected columns:", df_selected.shape) # Shape of DataFrame with selected columns:
features_for_clustering = df_selected[chemical_elements]

# Handle missing values in feature data
features_for_clustering.fillna(features_for_clustering.mean(), inplace=True)

# Reflect processed feature data back into df_selected
df_selected[chemical_elements] = features_for_clustering

# Scale feature data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)

# Dimensionality reduction using PCA (Principal Component Analysis)
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(scaled_features)

# Prepare DataFrame for visualization
plot_df = pd.DataFrame(reduced_data_pca, columns=['PCA_Component_1', 'PCA_Component_2'])
plot_df['Original_GROUP'] = df_selected['GROUP'] # Add original GROUP labels

# Perform K-Means clustering (k=3, random_state for reproducibility, n_init to prevent local optima)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
cluster_kmeans = kmeans.fit(reduced_data_pca)
plot_df['KMeans_Label'] = cluster_kmeans.labels_
kmeans_centers = cluster_kmeans.cluster_centers_

print("\nK-Means Cluster Labels Count:\n", pd.Series(plot_df['KMeans_Label']).value_counts()) # K-Means Cluster Labels Count:

# --- K-Means Clustering Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7)) # 1 row, 2 columns grid

# 1. Visualize K-Means Clustering Results
sns.scatterplot(
    x='PCA_Component_1', y='PCA_Component_2',
    hue='KMeans_Label', palette='tab10',
    data=plot_df, s=100, alpha=0.7, edgecolor='w',
    ax=axes[0]
)
# Visualize K-Means cluster centers
axes[0].scatter(kmeans_centers[:, 0], kmeans_centers[:, 1],
            color='purple', marker='X', s=300, label='KMeans Centers', edgecolor='black', linewidth=2)
axes[0].set_title('K-Means Clustering (k=3)')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].legend(title='KMeans Cluster')
axes[0].grid(True, linestyle='--', alpha=0.6)

# 2. Visualize Original 'GROUP' Labels (for comparison)
sns.scatterplot(
    x='PCA_Component_1', y='PCA_Component_2',
    hue='Original_GROUP', palette='tab10',
    data=plot_df, s=100, alpha=0.7, edgecolor='w',
    ax=axes[1]
)
axes[1].set_title('Original Region Labels (for comparison)')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].legend(title='Original Group', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
