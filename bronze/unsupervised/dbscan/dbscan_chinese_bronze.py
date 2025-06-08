import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN # Import DBSCAN
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('bronze.csv')
print("Original DataFrame shape:", df.shape) # Print original DataFrame shape

# Select all chemical element columns and 'GROUP' label for analysis
chemical_elements = ["Cu", "Sn", "Pb", "Zn", "Au", "Ag", "As", "Sb"]
df_selected = df[chemical_elements + ["GROUP"]].copy()
print("Shape of DataFrame with selected columns:", df_selected.shape) # Print shape of DataFrame with selected columns

# Extract features for clustering (all chemical elements)
features_for_clustering = df_selected[chemical_elements]

# Handle missing values: fill NaNs with the mean of each numerical column
features_for_clustering.fillna(features_for_clustering.mean(), inplace=True)
df_selected[chemical_elements] = features_for_clustering

# Scale feature data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)

# Perform PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(scaled_features)

# Prepare DataFrame for visualization
plot_df = pd.DataFrame(reduced_data_pca, columns=['PCA_Component_1', 'PCA_Component_2'])
plot_df['Original_GROUP'] = df_selected['GROUP']

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.20, min_samples=5)
cluster_dbscan = dbscan.fit(reduced_data_pca)
plot_df['DBSCAN_Label'] = cluster_dbscan.labels_

print("\nDBSCAN Cluster Label Counts:\n", pd.Series(plot_df['DBSCAN_Label']).value_counts())

# DBSCAN Clustering Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Visualize DBSCAN Clustering Results
sns.scatterplot(
    x='PCA_Component_1', y='PCA_Component_2',
    hue='DBSCAN_Label', palette='tab10',
    data=plot_df, s=100, alpha=0.7, edgecolor='w',
    ax=axes[0]
)
# Highlight noise points (-1 label in DBSCAN)
noise_points_dbscan = plot_df[plot_df['DBSCAN_Label'] == -1]
axes[0].scatter(noise_points_dbscan['PCA_Component_1'], noise_points_dbscan['PCA_Component_2'],
            color='black', marker='x', s=50, label='Noise Points (-1)', alpha=0.6)
axes[0].set_title('DBSCAN Clustering (eps=0.20, min_samples=5)')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].legend(title='DBSCAN Cluster')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Visualize Original 'GROUP' Labels (for comparison)
sns.scatterplot(
    x='PCA_Component_1', y='PCA_Component_2',
    hue='Original_GROUP', palette='tab10',
    data=plot_df, s=100, alpha=0.7, edgecolor='w',
    ax=axes[1]
)
axes[1].set_title('Original Region Labels (for comparison)')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].legend(title='Original Group', bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend position
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
