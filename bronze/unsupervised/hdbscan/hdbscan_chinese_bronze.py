import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv('bronze.csv')
print(df.shape) # Print initial DataFrame shape

# Select columns for analysis: main bronze components, additional metals (Pb, Sb, Ag), and GROUP label
df = df[["Cu", "Sn", "Pb", "Zn", "Sb", "Ag", "Au", "GROUP"]]
print(df.shape) # Print DataFrame shape after column selection

# Extract features for clustering
features_for_clustering = df[["Cu", "Zn", "Sn", "Sb", "Au", "Pb", "Ag"]]

# Handle missing values: fill NaNs with the mean of each numerical column
features_for_clustering.fillna(features_for_clustering.mean(), inplace=True)

# Reflect processed features back into the main DataFrame
df[["Au", "Cu", "Sn", "Pb", "Ag", "Sb", "Zn"]] = features_for_clustering

# Scale feature data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)

# Perform PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(scaled_features)

# Perform HDBSCAN clustering
hdb = hdbscan.HDBSCAN(min_cluster_size=4, gen_min_span_tree=True)
cluster = hdb.fit(reduced_data_pca)
hdbscan_labels = cluster.labels_

print("\nHDBSCAN Cluster Label Counts:\n", pd.Series(hdbscan_labels).value_counts()) # Print counts for each HDBSCAN cluster label
print("\nOriginal Group Label Counts:\n", df['GROUP'].value_counts()) # Print counts for each original group label

# Prepare DataFrame for visualization
plot_df = pd.DataFrame(reduced_data_pca, columns=['PCA_Component_1', 'PCA_Component_2'])
plot_df['HDBSCAN_Label'] = hdbscan_labels
plot_df['Original_GROUP'] = df['GROUP']

# --- Visualization ---

# Create figure for two subplots
plt.figure(figsize=(16, 7))

# Subplot 1: HDBSCAN Clustering Results
plt.subplot(1, 2, 1)
sns.scatterplot(
    x='PCA_Component_1',
    y='PCA_Component_2',
    hue='HDBSCAN_Label',
    palette='viridis',
    data=plot_df,
    s=100,
    alpha=0.7,
    edgecolor='w'
)
# Highlight noise points (-1 label in HDBSCAN)
noise_points_hdbscan = plot_df[plot_df['HDBSCAN_Label'] == -1]
plt.scatter(noise_points_hdbscan['PCA_Component_1'], noise_points_hdbscan['PCA_Component_2'],
            color='black', marker='x', s=50, label='Noise Points (-1)', alpha=0.6)

plt.title('HDBSCAN Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='HDBSCAN Cluster')
plt.grid(True, linestyle='--', alpha=0.6)


# Subplot 2: Original 'GROUP' Labels
plt.subplot(1, 2, 2)

sns.scatterplot(
    x='PCA_Component_1',
    y='PCA_Component_2',
    hue='Original_GROUP',
    palette='tab10',
    data=plot_df,
    s=100,
    alpha=0.7,
    edgecolor='w'
)
plt.title('Original Region Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Original Group', bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend position
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout() # Adjust subplots to prevent overlap
plt.show()
