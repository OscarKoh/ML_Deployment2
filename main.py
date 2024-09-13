from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import pandas as pd

# Use the best UMAP and clustering parameters from previous steps
best_n_neighbors = best_params[0]
best_min_dist = best_params[1]
n_components = 10  # Set this to the best number of components

# Apply UMAP with the best parameters
umap_model = umap.UMAP(n_neighbors=best_n_neighbors, min_dist=best_min_dist, n_components=n_components, random_state=42)
umap_transformed_data = umap_model.fit_transform(X_scaled)

# Define the parameter grid for K-Means fine-tuning (without n_clusters, since it's fixed at 2)
param_grid = {
    'init': ['k-means++', 'random'],  # Initialization methods
    'max_iter': [300, 500, 1000],  # Maximum number of iterations
}

# Initialize a list to store the results
results = []

# Loop through each combination of K-Means parameters
for params in ParameterGrid(param_grid):
    # Apply K-Means with the current parameters and fixed n_clusters=2
    kmeans = KMeans(n_clusters=2, init=params['init'], max_iter=params['max_iter'], random_state=42)
    kmeans_labels = kmeans.fit_predict(umap_transformed_data)
    
    # Calculate evaluation metrics
    silhouette_avg = silhouette_score(umap_transformed_data, kmeans_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, kmeans_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, kmeans_labels)
    
    # Store the results
    results.append({
        'params': params,
        'silhouette_score': silhouette_avg,
        'calinski_harabasz_score': calinski_harabasz_avg,
        'davies_bouldin_score': davies_bouldin_avg
    })
    
    print(f"Params: {params}, Silhouette Score: {silhouette_avg}, CH Index: {calinski_harabasz_avg}, DB Index: {davies_bouldin_avg}")

# Convert results to a DataFrame for easy analysis
results_df = pd.DataFrame(results)

# Sort the results by Silhouette Score to find the best combination
best_result = results_df.loc[results_df['silhouette_score'].idxmax()]
print("Best Parameters based on Silhouette Score:")
print(best_result)

# Apply K-Means with the best parameters and fixed n_clusters=2
best_kmeans = KMeans(n_clusters=2, 
                     init=best_result['params']['init'], 
                     max_iter=best_result['params']['max_iter'], 
                     random_state=42)
best_kmeans_labels = best_kmeans.fit_predict(umap_transformed_data)

# Final evaluation metrics
final_silhouette_avg = silhouette_score(umap_transformed_data, best_kmeans_labels)
final_calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, best_kmeans_labels)
final_davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, best_kmeans_labels)

# Print final results
print(f"Final Silhouette Score: {final_silhouette_avg}")
print(f"Final Calinski-Harabasz Index: {final_calinski_harabasz_avg}")
print(f"Final Davies-Bouldin Index: {final_davies_bouldin_avg}")

# Plot the final clustering results
plt.figure(figsize=(8, 6))
plt.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=best_kmeans_labels, cmap='viridis', s=50)
plt.title(f"K-Means Clustering with 2 Clusters\n"
          f"Silhouette Score = {final_silhouette_avg:.2f}\n"
          f"CH Index = {final_calinski_harabasz_avg:.2f}\n"
          f"DB Index = {final_davies_bouldin_avg:.2f}")
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.grid(True)
plt.show()
