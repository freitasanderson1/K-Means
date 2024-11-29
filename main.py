import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
df_features = df[features]

scaler = StandardScaler()
df_normalized = scaler.fit_transform(df_features)

silhouette_scores = []
k_values = range(2, 11) 

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_normalized)
    silhouette_avg = silhouette_score(df_normalized, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"K: {k}, Silhouette Score: {silhouette_avg:.4f}")

best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
print(f"\nMelhor número de clusters: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_normalized)

print("\nDistribuição dos clusters:")
print(df["Cluster"].value_counts())

print("\nDistribuição de 'Region' por cluster:")
print(df.groupby("Cluster")["Region"].value_counts())

print("\nDistribuição de 'Channel' por cluster:")
print(df.groupby("Cluster")["Channel"].value_counts())

plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Scores para diferentes K")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid()
plt.show()
