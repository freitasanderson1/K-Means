import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregar a base de dados
df = pd.read_csv("dataset.csv")

# Selecionar variáveis contínuas
features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
df_features = df[features]

# Normalizar os dados
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df_features)

# Aplicar K-Means para diferentes valores de K e calcular o coeficiente de silhueta
silhouette_scores = []
k_values = range(2, 11)  # Testar de 2 a 10 clusters

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_normalized)
    silhouette_avg = silhouette_score(df_normalized, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"K: {k}, Silhouette Score: {silhouette_avg:.4f}")

# Identificar o melhor K
best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
print(f"\nMelhor número de clusters: {best_k}")

# Reajustar o modelo com o melhor K
kmeans = KMeans(n_clusters=best_k, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_normalized)

# Visualizar a distribuição dos clusters
print("\nDistribuição dos clusters:")
print(df["Cluster"].value_counts())

# Análise de categorias após a clusterização
print("\nDistribuição de 'Region' por cluster:")
print(df.groupby("Cluster")["Region"].value_counts())

print("\nDistribuição de 'Channel' por cluster:")
print(df.groupby("Cluster")["Channel"].value_counts())

# Visualizar a pontuação de silhueta para diferentes K
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Scores para diferentes K")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid()
plt.show()
