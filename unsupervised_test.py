#K-MEANS Örneği
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
#Denetimsiz Öğrenme: Öğretmen yok → Verileri benzerliklere göre gruplama.
#veri oluşturma:
X = np.random.rand(100,2) #100veri noktası, 2 özellik

#KMEANS modeli oluştur ve atama
kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(X)

# Küme merkezleri ve atama
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

#sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')  # Küme merkezleri
plt.title('K-Means Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()
