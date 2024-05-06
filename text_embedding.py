import sklearn.cluster
from InstructorEmbedding import INSTRUCTOR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import torch
model = INSTRUCTOR('hkunlp/instructor-large')

sentences = [['Represent human activity sentence for clustering: ','Doing random activities.'],
             ['Represent human activity sentence for clustering: ','Boxing with both hands.'],
             ['Represent human activity sentence for clustering: ','Doing Biceps Curls.'],
             ['Represent human activity sentence for clustering: ',"Doing Chest Press."],
             ['Represent human activity sentence for clustering: ','Doing Shoulder and Chest Press.'],
             ['Represent human activity sentence for clustering: ','Doing Arm hold and Shoulder Press'],
             ['Represent human activity sentence for clustering: ','Arm Opener.'],
             ['Represent human activity sentence for clustering: ','Answering telephone.'],
             ['Represent human activity sentence for clustering: ','Wearing VR headsets.'],
             ['Represent human activity sentence for clustering: ','Sweeping table.'],
             ]

embeddings = model.encode(sentences)
#clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=2)
#clustering_model.fit(embeddings)
#cluster_assignment = clustering_model.labels_
#print(cluster_assignment)
print(embeddings.shape)

torch.save(embeddings, "original_embedding.pt")
arr1 = np.arange(10)
tsne = TSNE(n_components=2, perplexity=5, metric="cosine", init="random")  # Adjust perplexity here
embeddings_tsne = tsne.fit_transform(embeddings)
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=arr1, cmap='viridis')

# Annotate each point with its corresponding label
for i, label in enumerate(arr1):
    plt.text(embeddings_tsne[i, 0], embeddings_tsne[i, 1], str(label), color='black', fontsize=8)

plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster')
plt.show()