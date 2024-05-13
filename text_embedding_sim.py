import sklearn.cluster
from InstructorEmbedding import INSTRUCTOR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import torch
model = INSTRUCTOR('hkunlp/instructor-large')

sentences =[['Represent human activity sentence for clustering: ','the person is doing left hand bicep curl slowly.'],
            ['Represent human activity sentence for clustering: ','the person is wiping with right hand fast.'],
            ['Represent human activity sentence for clustering: ','the person is wiping with one hand slowly'],
            ['Represent human activity sentence for clustering: ','the person is wiping with one hand fast'],
            ['Represent human activity sentence for clustering: ','the person is wiping with one hand'],
            ['Represent human activity sentence for clustering: ','the person is wearing headset slowly.'],
            ['Represent human activity sentence for clustering: ','the person is wearing headset.'],
            ['Represent human activity sentence for clustering: ','the person is using telephone.'],
            ['Represent human activity sentence for clustering: ','the person is picking up phone.'],
            ['Represent human activity sentence for clustering: ','the person is openning and closing arms.'],
            ['Represent human activity sentence for clustering: ','the person is opening and clapping.'],
            ['Represent human activity sentence for clustering: ','the person is doing left hand bicep curl slowly.'],
            ['Represent human activity sentence for clustering: ','the person is doing left hand bicep curl fast.'],
            ['Represent human activity sentence for clustering: ','the person is doing left hand bicep curl.'],
            ['Represent human activity sentence for clustering: ','the person is doing chest press slowly.'],
            ['Represent human activity sentence for clustering: ','the person is doing chest press fast.'],
            ['Represent human activity sentence for clustering: ','the person is doing chest press by touching elbows.'],
            ['Represent human activity sentence for clustering: ','the person is doing chest press and shoulder press using both hands.'],
            ['Represent human activity sentence for clustering: ','the person is doing chest press and shoulder press slowly.'],
            ['Represent human activity sentence for clustering: ','the person is doing chest press and shoulder press fast.'],
            ['Represent human activity sentence for clustering: ','the person is doing chest press and shoulder press.'],
            ['Represent human activity sentence for clustering: ','the person is doing chest press.'],
            ['Represent human activity sentence for clustering: ','the person is doing bicep curls slowly.'],
            ['Represent human activity sentence for clustering: ','the person is doing bicep curls fast.'],
            ['Represent human activity sentence for clustering: ','the person is doing bicep curls.'],
            ['Represent human activity sentence for clustering: ','the person is doing bicep curl on both hands.'],
            ['Represent human activity sentence for clustering: ','the person is doing arm length clapping very slowly.'],
            ['Represent human activity sentence for clustering: ','the person is doing arm hold one one and shoulder press on other.'],
            ['Represent human activity sentence for clustering: ','the person is clapping very slowly.'],
            ['Represent human activity sentence for clustering: ','the person is boxing slowly.'],
            ['Represent human activity sentence for clustering: ','the person is boxing fast.'],
            ['Represent human activity sentence for clustering: ','the person is boxing.'],
            ['Represent human activity sentence for clustering: ','the person is answering call fast.'],
            ['Represent human activity sentence for clustering: ','the person is answering call.'],
            ['Represent human activity sentence for clustering: ','the person boxing with both hands.'],]

embeddings = model.encode(sentences)

sentences1 = [['Represent human activity sentence for clustering: ','Doing random activities.'],
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

embeddings1 = model.encode(sentences1)
#clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=2)
#clustering_model.fit(embeddings)
#cluster_assignment = clustering_model.labels_
#print(cluster_assignment)
#print(embeddings.shape)

#torch.save(embeddings, "simulated_embedding.pt")
arr1 = np.arange(35)
tsne = TSNE(n_components=2, perplexity=5, metric="cosine", init="random")  # Adjust perplexity here
embeddings_tsne = tsne.fit_transform(embeddings)

arr2 = np.arange(10)
tsne2 = TSNE(n_components=2, perplexity=5, metric="cosine", init="random")  # Adjust perplexity here
embeddings_tsne2 = tsne.fit_transform(embeddings1)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])#, c=arr1, cmap='viridis')
plt.scatter(embeddings_tsne2[:, 0], embeddings_tsne2[:, 1])#, c=arr1, cmap='viridis')

# Annotate each point with its corresponding label
for i, label in enumerate(arr1):
    plt.text(embeddings_tsne[i, 0], embeddings_tsne[i, 1], str(label), color='black', fontsize=8)

for i, label in enumerate(arr2):
    plt.text(embeddings_tsne2[i, 0], embeddings_tsne2[i, 1], str(label), color='black', fontsize=8)

plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
#plt.colorbar(label='Cluster')
plt.show()