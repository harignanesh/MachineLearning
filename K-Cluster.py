import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = make_blobs(n_samples =200,n_features = 2,centers =4,cluster_std=1.8,random_state=101)
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

KMeans_Model = KMeans(n_clusters=4)
KMeans_Model.fit(data[0])

fig, (ax1,ax2) = plt.subplots(1,2,sharey = True,figsize=(10,6))
ax1.set_title('K-Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=KMeans_Model.labels_,cmap="rainbow")
ax2.set_title('Original Data')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap="rainbow")
plt.show()
