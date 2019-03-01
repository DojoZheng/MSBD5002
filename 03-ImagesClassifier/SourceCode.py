
# coding: utf-8

# # MSBD5002 Assignment 3: Unsupervised learning of Images Segmentation

# ## Preparation

# ### Import Packages & Files

# In[2]:


# Magic functions
# get_ipython().run_line_magic('matplotlib', 'inline')

import time
import os, os.path
import random
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# PIL
from IPython.display import display
from PIL import Image


# ### Load images

# In[3]:


# import images
def loadImageNames(dir_path):
    images_dict = []
    file_names = [file for file in os.listdir(dir_path)]
    file_names.sort(reverse=False)
    file_count = len(file_names)
    for name in file_names:
        images_dict.append({"names": name})
    
    df = pd.DataFrame(images_dict)
    return df


# In[5]:


img_dir_path = '../images'
img_names_df = loadImageNames(img_dir_path)
img_names_df.head()


# ## Features extraction by CovNets

# In[6]:


# Keras
import keras
from keras.preprocessing import image as KerasImage

def predictImages(model, size, preprocess, decode_pred):
    '''
    Description: This function is mainly for predicting the types of the images
    Parameters:
        @model: the CovNet model used for prediction
        @size:  the size of the image for the ConNet
        @preprocess: the specific 'preprocess_input' function of the CovNet
        @decode_pred: the specific 'decode_prediction' function of the CovNet
    '''
    preds_arr = []
    n = len(img_names_df)
    n = 10
    
    # time calculation
    start = time.time()
    
    for name in img_names_df['names'][:n]:
        # Load the image
        path = os.path.join(img_dir_path, name)
        img = KerasImage.load_img(path, target_size=(size, size))
        
        # Preprocess the image
        x = KerasImage.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess(x)

        preds = model.predict(x)
        
        # Add the prediction to dataframe
        row = []
        for result in decode_pred(preds, top=3)[0]:
            # append class name
            row.append(result[1])
            
            # append class prediction probability
            row.append(result[2])
            
        # Add row to the prediction array
        preds_arr.append(row)
    
    # total time
    end = time.time()
    print("Prediction time: {}".format(end - start))
    
     # transform to DataFrame
    indice = list(img_names_df['names'].str.replace(".jpg", ""))
    columns = ["class1", "P(class1)", "class2", "P(class2)", "class3", "P(class3)"]
    preds_df = pd.DataFrame(preds_arr, index=indice[:n], columns=columns)
    
    return preds_df


# In[8]:


def featuresExtraction(model, size, preprocess):
    '''
    Description: This function is mainly for features extraction of the images
    Parameters:
        @model: the CovNet model used for prediction
        @size:  the size of the image for the ConNet
        @preprocess: the specific 'preprocess_input' function of the CovNet
    '''
    features_arr = []
    n = len(img_names_df)
#     n = 10

    # time calculation
    start = time.time()
    
    for name in img_names_df['names'][:n]:
        # Load the image
        path = os.path.join(img_dir_path, name)
        img = KerasImage.load_img(path, target_size=(size, size))
        
        # Preprocess the image
        x = KerasImage.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess(x)

        # Extract features
        features = model.predict(x)
        
        # Flatten the array
        flat_features = features.flatten()
        
        # Add the features to array
        features_arr.append(flat_features)
        
    # get the total time
    end = time.time()
    total_minutes = float(end - start) / 60.
    print("Features extraction minutes: {}".format(total_minutes))
    
    # transform the array to dataframe
#     features_df = pd.DataFrame(features_arr)
    
    return features_arr


# ### VGG16

# In[86]:


# import modules
from keras.applications.vgg16 import decode_predictions as vgg16_decode
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.vgg16 import VGG16

# define the size
vgg16_size = 224


# In[28]:


# Predict the images' class
vgg16_predictor = VGG16(include_top=True, weights='imagenet')
vgg16_predictions = predictImages(model=vgg16_predictor,
                                   size=vgg16_size,
                                   preprocess=vgg16_preprocess,
                                   decode_pred=vgg16_decode)
display(vgg16_predictions)


# In[87]:


# Features extraction
vgg16_transformer = VGG16(weights="imagenet", include_top=False)


# In[ ]:


vgg16_features = featuresExtraction(vgg16_transformer, vgg16_size, vgg16_preprocess)


# ### ResNet50

# In[9]:


# import modules
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.resnet50 import decode_predictions as resnet_decode

# define the image size
resnet_size = 224


# In[10]:


# Predict the images' class
resnet_predictor = ResNet50(include_top=True, weights='imagenet')
resnet_predictions = predictImages(model=resnet_predictor,
                                   size=resnet_size,
                                   preprocess=resnet_preprocess,
                                   decode_pred=resnet_decode)
display(resnet_predictions)


# In[11]:


# Features extraction
resnet_transformer = ResNet50(weights="imagenet", include_top=False)


# In[83]:


resnet_features = featuresExtraction(resnet_transformer, resnet_size, resnet_preprocess)


# ### Xception

# In[14]:


# import modules
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.applications.xception import decode_predictions as xception_decode

# define the image size
xception_size = 299


# In[25]:


# Predict the images' class
xception_predictor = Xception(include_top=True, weights='imagenet')
xception_predictions = predictImages(model=xception_predictor,
                                   size=xception_size,
                                   preprocess=xception_preprocess,
                                   decode_pred=xception_decode)
display(xception_predictions)


# In[15]:


# Features extraction
xception_transformer = Xception(weights="imagenet", include_top=False)


# In[16]:


xception_features = featuresExtraction(xception_transformer, xception_size, xception_preprocess)
print(xception_features[0])


# ### InceptionResNetV2

# In[60]:


# import modules
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as inception_preprocess
from keras.applications.inception_resnet_v2 import decode_predictions as inception_decode

# define the image size
inception_size = 299


# In[61]:


# Predict the images' class
inception_predictor = InceptionResNetV2(include_top=True, weights='imagenet')
inception_predictions = predictImages(model=inception_predictor,
                                   size=inception_size,
                                   preprocess=inception_preprocess,
                                   decode_pred=inception_decode)
display(inception_predictions)


# In[ ]:


inception_transformer = InceptionResNetV2(weights="imagenet", include_top=False)


# In[73]:


inception_features = featuresExtraction(inception_transformer, inception_size, inception_preprocess)
display(inception_features.head())


# ## Dimensions Reduction
# 
# Since there are still many features after features extraction, I will try to apply PCA to reduce the dimensions.

# In[19]:


from sklearn.decomposition import PCA

def pca_transform(features_df, n_components=None):
    # PCA transformation
    pca = PCA(n_components=n_components, random_state=728)
    output = pca.fit_transform(features_df)
    
    # plot the cumulative explained variance plot
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    
    return pca, output


# ### InceptionResNetV2

# In[78]:


inception_pca, inception_pca_features = pca_transform(inception_features)
inception_pca_df = pd.DataFrame(inception_pca_features)
inception_pca_df.shape


# In[ ]:


# store to file
inception_pca_df.to_csv("inception_pca.csv", index=False)


# ### ResNet50

# In[93]:


resnet_pca, resnet_pca_features = pca_transform(resnet_features)
resnet_pca_df = pd.DataFrame(resnet_pca_features)
resnet_pca_df.shape


# ### VGG16

# In[94]:


vgg16_pca, vgg16_pca_features = pca_transform(vgg16_features)
vgg16_pca_df = pd.DataFrame(vgg16_pca_features)
vgg16_pca_df.shape


# ### Xception

# In[132]:


xception_pca, xception_pca_features = pca_transform(xception_features)
xception_pca_df = pd.DataFrame(xception_pca_features)
xception_pca_df.shape


# ## Clustering

# ### Find out the suitable CovNet & K value

# In[23]:


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def silhouetteScore(dimension, pca_data, k_range):
    # Select the top N features
    train_data = []
    for i in range(len(pca_data)):
        train_data.append(pca_data[i][:dimension])

    train_data = np.array(train_data)
    print(train_data.shape)

    # Get the silhouette score list
    silhouette_list = []
    print("\n### Using the method of KMeans:")
    for n_clusters in k_range:
        clusterer = KMeans(n_clusters=n_clusters, algorithm="elkan", random_state=0).fit(train_data)
        preds = clusterer.predict(train_data)
        centers = clusterer.cluster_centers_
        score = silhouette_score(train_data, preds)
        silhouette_list.append(score)
        print("n_clusters = {}, silhouette_score: {:.4f}".format(n_clusters, score))

    return silhouette_list


# In[ ]:


# Compute the silhouette score with different CovNets PCA features
k_range = range(10, 30, 2)
inception_silhouette_scores = silhouetteScore(3000, inception_pca_features, k_range)
resnet_silhouette_scores = silhouetteScore(3000, resnet_pca_features, k_range)
vgg16_silhouette_scores = silhouetteScore(3000, vgg16_pca_features, k_range)
xception_silhouette_scores = silhouetteScore(3000, xception_pca_features, k_range)


# In[176]:


# Credit: Josh Hemann

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_clusters = len(resnet_silhouette_scores)
silhouette_scores = [inception_silhouette_scores,
                    resnet_silhouette_scores,
                    vgg16_silhouette_scores,
                    xception_silhouette_scores]


fig, ax = plt.subplots(figsize=(15, 5))

index = np.arange(3, 4*n_clusters, 4)
bar_width = 0.7

opacity = 0.5
error_config = {'ecolor': '0.3'}

labels_list = ['InceptionResNetV2', 'ResNet50', 'VGG16', 'Xception']
colors = ['b', 'r', 'g', 'y']
means = [means_men, means_women]

for i, (score, color, label) in enumerate(zip(silhouette_scores, colors, labels_list)):
    rects1 = ax.bar(x=index + bar_width*i,
                    height=score,
                    width=bar_width,
                    alpha=opacity,
                    color=color,
                    error_kw=error_config,
                    label=label)


ax.set_xlabel('number of clusters')
ax.set_ylabel('Silhouette Scores')
ax.set_title('Silhouette Score by K & CovNets')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(list(k_range))
ax.legend()

# fig.tight_layout()
plt.show()


# ### ResNet50 with 16 clusters
# 
# From the above picture, it is quite obvious that **ResNet50** can lead to a relatively high silhouette score. Thus, we will choose the PCA Featreus from ResNet50 to make some prediction.

# #### Optimal K in Elbow Curve 

# In[107]:


from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

def draw_elbow_curve(dimension, pca_data, elbow_range):
    # define the pca data
    pca_data = inception_pca_features

    # define the training data
    train_data = []
    for i in range(len(pca_data)):
        train_data.append(pca_data[i][:dimension])
    train_data = np.array(train_data)

    distortions = []
    for k in elbow_range:
        print("======={}========".format(k))
        kmeanModel = KMeans(n_clusters=k, algorithm="elkan", random_state=0)
        kmeanModel.fit(train_data)
        distortions.append(sum(np.min(cdist(train_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / train_data.shape[0])

    # Plot the elbow
    plt.plot(elbow_range, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


# In[177]:


draw_elbow_curve(dimension=3000,
                 pca_data=resnet_pca_features,
                 elbow_range=k_range)


# From the elbow curve, it's quite obvious that **16** can be the elbow value. Thus, we will choose 16 clusters for prediction.

# #### KMeans Clustering

# In[242]:


def KMeansClustering(k, dimension, pca_data):
    # define the training data
    train_data = []
    for i in range(len(pca_data)):
        train_data.append(pca_data[i][:dimension])
    
    train_data = np.array(train_data)
    
    # make predictions
    model = KMeans(n_clusters=k, n_jobs=-1, algorithm='elkan', random_state=100)
    model.fit(train_data)
    clusters_pred = model.predict(train_data)
    
    return clusters_pred
    


# In[179]:


# make prediction with kmeans
kmeans_k = 16
kmeans_d = 3000
pca_data = resnet_pca_features
kmeans_pred = KMeansClustering(kmeans_k, kmeans_d, pca_data)


# In[220]:


def PredictionDict(pred_arr, k):
    '''
    Get the dictionary from the predictions array
    ''' 
    # store the predictions into dictionary
    clusters_dict = {}
    
    # initialize the Series for each cluster key
    for i in range(k):
        clusters_dict[i] = []

    #   Key: the index of K
    # Value: the index string of the image
    for i, k_index in enumerate(pred_arr):
        value = img_names_df['names'][i].replace('.jpg', '')
        clusters_dict[k_index].append(value)
        
    # Transfer the value into Series so the dictionary
    # can be transformed into DataFrame
    for key, value in clusters_dict.items():
        clusters_dict[key] = pd.Series(value)
    
    return clusters_dict
        


# In[ ]:


'''
Write the prediction result to file
'''
clusters_pred_dict = PredictionDict(kmeans_pred, kmeans_k)
clusters_pred_dict.keys()

clusters_cols_dict = {}
for i in range(kmeans_k):
    clusters_cols_dict[i] = 'cluster{}'.format(i+1)

clusters_pred_pd = pd.DataFrame.from_dict(clusters_pred_dict)
clusters_pred_pd = clusters_pred_pd.rename(clusters_cols_dict, axis='columns')


# In[237]:


clusters_pred_pd.to_csv('resnet_predictions.csv', index=False)
clusters_pred_pd.head()


# ### Xception with 24 Clusters
# 
# From the silhouette score distribution bar chart, it is quite obvious that Xception with 24 clusters have a sudden increase. Thus, I would try 24 clusters with PCA features from Xception.

# In[249]:


# make prediction with kmeans
kmeans_k = 24
kmeans_d = 3000
pca_data = xception_pca_features
xception_kmeans_pred = KMeansClustering(kmeans_k, kmeans_d, pca_data)


# In[264]:


'''
Write the prediction result to file
'''
clusters_pred_dict = PredictionDict(xception_kmeans_pred, kmeans_k)
clusters_cols_dict = {}
for i in range(kmeans_k):
    clusters_cols_dict[i] = 'cluster{}'.format(i+1)

clusters_pred_pd = pd.DataFrame.from_dict(clusters_pred_dict)
clusters_pred_pd = clusters_pred_pd.rename(clusters_cols_dict, axis='columns')


# In[ ]:


import math
xception_predictions_df = clusters_pred_pd.applymap(lambda x: '\'{}\''.format(x))
xception_predictions_df = xception_predictions_df.applymap(lambda x: '' if x=='\'nan\'' else x)
xception_predictions_df.to_csv("xception_predictions.csv", index=False)


# ### Comparision

# Finally, I try to judge whether these 2 models are good by my own eyes. The following results is what I label the clusters.
# 
# #### ResNet50 PCA Clustering with KMeans
# | Cluster | Label |
# | :-----: | :---: |
# | cluster1| plane |
# | cluster2| cat, dog |
# | cluster3| bus, train |
# | cluster4| **NOT CLEAR** |
# | cluster5| **NOT CLEAR** |
# | cluster6| restaurant, dinner |
# | cluster7| dogs, cows, sheep |
# | cluster8| human faces |
# | cluster9| bicycle, motorcycle |
# |cluster10| living room |
# |cluster11| people |
# |cluster12| car |
# |cluster13| birds |
# |cluster14| bird |
# |cluster15| horses |
# |cluster16| **NOT CLEAR** |
# 
# #### Xception PCA Clustering with KMeans
# | Cluster | Label |
# | :-----: | :---: |
# | cluster1| computers, work room |
# | cluster2| car(for common drive) |
# | cluster3| bus |
# | cluster4| human |
# | cluster5| horse |
# | cluster6| restaurant |
# | cluster7| train |
# | cluster8| motorcyle |
# | cluster9| living room |
# |cluster10| **NOT CLEAR**(ship, animals) |
# |cluster11| **NOT CLEAR** |
# |cluster12| bicycle |
# |cluster13| cat |
# |cluster14| sport car |
# |cluster15| human face |
# |cluster16| dog |
# |cluster17| car(street, park, road) |
# |cluster18| plane |
# |cluster19| sheep, cow |
# |cluster20| car(more fancy) |
# |cluster21| train |
# |cluster22| bird |
# |cluster23| motorcycle, bicycle |
# |cluster24| dog, animals |

# ## Conclusion

# From the above comparision, I would like to draw a conclusion that the clusters by Xception Features Extraction and KMeans clustering have a better images segmentations.

# In[311]:


xception_predictions_df.to_csv("A3_dzhengah_20546139_prediction.csv", index=False)

