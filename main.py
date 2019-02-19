# Main code, from which other functions are called

#import files and packages
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import Utilfn



#def get_neighbors():

#def scatter_plot(hierarchy, level = 1):
#    R = dendrogram(hierarchy, color_threshold = 1)


#define user parameters here or do it interactively

#check if pickle file already exists
pickle_name = "beer_data_pickle.cpkl"
f= Path(pickle_name)
if not(f.is_file()):
    Utilfn.pickle_beer_data(pickle_name)

# load data    
x, names, labels, beer_list = Utilfn.load_data(pickle_name)
del pickle_name

#prep data for hierarchical clustering
wt = [10,0.15,0,0.15]
ds = 5000
include = ['blue moon', 'yazoo', 'coors', 'miller', 'laguanitas', 'dogfish'] 
x,names = Utilfn.prep_data(x,names,wt,include,ds)
del wt,ds

#get hierarchical clustering
hierarchy = Utilfn.hierarchical_cluster(x)




# plot dendrogram with interactive coloring
Utilfn.plot_dendrogram(hierarchy,names)

#get beer_cloud (sorted by closest)
#possibly - condense clusters
#plot 2d space cloud with interactive coloring based on hierarchical clustering colors

R = dendrogram(hierarchy, color_threshold = 1)
# I think these colors correspond to the nodes, not the leaves
colors = R['color_list']
order = R['leaves']
color_labels = [x for _,x in sorted(zip(order,colors))]


pca = PCA(n_components=2)
x_red = pca.fit_transform(x)
x_red = (x_red - x_red.min(axis=0))
x_red = x_red/x_red.max(axis = 0)
#plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize= (10,10))
ax.scatter(x_red[:,0],x_red[:,1])
#for i in range(0,len(names)):
#    plt.text(x_red[i,0],x_red[i,1],names[i],fontsize = 12)
ax.axis('off')