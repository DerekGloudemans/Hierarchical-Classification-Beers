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
ds = 2000
group_dist= 15
include = ['blue moon', 'yazoo', 'coors', 'miller', 'laguanitas', 'dogfish'] 
x,names = Utilfn.prep_data(x,names,wt,include,ds)
del wt,ds

#get hierarchical clustering
hierarchy,cl,x_mod = Utilfn.hierarchical_cluster(x,verbose = True)
hierarchy2, cl2 ,x_mod2 = Utilfn.hierarchical_cluster_balanced(x,verbose=True)

#extends lines to clean plot
for i in range(0,len(hierarchy)):
    hierarchy[i,2] = hierarchy[i,2]+max([cl[int(hierarchy[i,0])][2],cl[int(hierarchy[i,1])][2]])
for i in range(0,len(hierarchy2)):
    hierarchy2[i,2] = hierarchy2[i,2]+max([cl2[int(hierarchy2[i,0])][2],cl2[int(hierarchy2[i,1])][2]])
    
# get intracluster distances  
cluster_lists = Utilfn.get_cluster_items(cl)
cluster_lists2 = Utilfn.get_cluster_items(cl2)
dists1 = Utilfn.get_avg_dist(cluster_lists,x_mod)
dists2 = Utilfn.get_avg_dist(cluster_lists2,x_mod)



temp, temp2 = Utilfn.plot_avg_dist(names,dists1, hierarchy, dists2, hierarchy2 , num_inputs = 2)


    


