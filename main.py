# Main code, from which other functions are called

#import files and packages
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import Utilfn
import string


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
hierarchy,cl,x_mod = Utilfn.hierarchical_cluster(x,verbose = False)
hierarchy2, cl2 ,x_mod2 = Utilfn.hierarchical_cluster_balanced(x)

#extends lines to clean plot
for i in range(0,len(hierarchy)):
    hierarchy[i,2] = hierarchy[i,2]+max([cl[int(hierarchy[i,0])][2],cl[int(hierarchy[i,1])][2]])
for i in range(0,len(hierarchy2)):
    hierarchy2[i,2] = hierarchy2[i,2]+max([cl2[int(hierarchy2[i,0])][2],cl2[int(hierarchy2[i,1])][2]])
    
# plot intracluster distances  
cluster_lists = Utilfn.get_cluster_items(cl)
cluster_lists2 = Utilfn.get_cluster_items(cl2)
dists1 = Utilfn.get_avg_dist(cluster_lists,x_mod)
dists2 = Utilfn.get_avg_dist(cluster_lists2,x_mod)
Utilfn.plot_avg_dist(names,dists1, hierarchy, dists2, hierarchy2 , num_inputs = 2)

reps1 = Utilfn.get_most_representatives(cluster_lists,names,x_mod)

aug = []
for item in reps1:
    name = names[item[0]]
    name = name.replace("(","-")
    name = name.replace(")","-")
    
    if item[1] == 0:
        aug.append(name)
    else:
        aug.append("Archetype={}".format(name))
#add item names to cluster_list as an additional element
        
aug_cluster_list = []
for i in range (0,len(cl)):
    aug_cluster_list.append((cl[i][0],cl[i][1],cl[i][2],aug[i]))

# note distance isn't quite right now because it treats the distance to each node as the total distance between the two
    # need to save this alternate value
##convert to Newick tree format
temp = Utilfn.convert_to_newick(aug_cluster_list,len(aug_cluster_list)-1)
temp = temp + ":0;"
    
