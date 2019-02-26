# Main code, from which other functions are called

#import files and packages
from pathlib import Path
import numpy as np
import Utilfn

# user parameters here
show = True
ds = 500 # controls the downsample ratio - if over 25000, only beers from include will be included
depth_bound = 20
mode = 'r'
#include = ['blue moon','yazoo','coors','miller', 'yuengling', 'laguanitas','dogfish', 'tennessee brew works', 'smith and lentz', 'smith & lentz', 'bearded iris', 'jackalope', 'little harpeth', 'southern grist', 'fat bottom', 'new heights', 'tailgate', 'black abbey', 'good people', 'distihl'] 
include = ['blue moon','yazoo','coors','miller', 'tennessee brew works', 'smith and lentz', 'smith & lentz', 'bearded iris', 'jackalope', 'southern grist', 'fat bottom','distihl'] 


# check if pickle file already exists
pickle_name = "beer_data_pickle.cpkl"
f= Path(pickle_name)
if not(f.is_file()):
    Utilfn.pickle_beer_data(pickle_name)

# load data    
x, names, labels, beer_list = Utilfn.load_data(pickle_name)
del pickle_name

# prep data for hierarchical clustering
wt = [10,0.15,0,0.15]
#include = ['blue moon', 'yazoo', 'coors', 'miller', 'laguanitas', 'dogfish']
x,names = Utilfn.prep_data(x,names,wt,include,ds)
group_dist= len(names)/10
del wt,ds

# get hierarchical clustering
hierarchy,cl,x_mod = Utilfn.hierarchical_cluster(x,verbose = True)
hierarchy2, cl2 ,x_mod2 = Utilfn.hierarchical_cluster_balanced(x)

    
# plot intracluster distances  
cluster_lists = Utilfn.get_cluster_items(cl)
cluster_lists2 = Utilfn.get_cluster_items(cl2)
dists1 = Utilfn.get_avg_dist(cluster_lists,x_mod)
dists2 = Utilfn.get_avg_dist(cluster_lists2,x_mod)


# plot dendrogram and circular dendrogram
aug_cluster_list = Utilfn.augment_cluster_list(cluster_lists,cl,x_mod,names)

if show:
    Utilfn.plot_avg_dist(names,dists1, hierarchy, dists2, hierarchy2 , num_inputs = 2)
    Utilfn.plot_newick(aug_cluster_list, mode, depth_bound)
    Utilfn.plot_dendrogram(hierarchy,cl, names, group_dist, dim = (40,80))
    
