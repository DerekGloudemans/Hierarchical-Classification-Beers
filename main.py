# Main code, from which other functions are called

#use this line to plot in new window - %matplotlib auto

#import other files
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

import Utilfn


def plot_dendrogram(hierarchy,names,depth = 2):
    plt.figure(figsize = (10,20))
    #plt.style.use('seaborn-notebook')
    settings = {'orientation': 'left',
                'truncate_mode': None,
                'count_sort': 'ascending',
                'distance_sort': True,
                'leaf_rotation': 0,
                'leaf_font_size': 10,
                'color_threshold':1}
    dn = dendrogram(hierarchy, leaf_label_func = (lambda n: names[n]),**settings)
    return dn



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
wt = [10,0.15,.1,0.1]
ds = 5000000
include = ['blue moon', 'yazoo', 'coors', 'miller', 'laguanitas', 'dogfish'] 
x,names = Utilfn.prep_data(x,names,wt,include,ds)
del wt,ds

#get hierarchical clustering
hierarchy = Utilfn.hierarchical_cluster(x,verbose = True)

#possibly - condense clusters


# plot dendrogram with interactive coloring
dn = plot_dendrogram(hierarchy,names)

# plot 2d space cloud with interactive coloring