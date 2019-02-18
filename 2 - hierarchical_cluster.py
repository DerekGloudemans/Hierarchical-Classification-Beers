
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
# Please note - my implementation of the clustering is alarmingly simlar to scipy's implementation
# However, I came up with this imlementation organically - I didn't look at any implementations until it came time to plot results
# I did use the code from scipy.cluster.hierarchy.dendrogram to plot my result because I didn't want to fool around with manual plot encoding


import numpy as np
import _pickle as cPickle
from scipy.cluster.hierarchy import dendrogram

# load pickled data
f = open("beer_data_pickle.cpkl", 'rb')
x_original, names, labels, beer_list = cPickle.load(f)
f.close()   

# normalize x and weight each feature so min is 0, max is 1
x_from_min = x_original-x_original.min(axis=0) 
x = x_from_min/ x_from_min.max(axis=0)

# weight each feature to give it significance in determining similarity
weights = np.ones(np.size(x,1))
weights[0:5] = 5 # weights quantified variables
weights[5:122] = 1 #weights short names
weights[122:135] = 0.5 # weights styles
weights[135:] = 0.25
x = np.multiply(x,weights)


# filter out some data
idx = []
downsample_ratio = 100

#include = ['blue moon', 'yazoo', 'coors', 'miller', 'abita', 'new belgium', 'good people', 'distihl', 'laguanitas', 'dogfish', 'smith and lentz', 'black abbey']
include = ['yazoo']
for i in range(0,len(names)):
    if i % (downsample_ratio +1)  == 0:
        idx.append(i)
    else:    
        for item in include:
            if item in names[i].lower():
                idx.append(i)
                break
x = x[idx,:]
x_orig =x
names = [names[i] for i in idx]

#intialize node_list
# each entry stores ((child nodes) or None, distance between children, num of beers represeneted by node)
node_list = []
for i in range (0,len(x)):
    node_list.append((None, 0,1))

# this will be added and subtracted from to keep track of new and removed nodes
active = [i for i in range(0,len(x))]
    
# calculate Euclidean distance between each pair of beers
dist = np.zeros([np.size(x,0),np.size(x,0)])
for i in active:
    for j in active:
        if i == j:
            dist[i,j] = np.Infinity
        else:
            dist[i,j] = np.sqrt(np.sum(np.square(x[i,:] - x[j,:])))
        
    if i%100 == 0 and False:
        print("On row {} of {}".format(i,np.size(dist,0)))
        
# at each step:
while len(active) > 1:
    print("Current number of active clusters: {}".format(len(active)))
    #select min non-zero distance
    min_arg = np.argmin(dist)
    i = min_arg // len(dist)
    j = min_arg % len(dist)
    
    #create new node that is weighted average of these locations and add row and column to dist
    node_list.append(((i,j),dist[i,j], node_list[i][2] + node_list[j][2]))
    
    # add new idx to index list
    active.append(len(node_list)-1)
    # remove old indices from index list
    active.remove(i)
    active.remove(j)
    
    #set row and column i and j (removed this round) to inf (so they don't get picked again)
    dist[i,:] = np.Infinity
    dist[:,i] = np.Infinity
    dist[j,:] = np.Infinity
    dist[:,j] = np.Infinity 

    
    
    # append new node weighted avg to end of x
    new_node_x = np.asmatrix(np.divide((np.multiply(x[i,:],node_list[i][2]) + np.multiply(x[j,:],node_list[j][2])), \
                           (node_list[i][2]+ node_list[j][2])))
    x = np.concatenate((x,new_node_x),0)
    
    # calculate distance to all other nodes (reflect across diagonal)
    # add row and column to dist, then fill in (technically only need to fill in indices in active)
    new_row = np.zeros([1, np.size(dist,1)])
    dist = np.concatenate((dist,new_row),0)
    new_col = np.zeros([np.size(dist,0), 1])
    dist = np.concatenate((dist,new_col),1)
    
    new_idx = len(node_list)-1
    for k in range(0,new_idx+1):
        if k in active:
            if k == new_idx:
                dist[new_idx,k] = np.Infinity
            else:
                dist[k,new_idx] = np.sqrt(np.sum(np.square(x[k,:] - x[new_idx,:])))
                dist[new_idx,k] = np.sqrt(np.sum(np.square(x[k,:] - x[new_idx,:])))
        else:
            dist[k,new_idx] = np.Infinity
            dist[new_idx,k] = np.Infinity

# convert list structure into array structure for compatibility with scipy plotting
hierarchy = np.zeros([len(x_orig)-1,4])
      
for i in range(0,len(hierarchy)):
    item = node_list[i+len(x_orig)]
    if item[0] != None:
        hierarchy[i,0] = item[0][0]
        hierarchy[i,1] = item[0][1]
        hierarchy[i,2] = item[1]
        hierarchy[i,3] = item[2]
        
dendrogram(hierarchy)