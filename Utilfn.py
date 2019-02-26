# This file contains all utility functions used for this assignment
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
import csv
import re
from ete3 import Tree, TreeStyle


#------------------------ pickle_beer_data_brewery_DB ------------------------#
# This function loads the Excel csv file of breweryDB data and parses it into a
# matrix of items x features, as well as an item name list and a label name list
def pickle_beer_data_breweryDB(save_file, beer_file='raw_data.csv', \
                     brewery_file = 'brewery-brewdb.csv', tag_file = 'tag_inclusion_list.csv' ):
    
    # Load data from csv file
    beer_list = []
    file1 = open(beer_file, encoding = 'utf-8')
    csv_reader = csv.reader(file1,delimiter = ',')
    
    for item in csv_reader:
        beer = {}
        beer["name"] = item[0]
        beer["brewery"] = item[2]
        beer["abv"] = item[1]
        beer["ibu"] = item[4]
        beer["fgmin"] = item[3]
        beer["og"] = item[6]
        beer["srm"] = item[9]
        beer["short_name"] = item[8]
        beer["style"] = item[11]
        beer["description"] = item[12]
        beer_list.append(beer)
    
    file1.close()  
    del beer_list[0] # tags    
    print("All beers appended to beer list")
    
    # Remove items with incomplete data
    complete_beer_list = []
    for i in range(0, len(beer_list)): 
        complete = True
        item = beer_list[i]
        for key in item.keys():
            if item.get(key) == '':
                complete = False 
                break
        if complete:
            complete_beer_list.append(item)
            
    # gives counts for every word in descriptions
    word_lists = []
    # Note: I copied this script to convert a string into a list of words from
    # https://stackoverflow.com/questions/6181763/converting-a-string-to-a-list-of-words
    for item in beer_list:
        text = item['description']
        words = re.sub("[^\w]", " ",  text).split()
        word_lists.append(words)
    
    # makes list of overall words
    all_words = []
    for item in word_lists:
        for word in item:
            word = word.lower()
            #word = word.translate(None, string.punctuation)
            all_words.append(word)
    
    if False:        
        # get counts of all words
        word_counts = Counter(all_words)
        #sort by most common
        temp = word_counts.most_common()
    # here the excel file is completed to correspond to these tags
    
    # load tags
    file2 = open(tag_file, encoding = 'utf-8')
    csv_reader = csv.reader(file2,delimiter = ',')
    
    tag_list = []
    for item in csv_reader:
        if item[2] != '0':
            tag_list.append(item[0])
    file2.close()
    
    # Assign tags to each beer for which they are relevant
    for i in range(0,len(complete_beer_list)):
        beer_tags = []
        for item in word_lists[i]:
            if item in tag_list:
                beer_tags.append(item)
        complete_beer_list[i]['tags'] = beer_tags
     
    # Replace brewery ID with brewery
    brewery_dict = {}
    file1 = open(brewery_file, encoding = 'utf-8')
    csv_reader = csv.reader(file1,delimiter = ',')   
    for item in csv_reader:
        id = item[2]
        name = item[3]
        brewery_dict[id] = name
        
    for i in range(0,len(complete_beer_list)):
        item = complete_beer_list[i]
        if item["brewery"] in brewery_dict.keys():
            complete_beer_list[i]["brewery"] = brewery_dict[item["brewery"]]   
        else:
            complete_beer_list[i]["brewery"] = "UNKNOWN"
    
    #overwrite with new data
    beer_list = complete_beer_list
    
    # create x_labels
    names = []
    for item in beer_list:
        names.append("{}  ---- {}".format(item['name'],item['brewery']))
    
    # create feature labels
    all_tags = []
    for item in beer_list:
        for tag in item['tags']:
            all_tags.append(tag)
    tag_labels = sorted(list(set(all_tags)))
    
    # create x
    x1 = np.zeros([len(beer_list),len(tag_labels)])    
    for i in range(0,len(beer_list)):
        for item in beer_list[i]['tags']:
            idx = tag_labels.index(item)
            x1[i,idx] = 1
    
    # add additional values    
    additional_labels = ['abv', 'ibu', 'srm', 'og', 'fgmin']        
    x2 = np.zeros([len(beer_list),5])
    for i in range(0, len(beer_list)):
        item = beer_list[i]
        x2[i,0] = float(item['abv'])
        x2[i,1] = float(item['ibu'])
        x2[i,2] = float(item['srm'])
        x2[i,3] = float(item['og'])
        x2[i,4] = float(item['fgmin'])
    
    #short_name_labels
    short_name_labels = []
    style_labels = []
    for item in beer_list:
        short_name_labels.append(item['short_name'])
        style_labels.append(item['style'])
    
    short_name_labels = sorted(list(set(short_name_labels)))
    style_labels = sorted(list(set(style_labels)))
    
    # create x3 and x4
    x3 = np.zeros([len(beer_list),len(short_name_labels)])    
    x4 = np.zeros([len(beer_list),len(style_labels)]) 
    for i in range(0,len(beer_list)):
        item = beer_list[i]
        idx = short_name_labels.index(item['short_name'])
        idx2 = style_labels.index(item['style'])
        x3[i,idx] = 1
        x4[i,idx2] = 1
        
    # compile all features
    x = np.concatenate((x2,x3,x4,x1),1)
    labels = additional_labels + style_labels + short_name_labels + tag_labels
        
    # Save data    
    f = open(save_file, 'wb')
    cPickle.dump((x,names,labels,beer_list),f)
    f.close()    
    print("Data pickled.")


#------------------------ pickle_beer_data_kaggle ------------------------#
# This function loads the Excel csv file of the Kaggle Craft beers data and 
# parses it into a matrix of items x features, as well as an item name list 
# and a label name list
def pickle_kaggle_data():
    # Load data from csv file
    beer_list = []
    file1 = open("beers-kaggle.csv", encoding = 'utf-8')
    csv_reader = csv.reader(file1,delimiter = ',')
    
    for item in csv_reader:
        beer = {}
        beer["abv"] = item[0]
        beer["ibu"] = item[1]
        beer["style"] = item[2]
        beer["name"] = item[3]
    
        beer_list.append(beer)
    
    file1.close()  
    del beer_list[0] # tags    
    print("All beers appended to beer list")
    
    # create x_labels
    names = []
    for item in beer_list:
        names.append(item['name'])
        
    #short_name_labels
    style_labels = []
    for item in beer_list:
        style_labels.append(item['style'])
    
    style_labels = sorted(list(set(style_labels)))
    
    # create x1
    x1 = np.zeros([len(beer_list),2])
    for i in range(0,len(beer_list)):
        x1[i,0] = beer_list[i]["abv"]
        x1[i,1] = beer_list[i]["ibu"]
        
    # create x2
    x2 = np.zeros([len(beer_list),len(style_labels)]) 
    for i in range(0,len(beer_list)):
        item = beer_list[i]
        idx = style_labels.index(item['style'])
        x2[i,idx] = 1        
        
        # compile all features
    x = np.concatenate((x1,x2),1)
    labels = ["abv", "ibu"] + style_labels 
        
    # Save data    
    f = open("beer_data_pickle_kaggle.cpkl", 'wb')
    cPickle.dump((x,names,labels,beer_list),f)
    f.close()    
    print("Data pickled.")


#---------------------------------- load_data --------------------------------#
# This function loads data from pickle file specified as input
# returns:
# x - numpy array of item features 
# names - list of names for each item (beer)
# labels - list of labels for each feature
# beer_list - list of beers, passed as legacy data and not used further
def load_data(file_name):
    
    f = open(file_name, 'rb')
    x_original, names, labels, beer_list = cPickle.load(f)
    f.close()  
    
    print("Data loaded from pickle file.")
    return x_original, names, labels, beer_list


#---------------------------------- load_data --------------------------------#
# This function reduces the size of x by the specified ratio (but includes all)
# items with a name that contain a string in include, and also normalizes and 
# weights each feature
# returns:
# x - numpy array of item normalized and weighted features for reduced dataset
# names - list of names for each item (beer) for reduced dataset
def prep_data(x,names, feat_weights, include=[], downsample = 100):
    # normalize x and weight each feature so min is 0, max is 1
    x_from_min = x-x.min(axis=0) 
    x = x_from_min/ x_from_min.max(axis=0)
    
    if len(x[1]) >100:
        # weight each feature to give it significance in determining similarity
        weights = np.ones(np.size(x,1))
        weights[0:5] = feat_weights[0] # weights quantified variables
        weights[5:122] = feat_weights[1] #weights short names
        weights[122:135] = feat_weights[2] # weights styles
        weights[135:] = feat_weights[3]
        x = np.multiply(x,weights)
    
    
    
    # filter out some data
    idx = []
        
    for i in range(0,len(names)):
        if (i+1) % (downsample)  == 0:
            idx.append(i)
        else:    
            for item in include:
                if item in names[i].lower():
                    idx.append(i)
                    break
    x = x[idx,:]
    names = [names[i] for i in idx]
    
    print("Data reduced to {} items for clustering.".format(len(names)))
    return x,names


#------------------------- hierarchical cluster ------------------------------#
# This function clusters data hierarchically via a bottom-up approach and
# Euclidean distance metric
# returns:
# x_mod - numpy array of item features and new node features
# node_list - list of each node in the structure. Each node contains the values:
    # (subnode1 if any, subnode2 if any, distance between nodes,
    # number of items in cluster)
# hierarchy - numopy array of the same information as in node_list, except only
    # branch nodes are included (i.e. no leaf, single item nodes)
def hierarchical_cluster(x, verbose = False):
    
    #store initial number of items
    num_items = len(x)
    
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
            
        if i%100 == 0 and verbose:
            print("Calculating distance for row {} of {}".format(i,np.size(dist,0)))
            
    # at each step:
    while len(active) > 1:
        if verbose:
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
    hierarchy = np.zeros([num_items-1,4])
          
    for i in range(0,len(hierarchy)):
        item = node_list[i+num_items]
        if item[0] != None:
            hierarchy[i,0] = item[0][0]
            hierarchy[i,1] = item[0][1]
            hierarchy[i,2] = item[1]
            hierarchy[i,3] = item[2]
            
    print("Hierarchical clustering complete.")
    return hierarchy, node_list, x


#------------------------- hierarchical cluster ------------------------------#
# This function clusters data hierarchically via a bottom-up approach and
# Euclidean distance metric, creating a balanced tree structure
# returns:
# x_mod - numpy array of item features and new node features
# node_list - list of each node in the structure. Each node contains the values:
    # (subnode1 if any, subnode2 if any, distance between nodes,
    # number of items in cluster)
# hierarchy - numopy array of the same information as in node_list, except only
    # branch nodes are included (i.e. no leaf, single item nodes)
def hierarchical_cluster_balanced(x,group_dist = 5, verbose = False):
    
    #store initial number of items
    num_items = len(x)
    
    #intialize node_list
    # each entry stores ((child nodes) or None, distance between children, num of beers represeneted by node)
    node_list = []
    for i in range (0,len(x)):
        node_list.append((None, 0,1))
    
    # this will be added and subtracted from to keep track of new and removed nodes
    active = [i for i in range(0,len(x))]
    #  when a node is created, it is added to next_active and it's constituent nodes are removed
    # at the end of the iteration, next_active becomes active
    next_active = []    
    
    # calculate Euclidean distance between each pair of beers
    dist = np.zeros([np.size(x,0),np.size(x,0)])
    for i in active:
        for j in active:
            if i == j:
                dist[i,j] = np.Infinity
            else:
                dist[i,j] = np.sqrt(np.sum(np.square(x[i,:] - x[j,:])))
            
        if i%100 == 0 and verbose:
            print("Calculating distance for row {} of {}".format(i,np.size(dist,0)))
            
    # at each step:
    while len(active) > 1:
        if verbose:
            print("Current number of active clusters: {}".format(len(active)))
        
        #select min non-zero distance - need to select only from active
        dist_copy = np.ones([np.size(dist,0),np.size(dist,1)])*np.Infinity
        for i in active:
            for j in active:
                    dist_copy[i,j] = dist[i,j]
                    
        min_arg = np.argmin(dist_copy)
        i = min_arg // len(dist)
        j = min_arg % len(dist)
        
        #create new node that is weighted average of these locations and add row and column to dist
        node_list.append(((i,j),dist[i,j], node_list[i][2] + node_list[j][2]))
        
        # add new idx to index list
        next_active.append(len(node_list)-1)
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
        
        #problem is here 
        new_idx = len(node_list)-1
        for k in range(0,new_idx+1):
            #if k in next_active:
                if k == new_idx:
                    dist[new_idx,k] = np.Infinity
                else:
                    dist[k,new_idx] = np.sqrt(np.sum(np.square(x[k,:] - x[new_idx,:])))
                    dist[new_idx,k] = np.sqrt(np.sum(np.square(x[k,:] - x[new_idx,:])))

    
        #check if active is of length 1, if so, reset next_active
        if len(active) < 2:
            active = next_active + active
            next_active = []
    
    # convert list structure into array structure for compatibility with scipy plotting
    hierarchy = np.zeros([num_items-1,4])
          
    for i in range(0,len(hierarchy)):
        item = node_list[i+num_items]
        if item[0] != None:
            hierarchy[i,0] = item[0][0]
            hierarchy[i,1] = item[0][1]
            hierarchy[i,2] = item[1]
            hierarchy[i,3] = item[2]
           
    print("Balanced hierarchical clustering complete.")
    return hierarchy, node_list, x


#-------------------------- get_cluster_items --------------------------------#
# This function finds all items that are in a cluster (node) or its constituent nodes
# returns:
# item_lists - list of lists, where list item i is a list of all items (beers)
    # contained within cluster (node) i in node_list 
def get_cluster_items(node_list):
    item_lists = []
    for item in node_list:
        items = []
        if item[0] != None:
            items.append(item[0][0])
            items.append(item[0][1]) 
            for subitem in item_lists[item[0][1]]:
                items.append(subitem)
            for subitem in item_lists[item[0][0]]:
                items.append(subitem)
        item_lists.append(sorted(list(set(items))))
        
    return item_lists


#---------------------------- get_avg_dist  ----------------------------------#
# This function finds the average distance from each item in a cluster to the 
# centroid of the cluster
# returns:
# avg_dist - a list of distances, where item i in the lists corresponds to the 
    #average intracluster distance to centroid for cluster i in node_list
def get_avg_dist(cluster_lists,x_mod):

    avg_dist = np.zeros([len(cluster_lists)])
    x = x_mod
    
    for i in range(0,len(cluster_lists)):
        sum_dist = 0
        for item in cluster_lists[i]:
            sum_dist = sum_dist + np.sqrt(np.sum(np.square(x[i,:] - x[item,:])))
        avg_dist[i] = sum_dist
    return avg_dist




#--------------------------- plot_avg_dist  ----------------------------------#
# This function plots the average intracluster distance for one or optionally 2
# heiarchical structures as created by hierarchical_cluster
# returns:
# avg_dists1 - result of call to get_avg_dist for hierarchy 1
    #avg_dists2 - result of call to get_avg_dist for hierarchy 1, if input
def plot_avg_dist(names,dists1, hierarchy1, dists2= None, hierarchy2 = None, num_inputs = 1):

    avg_dists1 = []
    active = [i for i in range(0,len(names))]
    next_cluster = 0
    while len(active) > 1:
        dist = np.average(dists1[active])
        avg_dists1.append(dist)
        
        #add next_cluster to active and remove its constituent cluster
        active.append(next_cluster+len(names))
        active.remove(int(hierarchy1[next_cluster,0]))
        active.remove(int(hierarchy1[next_cluster,1]))
        next_cluster = next_cluster + 1
    dist = dists1[active[0]]
    avg_dists1.append(dist)
    
    if True or num_inputs == 2:
        avg_dists2 = []
        active = [i for i in range(0,len(names))]
        next_cluster = 0
        while len(active) > 1:
            dist = np.average(dists2[active])
            avg_dists2.append(dist)
            
            #add next_cluster to active and remove its constituent cluster
            active.append(next_cluster+len(names))
            active.remove(int(hierarchy2[next_cluster,0]))
            active.remove(int(hierarchy2[next_cluster,1]))
            next_cluster = next_cluster + 1
        dist = dists2[active[0]]
        avg_dists2.append(dist)
    
    ax = [i for i in range(1,len(avg_dists1)+1)]
    ax.reverse()
    
    plt.figure(figsize = (10,10))
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 18})
    plt.plot(ax,avg_dists1)
    
    if num_inputs ==2:
        plt.plot(ax,avg_dists2)
        plt.xlabel('Number of clusters')
        plt.ylabel('Avg. distance within clusters')
        plt.legend(['Unbalanced', 'Balanced'])
        

    
    if num_inputs ==2:
        return avg_dists1,avg_dists2    
    return avg_dists1



#----------------------- get_most_representatives ----------------------------#
# This function returns the index (in x) of the most representative item (i.e.
# the item closest to the cluster centroid) for each cluster in the clustering
# returns:
# most_rep - a list of indices, where each item i in the list represents the most
    # representative item (beer) for cluster i in node_list or cluster_lists
def get_most_representatives(cluster_lists,names,x_mod):
    most_rep = []
    for i in range(0,len(cluster_lists)):
        if len(cluster_lists[i]) == 0:
            most_rep.append((i,0))
        else:
            min_dist = np.Infinity
            min_item = None
            for item in cluster_lists[i]:
                if item < len(names):
                    dist = np.sqrt(np.sum(np.square(x_mod[i,:] - x_mod[item,:])))
                    if dist < min_dist:
                        min_item = item
                        min_dist = dist
            most_rep.append((min_item,min_dist))
    return most_rep

#------------------------ augment_cluster_list -------------------------------#
# This function adds to each node from node_)list the item name, replacing characters
# incompatible with the Newick Tree format. This function is a precursor to 
# conversion into Newick Tree format
# returns:
# aug_cluster_list- same as node_list, but with an additional field that holds 
    # item name at the end of each item tuple
def augment_cluster_list(cluster_lists,cl,x_mod,names):
    reps = get_most_representatives(cluster_lists,names,x_mod)
    
    aug = []
    for item in reps:
        name = names[item[0]]
        name = name.replace("(","-")
        name = name.replace(")","-")
        name = name.replace(":","-")
        
        if item[1] == 0:
            aug.append(name)
        else:
            aug.append("Archetype={}".format(name))
    #add item names to cluster_list as an additional element
            
    aug_cluster_list = []
    for i in range (0,len(cl)):
        aug_cluster_list.append((cl[i][0],cl[i][1],cl[i][2],aug[i]))
    
    return aug_cluster_list


#------------------------ convert_to_newick -------------------------------#
# Recursively converts an aug_cluster_list representation of a tree into 
# Newick Tree format. Note that the string ":0;" must be added to the output
# as this could not be easily added inside the function due to recursion
#returns:
#out - string newick representation of hierarchical clustering
def convert_to_newick(aug_cluster_list,cluster_num):
    
    cluster = aug_cluster_list[cluster_num]
    
    #if no children (i.e. leaf node): output name
    if cluster[0] == None:
        out = cluster[3]
    else:
        # "({recurse on child }:distance to child1,{recurse on child2}:distance to child2)item_name" - this should be what it represents
        out = "({}:{},{}:{}){}"\
        .format(convert_to_newick(aug_cluster_list,cluster[0][0]),cluster[1],\
                convert_to_newick(aug_cluster_list,cluster[0][1]),cluster[1],cluster[3])

    return out

#------------------------ convert_to_newick_db -------------------------------#
# Recursively converts an aug_cluster_list representation of a tree into 
# Newick Tree format. if db is specified, this indicates the maximum number of 
# branches that will be expanded before making all additional subnodes direct
# children of the node. Note that the string ":0;" must be added to the output
# as this could not be easily added inside the function due to recursion
#returns:
#out - string newick representation of depth-bounded hierarchical clustering
def convert_to_newick_db(aug_cluster_list,cluster_num,db=100):
    
    cluster = aug_cluster_list[cluster_num]
    
    #if no children (i.e. leaf node): output name
    if cluster[0] == None:
        out = cluster[3]
    
    # reached user-defined depth bound
    elif db == 0:
        # now we need to condense all further nodes as branches from this node
        out = "{},{}"\
        .format(convert_to_newick_db(aug_cluster_list,cluster[0][0],0),\
                convert_to_newick_db(aug_cluster_list,cluster[0][1],0))
    else:
        # "({recurse on child }:distance to child1,{recurse on child2}:distance to child2)item_name" - this should be what it represents
        out = "({}:{},{}:{})"\
        .format(convert_to_newick_db(aug_cluster_list,cluster[0][0],db-1),cluster[1],\
                convert_to_newick_db(aug_cluster_list,cluster[0][1],db-1),cluster[1])

    return out


#---------------------------- plot_dendrogram --------------------------------#
# Plots a dendrogram representation of the hierarchical clustering specified by
# hierarchy, coloring items up to group_dist from one another in the same color
# dim - specifies size of figure
def plot_dendrogram(hierarchy,cl,names,group_dist = 5, dim = (60,120)):
    
    # extends lines to clean plot
    for i in range(0,len(hierarchy)):
        hierarchy[i,2] = hierarchy[i,2]+max([cl[int(hierarchy[i,0])][2],cl[int(hierarchy[i,1])][2]])

    plt.figure(figsize =dim)
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 18})
    
    #use this line to plot in new window - %matplotlib auto
    settings = {'orientation': 'left',
                'truncate_mode': None,
                'count_sort': 'ascending',
                'distance_sort': 'descending',
                'leaf_rotation': 0,
                'leaf_font_size': 25,
                'color_threshold':group_dist}
    dendrogram(hierarchy, leaf_label_func = (lambda n: names[n]),**settings)
    
    if False:
        plt.savefig("dendrogram.png", bbox_inches = 'tight', pad_inches = 0.5)
    

#----------------------------- plot_newick -----------------------------------#
# Plots a dendrogram representation of the hierarchical clustering specified by 
# aug_cluster_list, converting it into Newick Tree format first. Plots using the
# ete3 module (must be installed prior to use of this function).
# mode - 'c' or 'r' will plot as a circular or rectangular tree, respectively
# db - if -1, no depth bound is used, otherwise, specifies the depth bound for the plot
def plot_newick(aug_cluster_list,mode = 'c', db = -1):
    circular_style = TreeStyle()
    circular_style.mode = mode # draw tree in circular mode
    circular_style.scale = 20
    circular_style.arc_span = 360
    
    if db > 0:
        newick = convert_to_newick_db(aug_cluster_list,len(aug_cluster_list)-1,db)
        circular_style.mode = mode
    else:
        newick = convert_to_newick(aug_cluster_list,len(aug_cluster_list)-1)
    newick = newick + ':0;'
    
    t = Tree(newick,format = 1)
    t.show(tree_style=circular_style)
    if False:
        t.render('tree3.png',w=100, units = 'in', tree_style = circular_style)
