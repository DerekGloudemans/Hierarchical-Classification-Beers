
import numpy as np
import _pickle as cPickle

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
downsample_ratio = 10000

include = ['blue moon', 'yazhoo', 'coors', 'miller', 'abita', 'new belgium', 'good people', 'distihl', 'laguanitas', 'dogfish', 'smith and lentz', 'black abbey']
for i in range(0,len(names)):
    if i % downsample_ratio  == 0:
        idx.append(i)
    else:    
        for item in include:
            if item in names[i].lower():
                idx.append(i)
                break
x = x[idx,:]
names = [names[i] for i in idx]


# calculate Euclidean distance between each pair of beers
dist = np.zeros([np.size(x,0),np.size(x,0)])
for i in range(0,np.size(dist,0)):
    for j in range(0,np.size(dist,1)):
        dist[i,j] = np.sqrt(sum(np.square(x[i,:] - x[j,:])))
        
    if i%100 == 0:
        print("On row {} of {}".format(i,np.size(dist,0)))