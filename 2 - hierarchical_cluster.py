
import numpy as np
import _pickle as cPickle

# Load pickled data
f = open("beer_data_pickle.cpkl", 'rb')
x, names, labels, beer_list = cPickle.load(f)
f.close()   

