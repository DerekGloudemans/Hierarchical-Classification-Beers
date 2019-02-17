#load files from csv into python

# remove items with missing values

# replace brewery id number with brewery name

from collections import Counter
import numpy as np
import csv
import _pickle as cPickle
import re
import string


# Load data from csv file
beer_list = []
file1 = open('raw_data.csv', encoding = 'utf-8')
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


# Feature selection 
        
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
file2 = open('tag_inclusion_list.csv', encoding = 'utf-8')
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
 
# Replace brewery ID with Brewery
brewery_dict = {}
file1 = open('brewery-brewdb.csv', encoding = 'utf-8')
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
x = np.concatenate((x1,x2,x3,x4),1)
labels = additional_labels + style_labels + short_name_labels + all_tags
    
# Save data    
f = open("beer_data_pickle.cpkl", 'wb')
cPickle.dump((x,names,labels,beer_list),f)
f.close()    
