#load files from csv into python

# remove items with missing values

# replace brewery id number with brewery name

from collections import Counter
import numpy as np
import csv
import pickle
import re
import string



# Load data from csv file
beer_list = []
file1 = open('raw_data.csv', encoding = 'utf-8')
csv_reader = csv.reader(file1,delimiter = ',')

for item in csv_reader:
    beer = {}
    beer["name"] = item[0]
    beer["brewery_id"] = item[2]
    beer["abv"] = item[1]
    beer["ibu"] = item[4]
    beer["fgmin"] = item[3]
    beer["og"] = item[5]
    beer["srm"] = item[7]
    beer["short_name"] = item[6]
    beer["style"] = item[8]
    beer["description"] = item[9]
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
        

word_lists = []
# Note: I copied this script to convert a string into a list of words from
# https://stackoverflow.com/questions/6181763/converting-a-string-to-a-list-of-words
for item in beer_list:
    text = item['description']
    words = re.sub("[^\w]", " ",  text).split()
    word_lists.append(words)

all_words = []
for item in word_lists:
    for word in item:
        word = word.lower()
        #word = word.translate(None, string.punctuation)
        all_words.append(word)
word_set = list(set(all_words))

word_counts = Counter(all_words)
temp = word_counts.most_common()

## Assign labels here