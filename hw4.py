from sklearn import tree, metrics
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from subprocess import call
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

#read file
data = []
headers = ""
with open('data.csv', 'r') as csvFile:
    csvData = csv.reader(csvFile, delimiter=',')
    headers = next(csvData)
    for row in csvData:
        data.append(row)

#delete row with null data 
data = np.delete(data,2,0)

#first random data. Then make training, validation, and dev set
np.random.shuffle(data)
x1 = data[:4,1:9]
y1 = data[:4,-1]
x2 = data[4:8,1:9]
y2 = data[4:8,-1]
x3 = data[8:,1:9]
y3 = data[8:,-1]

#train data with 3 different max depths
maxDepth = [2,3,4]
for i in maxDepth:
    clf1 = tree.DecisionTreeClassifier(max_depth = i)
    fold1 = np.reshape(np.append(x1,x2),(8,8))
    foldy1 = np.reshape(np.append(y1,y2),(8,1))
    clf1 = clf1.fit(fold1,foldy1)
    result1 = clf1.predict(x3)
    acc = metrics.accuracy_score(y3, result1)

    clf2 = tree.DecisionTreeClassifier(max_depth = i)
    fold2 = np.reshape(np.append(x1,x3),(8,8))
    foldy2 = np.reshape(np.append(y1,y3),(8,1))
    clf2 = clf2.fit(fold2,foldy2)
    result2 = clf2.predict(x2)
    acc += metrics.accuracy_score(y2, result2)

    clf3 = tree.DecisionTreeClassifier(max_depth = i)
    fold3 = np.reshape(np.append(x2,x3),(8,8))
    foldy3 = np.reshape(np.append(y2,y3),(8,1))
    clf3 = clf3.fit(fold3,foldy3)
    result3 = clf3.predict(x1)
    acc += metrics.accuracy_score(y1, result3)

    print("The average absolute error with a max depth of",i,"is",acc/3)

#tree.plot_tree(clf3,feature_names=headers)
#plt.show()
    
num_nodes = [10,50,100]
for i in num_nodes:
    clf1 = RandomForestClassifier(n_estimators = i) 
    clf1 = clf1.fit(fold1,np.ravel(foldy1))
    result1 = clf1.predict(x3)
    acc = metrics.accuracy_score(y3, result1)

    clf2 = RandomForestClassifier(n_estimators = i)
    clf2 = clf2.fit(fold2,np.ravel(foldy2))
    result2 = clf2.predict(x2)
    acc += metrics.accuracy_score(y2, result2)

    clf3 = RandomForestClassifier(n_estimators = i)
    clf3 = clf3.fit(fold3,np.ravel(foldy3))
    result3 = clf3.predict(x1)
    acc += metrics.accuracy_score(y3, result3)

    print("The average absolute error with",i,"number of trees is",acc/3)
    
#plot best model from 3rd fold - 10 trees
estimate = clf3.estimators_[5]
export_graphviz(estimate, out_file= 'tree.dot', rounded = True, proportion = False)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
Image(filename = 'tree.png')


