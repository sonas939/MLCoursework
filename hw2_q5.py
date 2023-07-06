from math import *
from matplotlib import ft2font
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#--------------------------------- i ---------------------------------#

#ignore col1, compare to col8
columns =  ["name", "stamina", "attack_value" ,"defense_value", "capture_rate", "flee_rate", "spawn_chance", "combat_point"]
data = pd.read_csv ('hw2_data.csv', usecols=columns)

#average of combat points
combats = data.loc[:, ["combat_point"]]
csum = combats["combat_point"].sum()
combat_avg = csum/146

#replace combat point values with 1 or -1 (class)
for i in range(len(data)):
    if (data.at[i, "combat_point"] > combat_avg):
        data.at[i, "combat_point"] = 1
    elif (data.at[i, "combat_point"] < combat_avg):
        data.at[i, "combat_point"] = -1

#--------------------------------- iii ---------------------------------#
# 5-fold cross-validation process
# randomly split the data into 5 folds and use a 5-fold cross-validation.
data = data.reindex(np.random.permutation(data.index))
data = data.reset_index(drop=True)
#get rid of names col
data.drop(columns=data.columns[0], axis=1, inplace=True)
#add 1s to first col
data.insert(0, 'const', 1)
f1 = data.loc[0:28]
f2 = data.loc[29:57]
f3 = data.loc[58:86]
f4 = data.loc[87:115]
f5 = data.loc[116:147]

#train1: all values but combat point
#train1y: only combat point
train1 = pd.concat([f2,f3,f4,f5]) #117 rows, 8 cols
train1y = train1.loc[:, "combat_point"] #117 rows, 1 col
train1.drop(columns[7], axis=1, inplace=True) #get rid of last col -- 7 cols
#test1: all values but combat point
#test1y: only combat point
test1 = f1 #29 rows, 8 cols
test1y = test1.loc[:, "combat_point"] #29 rows, 1 col
test1.drop(columns[7], axis=1, inplace=True) #get rid of last col -- 7 cols

#------num 2------#
#redoing this part because combat value gets deleted for f5 
# 5-fold cross-validation process
# randomly split the data into 5 folds and use a 5-fold cross-validation.
data = data.reindex(np.random.permutation(data.index))
data = data.reset_index(drop=True)
#get rid of names col
data.drop(columns=data.columns[0], axis=1, inplace=True)
#add 1s to first col
data.insert(0, 'const', 1)
f1 = data.loc[0:28]
f2 = data.loc[29:57]
f3 = data.loc[58:86]
f4 = data.loc[87:115]
f5 = data.loc[116:147]

train2 = pd.concat([f1,f3,f4,f5]) #117 rows, 8 cols
train2y = train2.loc[:, "combat_point"]
train2.drop(columns[7], axis=1, inplace=True) #get rid of last col

test2 = f2 #29 rows, 8 cols
test2y = test2.loc[:, "combat_point"]
test2.drop(columns[7], axis=1, inplace=True) #get rid of last col

#------num 3------#
#redoing this part because combat value gets deleted for f5 
# 5-fold cross-validation process
# randomly split the data into 5 folds and use a 5-fold cross-validation.
data = data.reindex(np.random.permutation(data.index))
data = data.reset_index(drop=True)
#get rid of names col
data.drop(columns=data.columns[0], axis=1, inplace=True)
#add 1s to first col
data.insert(0, 'const', 1)
f1 = data.loc[0:28]
f2 = data.loc[29:57]
f3 = data.loc[58:86]
f4 = data.loc[87:115]
f5 = data.loc[116:147]

train3 = pd.concat([f1,f2,f4,f5]) #117 rows, 8 cols
train3y = train3.loc[:, "combat_point"]
train3.drop(columns[7], axis=1, inplace=True) #get rid of last col

test3 = f3 #29 rows, 8 cols
test3y = test3.loc[:, "combat_point"]
test3.drop(columns[7], axis=1, inplace=True) #get rid of last col

#------num 4------#
# #redoing this part because combat value gets deleted for f5 
# 5-fold cross-validation process
# randomly split the data into 5 folds and use a 5-fold cross-validation.
data = data.reindex(np.random.permutation(data.index))
data = data.reset_index(drop=True)
#get rid of names col
data.drop(columns=data.columns[0], axis=1, inplace=True)
#add 1s to first col
data.insert(0, 'const', 1)
f1 = data.loc[0:28]
f2 = data.loc[29:57]
f3 = data.loc[58:86]
f4 = data.loc[87:115]
f5 = data.loc[116:147]

train4 = pd.concat([f1,f2,f3,f5]) #117 rows, 8 cols
train4y = train4.loc[:, "combat_point"]
train4.drop(columns[7], axis=1, inplace=True) #get rid of last col

test4 = f4 #29 rows, 8 cols
test4y = test4.loc[:, "combat_point"]
test4.drop(columns[7], axis=1, inplace=True) #get rid of last col

#------num 5------#
#redoing this part because combat value gets deleted for f5 
# 5-fold cross-validation process
# randomly split the data into 5 folds and use a 5-fold cross-validation.
data = data.reindex(np.random.permutation(data.index))
data = data.reset_index(drop=True)
#get rid of names col
data.drop(columns=data.columns[0], axis=1, inplace=True)
#add 1s to first col
data.insert(0, 'const', 1)
f1 = data.loc[0:28]
f2 = data.loc[29:57]
f3 = data.loc[58:86]
f4 = data.loc[87:115]
f5 = data.loc[116:147]

train5 = pd.concat([f1,f2,f3,f4]) #116 rows, 7 cols?
train5y = train5.loc[:, "combat_point"]
train5.drop(columns[7], axis=1, inplace=True) #get rid of last col

test5 = f5 #30 rows, 8 cols
test5y = test5.loc[:, "combat_point"]
test5.drop(columns[7], axis=1, inplace=True) #get rid of last col

#--------------------------------- v ---------------------------------#
#train1: 6 features and first column of 1s
#test1: 6 features and first column of 1s
#train1y: only class number
#test1y: only class number

w = [1, 1, 1, 1, 1, 1, 1]
n = len(train1)
for i in range(n):
        # multiply x(n)s by w(0)
        wx = np.dot(w, train1[i].T) 
print(wx) 


def linear_perception (trainx, trainy): #calculate w using trainx trainy, test with testx testy
    for i in range(n):
        # multiply x(n)s by w(0)
        wx = np.dot(w, trainx[i].T) #should return scalar value
        #now we check if the new classes are valid
        #find first sample classified incorrectly
        if (((wx > 0) & (trainy[i] == 1)) or ((wx < 0) & (trainy[i]) == -1)):
            continue
        else:
            break
    #create w1 with whatever xn's matched = false
    w = w + trainx[i].T #should equal an array
    #restart the process with w1 instead of w0

# for i in range(225):
#     pur = linear_perception(train1, w)
#     print(pur)