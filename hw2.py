import numpy as np
import pandas as pd
import csv

'''read csv data into dataframe and randomly reindex data'''
def read_file(filename):
    #read data into dataframe
    data = pd.read_csv(filename)   
    #reindex data randomly
    data = data.reindex(np.random.permutation(data.index))
    data = data.reset_index(drop=True)
    data = data.drop('name',1)                                    
    return data

'''perform linear regression'''
def linear_regression(train, test):
    train_x = train.drop('combat_point',1)
    train_x.insert(0,'w0',1)
    train_y = train['combat_point']
    test_x = test.drop('combat_point',1)
    test_x.insert(0,'w0',1)
    test_y = test['combat_point']
    #calculate w
    w = (np.linalg.inv(train_x.T @ train_x) @ train_x.T) @ train_y
    #calculate RSS
    diff = test_y - (test_x.to_numpy() @ w)
    RSS = np.sqrt(diff.T @ diff)
    return RSS

'''perform linear perceptron. Classify samples to class 1 or -1'''
def linear_perceptron(train, test, iterations=200):
    #modify dataframes for linear perceptron
    train_x = train.drop('class',1)
    train_x.insert(0,'w0',1)
    train_class = train['class']
    test_x = test.drop('class',1)
    test_x.insert(0,'w0',1)
    test_class = test['class']
    
    #set w(0) to [1,1,1,1,1,1,1] for now - arbitrary value
    w = np.ones((1,len(train_x.axes[1])))
    
    #chose 200 as default number of iterations
    for i in range(iterations):
        #iterate through train data. As soon as misclassified sample is found, break out of loop and recalculate w
        sampleWrong = []
        sampleClass = 0
        for j in range(len(train_x)):  
            sample = w @ np.transpose(train_x.iloc[[j]].to_numpy())
            if sample > 0 and train_class.iloc[j] == 1:
                continue
            elif sample < 0 and train_class.iloc[j] == -1:
                continue
            else:
                sampleWrong = train_x.iloc[[j]].to_numpy()
                sampleClass = train_class.iloc[j]
                break
        w = w + sampleClass * sampleWrong
    
    #compute accuracy on test data
    correct = 0
    for i in range(len(test_x)):
        sample = w @ np.transpose(test_x.iloc[[i]].to_numpy())
        if (sample > 0 and test_class.iloc[i] == 1) or (sample < 0 and test_class.iloc[i] == -1):
            correct += 1
    return correct/len(test_x)

def main():
    data = read_file("hw2_data.csv")

    #Q4 - remove certain columns to see how RSS changes
    #data = data.drop('attack_value',1)
    #data = data.drop('defense_value',1)
    #data = data.drop('spawn_chance',1)
    #data = data.drop('flee_rate',1)
    #data = data.drop('capture_rate',1)
    #data = data.drop('stamina',1)

    #split data into 5 folds
    dataSize = (len(data))/5
    x1 = data.loc[0:dataSize]
    x2 = data.loc[dataSize+1:dataSize*2]
    x3 = data.loc[dataSize*2+1:dataSize*3]
    x4 = data.loc[dataSize*3+1:dataSize*4]
    x5 = data.loc[dataSize*4+1:dataSize*5]

    #test on 5 different sets 
    linReg=[]
    train1 = pd.concat([x2,x3,x4,x5])
    test1 = x1
    linReg.append(linear_regression(train1,test1))
    train2 = pd.concat([x1,x3,x4,x5])
    test2 = x2
    linReg.append(linear_regression(train2,test2))
    train3 = pd.concat([x1,x2,x4,x5])
    test3 = x3
    linReg.append(linear_regression(train3,test3))
    train4 = pd.concat([x1,x2,x3,x5])
    test4 = x4
    linReg.append(linear_regression(train4,test4))
    train5 = pd.concat([x1,x2,x3,x4])
    test5 = x5
    linReg.append(linear_regression(train5,test5))
    
    print("--------Linear Regression--------")
    for i in range(len(linReg)):
        print("RSS for Fold",i+1,":",linReg[i])
    print("Average RSS :",np.average(linReg))

    #linear perceptron
    #calculate mean of combat points for classification
    meanPoints = np.average(data['combat_point'])
    data['class'] = 1
    for i in range(len(data)):
        if data.at[i,'combat_point'] < meanPoints:
            data.at[i,'class'] = -1
    data = data.drop('combat_point',1)
    x1 = data.loc[0:dataSize]
    x2 = data.loc[dataSize+1:dataSize*2]
    x3 = data.loc[dataSize*2+1:dataSize*3]
    x4 = data.loc[dataSize*3+1:dataSize*4]
    x5 = data.loc[dataSize*4+1:dataSize*5]

    #test on 5 different sets 
    linPer=[]
    train1 = pd.concat([x2,x3,x4,x5])
    test1 = x1
    linPer.append(linear_perceptron(train1,test1))
    train2 = pd.concat([x1,x3,x4,x5])
    test2 = x2
    linPer.append(linear_perceptron(train2,test2))
    train3 = pd.concat([x1,x2,x4,x5])
    test3 = x3
    linPer.append(linear_perceptron(train3,test3))
    train4 = pd.concat([x1,x2,x3,x5])
    test4 = x4
    linPer.append(linear_perceptron(train4,test4))
    train5 = pd.concat([x1,x2,x3,x4])
    test5 = x5
    linPer.append(linear_perceptron(train5,test5))
    print("--------Linear Perceptron--------")
    for i in range(len(linPer)):
        print("Accuracy for Fold",i+1,":",linPer[i])
    print("Average Accuracy :",np.average(linPer))

if __name__ == "__main__":
    main()