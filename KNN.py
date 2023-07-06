import numpy as np
from collections import Counter
import csv

'''read csv data into numpy array WITH class'''
def read_file(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = np.array(list(reader)).astype(int)
        return data

'''Compute euclidean distance of vectors x1 and x2 (l2 norm)'''    
def euclidean_dist(x1, x2):
    return np.sqrt(np.sum(np.square(np.subtract(x1,x2))))

'''Compute the Manhattan distance of vectors x1 and x2 (l1 norm)'''
def manhattan_dist(x1,x2):
    return np.sum(np.abs(np.subtract(x1,x2)))

'''Computes accuracy of KNN'''
def compute_accuracy(dev, dev_out, acc_mode = 1):
    correctA = 0
    correctB = 0
    samplesA = 0
    samplesB = 0
    for i in range(len(dev_out)):
        if dev[i][3] == 1:
            samplesA += 1
            if dev_out[i] == dev[i][3]:
                correctA += 1
        if dev[i][3] == 2:
            samplesB += 1
            if dev_out[i] == dev[i][3]:
                correctB += 1
    if acc_mode == 1:
        return (correctA + correctB)/len(dev_out)
    elif acc_mode == 2:
        return 0.5 * (correctA/samplesA) + 0.5 * (correctB/samplesB)

'''K Nearest Neighbor Algorithm'''
def KNN(num_k, train, dev, acc_mode = 1, norm = 2):
    # trim class off train and dev data so doesn't impact vector math
    train_trim = [i[0:3] for i in train]
    dev_trim = [i[0:3] for i in dev]
    acc = []

    for k in range(1, num_k + 2, 2): 
        dev_out = []
        votes = []
        for n in dev_trim:
            dist = []
            count = 0
            #compute distance (l1/l2 norm) between each dev sample and training data
            for m in train_trim:
                if (norm == 1):
                    dist.append([manhattan_dist(np.array(n), np.array(m)),count])
                else:
                    dist.append([euclidean_dist(np.array(n), np.array(m)),count])
                count += 1
            #sort distances in ascending order. Find first k minimum vals WITH indexes
            dist.sort()
            dist_sort = dist[0:k]
            votes = []
            #iterate through first k minimum values. Use train index to retrieve class of train sample
            for i in dist_sort:
                votes.append(train[i[1]][3])
            #find most common class - 1 or 2
            dev_out.append(Counter(votes).most_common(1)[0][0])
        #sum up number of classes that are same between dev_out and dev and calculate accuracy
        #calculates BAcc if acc_mode = 2 and Acc if acc_mode = 1
        acc.append([compute_accuracy(dev, dev_out, acc_mode),k])
        #max accuracy and k value is returned
    #print(acc)
    return max(acc)

def main():
    k = 11
    train = read_file("data_train.csv")
    dev   = read_file("data_dev.csv")
    test = read_file("data_test.csv")
    #KNN run on dev data
    print(KNN(k, train, dev, 1))
    print(KNN(k, train, dev, 2))

    #KNN run on test data
    #print(KNN(k, train, test, 1))
    #print(KNN(k, train, test, 2))
    
    #l1 norm
    k = 7
    #print(KNN(k, train, dev, 1, 1))
    #print(KNN(k, train, dev, 2, 1))

if __name__ == "__main__":
    main()