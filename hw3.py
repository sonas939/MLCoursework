import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.datasets import cifar10
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers, optimizers, regularizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import confusion_matrix
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import time


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = rgb2gray(X_train)
X_test = rgb2gray(X_test)

train_images_3d, train_images_y = X_train[0:40000].reshape(40000,32,32,1), y_train[:40000]
val_images_3d, val_images_y = X_train[40000:50000].reshape(10000,32,32,1), y_train[40000:]
flatten_X_test = X_test.reshape(10000,32,32,1)

#Question 1
def plot_data(X_train):
    plt.imshow(X_train[0], cmap='gray')
    plt.show()
    plt.imshow(X_train[12], cmap='gray')
    plt.show()
    plt.imshow(X_train[14], cmap='gray')
    plt.show()
    plt.imshow(X_train[16], cmap='gray')
    plt.show()
    plt.imshow(X_train[18], cmap='gray')
    plt.show()
    plt.imshow(X_train[20], cmap='gray')
    plt.show()

def count_samples(y_train):
    class_count = [0] * 10
    for i in y_train:
        class_count[i[0]] += 1
    return class_count

def FNN(X_train, X_test, y_train, y_test):
    #use last 10,000 samples as validation data
    train_data = X_train[0:40000]
    val_data = X_train[40000:50000]

    flatten_X_train = train_data.reshape((-1, 32*32))
    flatten_val_data = val_data.reshape((-1, 32*32))
    flatten_X_test = X_test.reshape((-1, 32*32))
    
    #model 1 fitting and validation
    model1 = Sequential([Dense(1024, input_shape=(1024,), activation='relu'),
                    Dense(512, activation='relu'),
                    Dense(256, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(10, activation='softmax')])
    model1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'],)
    startTime1 = time.time()
    model1.fit(flatten_X_train, to_categorical(y_train[0:40000]), epochs=10, batch_size=128,)
    endTime1 = time.time()
    performance1 = model1.evaluate(flatten_val_data, to_categorical(y_train[40000:50000]))
    model1.summary()
    print("Accuracy on Test samples with model 1: {0}".format(performance1[1]))
    print("Time Duration of Model 1:",endTime1-startTime1)

    #model 2 fitting and validation
    model2 = Sequential([Dense(512, input_shape=(1024,), activation='sigmoid'),
                    Dense(256, activation='sigmoid'),
                    Dense(128, activation='sigmoid'),
                    Dense(10, activation='softmax')])
    model2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'],)
    startTime2 = time.time()
    model2.fit(flatten_X_train, to_categorical(y_train[0:40000]), epochs=15, batch_size=256,)
    endTime2 = time.time()
    performance2 = model2.evaluate(flatten_val_data, to_categorical(y_train[40000:50000]))
    model2.summary()
    print("Accuracy on Test samples with model 2: {0}".format(performance2[1]))
    print("Time Duration of Model 2:",endTime2-startTime2)

    model3 = Sequential([Dense(1024, input_shape=(1024,), activation='relu'),
                    Dense(512, activation='relu'),
                    Dense(256, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(10, activation='softmax')])
    model3.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'],)
    startTime3 = time.time()
    model3.fit(flatten_X_train, to_categorical(y_train[0:40000]), epochs=20, batch_size=128,)
    endTime3 = time.time()
    performance3 = model3.evaluate(flatten_val_data, to_categorical(y_train[40000:50000]))
    model3.summary()
    print("Accuracy on Test samples with model 3: {0}".format(performance3[1]))
    print("Time Duration of Model 3:",endTime3-startTime3)

    model2_testPerformance = model2.evaluate(flatten_X_test, to_categorical(y_test))
    print("Accuracy on Test samples with model 2: {0}".format(model2_testPerformance[1]))
    model2_pred = model2.predict(flatten_X_test)
    model2_pred = np.argmax(model2_pred, axis=1)
    model2_acc = np.argmax(to_categorical(y_test), axis=1)
    conf_matr = confusion_matrix(model2_acc, model2_pred)
    print(conf_matr)

def CNN(X_train, X_test, y_train, y_test):
    train_images_3d = X_train[0:40000].reshape(40000,32,32,1)
    val_images_3d = X_train[40000:50000].reshape(10000,32,32,1)
    test_images_3d = X_test.reshape(10000,32,32,1)
    
    # Define 2 groups of layers: features layer (convolutions) and classification layer

    # Model 1
    common_features1 = [Conv2D(32, kernel_size=5, activation='sigmoid', input_shape=(32,32,1)), 
                MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                Conv2D(64, kernel_size=5, activation='sigmoid'),
                MaxPooling2D(pool_size=(2,2)), Flatten(),]
    classifier1 = [Dense(32, activation='sigmoid'), Dense(10, activation='softmax'),]

    cnn_model1 = Sequential(common_features1+classifier1)
    print(cnn_model1.summary())  # Compare number of parameteres against FFN
    cnn_model1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'],)
    startTime1 = time.time()
    cnn_model1.fit(train_images_3d, to_categorical(y_train[0:40000]), epochs=10, batch_size=256,)
    endTime1 = time.time()
    performance1 = cnn_model1.evaluate(val_images_3d, to_categorical(y_train[40000:50000]))

    print("Accuracy on Test samples: {0}".format(performance1[1]))
    print("Time Duration of Model 1:",endTime1-startTime1)

    # Model 2
    common_features2 = [Conv2D(32, kernel_size=3, activation='relu', input_shape=(32,32,1)), 
                MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
                Conv2D(128, kernel_size=3, activation='relu'),
                Conv2D(64, kernel_size=3, activation='relu'),
                MaxPooling2D(pool_size=(2,2)), Flatten(),]
    classifier2 = [Dense(64, activation='sigmoid'), Dense(10, activation='softmax'),]

    cnn_model2 = Sequential(common_features2+classifier2)
    print(cnn_model2.summary())  # Compare number of parameteres against FFN
    cnn_model2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'],)
    startTime2 = time.time()
    cnn_model2.fit(train_images_3d, to_categorical(y_train[0:40000]), epochs=10, batch_size=256,)
    endTime2 = time.time()
    performance2 = cnn_model2.evaluate(val_images_3d, to_categorical(y_train[40000:50000]))

    print("Accuracy on Test samples: {0}".format(performance2[1]))
    print("Time Duration of Model 1:",endTime2-startTime2)

    # Model 3
    common_features3 = [Conv2D(32, kernel_size=3, activation='relu', input_shape=(32,32,1)), 
                MaxPooling2D(pool_size=(2,2)),
                Conv2D(128, kernel_size=3, activation='relu'),
                MaxPooling2D(pool_size=(2,2)),
                Conv2D(64, kernel_size=3, activation='relu'),
                MaxPooling2D(pool_size=(2,2)), Flatten(),]
    classifier3 = [Dense(128, activation='relu'), Dense(10, activation='softmax'),]

    cnn_model3 = Sequential(common_features3+classifier3)
    print(cnn_model3.summary())  # Compare number of parameteres against FFN
    cnn_model3.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'],)
    startTime3 = time.time()
    cnn_model3.fit(train_images_3d, to_categorical(y_train[0:40000]), epochs=10, batch_size=256,)
    endTime3 = time.time()
    performance3 = cnn_model3.evaluate(val_images_3d, to_categorical(y_train[40000:50000]))

    print("Accuracy on Test samples: {0}".format(performance3[1]))
    print("Time Duration of Model 1:",endTime3-startTime3)

    #Best model
    performance_best_model = cnn_model3.evaluate(test_images_3d, to_categorical(y_test))
    print("Accuracy on Test samples: {0}".format(performance_best_model[1]))

def optimize_cnn(hyperparameter):
    # Define model using hyperparameters 

    cnn_model = Sequential([Conv2D(hyperparameter['input_nodes'], kernel_size=hyperparameter['conv_kernel_size'], activation='relu', input_shape=(32,32,1)), 
            MaxPooling2D(pool_size=(2,2)),
            Dropout(hyperparameter['dropout_prob']),
            Conv2D(64, kernel_size=hyperparameter['conv_kernel_size'], activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(hyperparameter['dropout_prob']), 
            Flatten(),
            Dense(32, activation='relu'), 
            Dense(10, activation='softmax'),])

    cnn_model.compile(optimizer=hyperparameter['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    cnn_model.fit(train_images_3d, to_categorical(train_images_y), epochs=2, batch_size=256, verbose=0)
    # Evaluate accuracy on validation data
    performance = cnn_model.evaluate(val_images_3d, to_categorical(val_images_y), verbose=0)

    print("Hyperparameters: ", hyperparameter, "Accuracy: ", performance[1])
    print("----------------------------------------------------")
    # We want to minimize loss i.e. negative of accuracy
    return({"status": STATUS_OK, "loss": -1*performance[1], "model":cnn_model})

space = {
# The kernel_size for convolutions:
'conv_kernel_size': hp.choice('conv_kernel_size', [1, 3, 5]),
#N Number of input nodes:
'input_nodes': hp.choice('input_nodes', [32, 64, 128]),
# Uniform distribution in finding appropriate dropout values
'dropout_prob': hp.uniform('dropout_prob', 0.1, 0.35),
# Choice of optimizer 
'optimizer': hp.choice('optimizer', ['Adam', 'sgd']),
}

trials = Trials()

# Find the best hyperparameters
best = fmin(
        optimize_cnn,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=10,
    )

print("==================================")
print("Best Hyperparameters", best)

test_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

performance = test_model.evaluate(flatten_X_test, to_categorical(y_test))

print("==================================")
print("Test Accuracy: ", performance[1])
