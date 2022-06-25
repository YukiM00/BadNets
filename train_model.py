import os, argparse
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def poison(x,y,width,height,wide,poison_rate=0.1, targeted=0):
    nb_poison = int(x.shape[0]*poison_rate)
    x[:nb_poison,(width):(width+wide),(height):(height+wide)] = 1.0  

    y = np.argmax(y,axis=1)
    if targeted != -1 : 
        #targeted
        y[:nb_poison] = targeted
    else: 
        #non-targeted
        y[:nb_poison] += 1
        y[y==10] = 0
    y = to_categorical(y,10)
    return x,y

def load_cifar10_dataset(IMG_SIZE=32):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    x_train = x_train.reshape(x_train.shape[0], IMG_SIZE, IMG_SIZE, 3)
    x_val   = x_val.reshape(x_val.shape[0], IMG_SIZE, IMG_SIZE, 3)
    x_test  = x_test.reshape(x_test.shape[0], IMG_SIZE, IMG_SIZE, 3)

    x_train = x_train.astype('float32')/255.0 
    x_val = x_val.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0

    return x_train,x_val,x_test,to_categorical(y_train,10),to_categorical(y_val,10),to_categorical(y_test,10)

def load_mnist_dataset(IMG_SIZE=28):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    x_train = x_train.reshape(x_train.shape[0], IMG_SIZE, IMG_SIZE, 1)
    x_val   = x_val.reshape(x_val.shape[0], IMG_SIZE, IMG_SIZE, 1)
    x_test  = x_test.reshape(x_test.shape[0], IMG_SIZE, IMG_SIZE, 1)

    x_train = x_train.astype('float32')/255.0 
    x_val = x_val.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0

    return x_train,x_val,x_test,to_categorical(y_train,10),to_categorical(y_val,10),to_categorical(y_test,10)


def load_simple_model(input_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=0.001,momentum=0.9, nesterov=True),
                metrics=['accuracy'])
    
    model.summary()

    return model

def predict(x,x_b,y,model,targeted=0):
    print("Clean Results")
    trues = np.argmax(y,axis=1)
    preds_clean = np.argmax(model.predict(x), axis=1)
    matrix_clean = confusion_matrix(trues,preds_clean)
    print("Acc:",accuracy_score(trues,preds_clean))
    print(matrix_clean)

    print("Trigger Results")
    preds_poison = np.argmax(model.predict(x_b), axis=1)
    matrix_poison = confusion_matrix(trues,preds_poison)
    print("Poison Acc:",accuracy_score(trues,preds_poison))
    asr = 0
    for i in range(y.shape[1]):
        asr = asr + matrix_poison[i,targeted]
    asr = asr/x_b.shape[0]
    print("Asr:",asr)
    print(matrix_poison)
    
    return matrix_clean,matrix_poison

def sampling_image(filename,img,dataset):
    if dataset == 'mnist':
        img = img.reshape(28,28)
        plt.gray()
    else:
        img = img.reshape(32,32,3)
    plt.imsave(filename,img)


# args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', type=str, help='mnist or cifar10')
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--poison_rate', default=0.1, type=float)
parser.add_argument('--targeted_class', default=0, type=int,help='-1:non-targeted')
args = parser.parse_args()
print(args)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

os.makedirs('models',exist_ok=True)
os.makedirs('samples',exist_ok=True)
num_class = 10
dataset = args.dataset
epochs = args.epochs
batch_size = args.batch_size
poison_rate = args.poison_rate
targeted_class = args.targeted_class

# load dataset
if dataset =='mnist':
    IMG_SIZE = 28
    x_train,x_val,x_test,y_train,y_val,y_test = load_mnist_dataset(IMG_SIZE=IMG_SIZE)
elif dataset =='cifar10':
    IMG_SIZE = 32
    x_train,x_val,x_test,y_train,y_val,y_test = load_cifar10_dataset(IMG_SIZE=IMG_SIZE)
else:
    print("-UNKNOWN DATASET--")
    exit()

x_train, y_train = shuffle(x_train, y_train)
x_train_poison, y_train_poison = x_train.copy(),y_train.copy()
x_test_poison, y_test_poison = x_test.copy(), y_test.copy()
print("Train ",x_train.shape[0])
print("Val ",x_val.shape[0])
print("Test ",x_test.shape[0])

# poisoning dataset
print("Poison Train:",poison_rate*100,"%")
width = x_train[0].shape[0]-3
height = x_train[0].shape[1]-3
wide = 1

x_train_poison, y_train_poison = poison(x_train_poison,y_train_poison,
                                        width=width,height=height,wide=wide,
                                        poison_rate=poison_rate,targeted=targeted_class)

x_test_poison,  y_test_poison = poison(x_test_poison,y_test_poison,
                                        width=width,height=height,wide=wide,
                                        poison_rate=1.0,targeted=targeted_class)

# sampling train data
sampling_image(filename='samples/img_{0}_clean.png'.format(dataset),
            img=x_train[0],
            dataset=dataset)
sampling_image(filename='samples/img_{0}_trigger.png'.format(dataset),
            img=x_train_poison[0],
            dataset=dataset)

# load model
model = load_simple_model(input_shape=x_train_poison.shape[1:])

# train
model.fit(x_train_poison,y_train_poison,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_val, y_val))

model.save_weights('models/{0}_target_{1}_simplemodel_weight.h5'.format(dataset,targeted_class))

matrix_clean,matrix_poison = predict(x_test,x_test_poison,y_test,model,targeted=targeted_class)






