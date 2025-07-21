import random
import numpy as np
import cv2

import tensorflow as tf
import matplotlib.pyplot as plt

def get_classes(matriz):
    res = []
    for v in [x[2] for x in matriz]:
        #print (v)
        if v not in res:
            res.append(v)
    return res

def prep_data(data, CATEGORIES, IMG_SIZE, numero_de_canais, shuffle=True):
    
    if shuffle:
        random.shuffle(data)
    
    X = []
    y = []
    for features, label, name in data:
        X.append(features)
        y.append(label)
    
    
    res = np.eye(  len(CATEGORIES)  )[y]

#--------------------------------------------------------------------------
    
    #print('Shape dos dados[0]: ', np.array(X)[0].shape)
    #print('Shape de entrada dos dados: ', np.array(X).shape)
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, numero_de_canais)
    #X = np.array(X)


    
    #print('Shape de saida dos dados: ', np.array(X).shape)
#--------------------------------------------------------------------------
    return X, res


def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array, new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def plot_image(prediction_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
    predicted_label = np.argmax(prediction_array)        
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("Classe - {} | {:2.0f}% (true class {})".format(CATEGORIES[predicted_label], 100*np.max(prediction_array), CATEGORIES[true_label]), color=color)

def plot_value_array(prediction_array, true_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(CATEGORIES)), prediction_array, color= "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)

    thisplot[predicted_label].set_color('red')
    print(CATEGORIES[true_label])
    thisplot[true_label].set_color('green')