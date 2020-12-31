# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 21:58:05 2020

@author: Abhinav
"""

import imageio
import matplotlib.pyplot as plt
from os import path
import numpy as np
from PIL import Image
import pandas as pd 
from labels import BLACK, RED, BLUE, OBJECT


x = []
y = []

# #Labels
# BLACK = 'Black'
# RED = 'Red'
# BLUE = 'Blue'
# OBJECT = "Object"

labels = [BLACK, RED, BLUE, OBJECT]

size = (38,36)
#Ducks
for i in range(len(labels)):
    n=1
    while True:
        duck = '../images/objects/'+labels[i].lower()+'/duck'+str(n)+'.png'
        obj = '../images/objects/'+labels[i].lower()+'/object'+str(n)+'.png'
        if(path.exists(duck) == True):
            im = Image.open(duck)
            rgb = im.convert('RGB')
            rgb = rgb.resize(size)
            im_array = np.array(rgb)
            imgplot = plt.imshow(im_array)
            x.append(im_array)
            y.append(labels[i])
        if(path.exists(obj) == True):
            print('Object image')
            im = Image.open(obj)
            rgb = im.convert('RGB')
            rgb = rgb.resize(size)
            im_array = np.array(rgb)
            imgplot = plt.imshow(im_array)
            x.append(im_array)
            y.append(labels[i])
        if(path.exists(obj) == False and path.exists(duck) == False):
            break
        n+=1
        
X = np.array(x)
Y = np.array(y)

print(X.shape)
print(Y.shape)

for i in range(len(X)):
    print('Image..')
    img = X[i]
    print(img.shape)
    plt.imshow(img, cmap="Greys")
    plt.show()

print(X.shape)

np.save("./dataset/X.npy", X)
np.save("./dataset/Y.npy", Y)

