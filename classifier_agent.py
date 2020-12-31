# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 05:33:02 2020

@author: Abhinav
"""
import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


f = '../images/objects/red/duck1.png'

size = (38,36)


model = keras.models.load_model("brain")

im = Image.open(f)
rgb = im.convert('RGB')
rgb = rgb.resize(size)
im_array = np.array(rgb)
imgplot = plt.imshow(im_array)

c = model.predict(im_array.reshape(1, 38, 36, 3,))
print(c)

# print(im_array.reshape(-1, 38, 36, 3).shape)