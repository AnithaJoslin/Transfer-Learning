import os
import glob
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model


fruit_label = ['Banana', 'Chestnut', 'Kiwi', 'Pear']
def process_img(file_path, save_file_path):
    img = Image.open(file_path)
    img = img.resize([100, 100])
    img.save(save_file_path)
    img = plt.imread(save_file_path)
    print(img.shape)
    img = preprocess_input(img)
    print(img.shape)
    return img

if __name__ == '__main__':
    model = load_model('model/model.h5')
    file_path = input("Enter File Path\n")
    save_file_path = input("Enter saved file path:")
    img = process_img(file_path, save_file_path)
    print(img.shape)
    prediction = model.predict(np.array([img]))
    prediction = np.argmax(prediction)
    print(fruit_label[prediction])



