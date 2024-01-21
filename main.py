"""This script runs the whole MOCE method."""

import numpy as np
import os
import cv2
import time
from tools import target_model as model
from tools import clustering as ct
from tools import evaluate as ev

np.random.seed(76923)

# ImageNet labels.
with open("map_clsloc.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

data_dir = './dataset'                        # Dataset Directory.
class_names = np.sort(os.listdir(data_dir))   # class name list.
image_len = 50                                # The number of images.
arg_model = "inception_v3"                    # model = vgg19, resnet50, inception_v3.
img_size = 299                                # inception_v3 = 299, others = 224.
layer_num = 15                                # layer number = 0~15. 15 is the layer number closest to the flatten layer.
gamma = 10                                    # top 10%.
alpha = 0.5                                   # Sr weight.
beta = 1                                      # Se weight.

def main():
    for class_name in class_names:
        class_path = os.path.join(data_dir,class_name)
        img_names = np.sort(os.listdir(class_path))
        img_names = np.random.choice(img_names, image_len, replace=True, p=None)
        
        # Matching class name (n02119789 -> kit_fox).
        for i in range(len(categories)):
            if class_name == categories[i].split(' ')[0]:
                name = categories[i].split(' ')[2]
                break
        class_name = name.replace("_"," ")
        print(class_name)
        images = []
        
        # Storing images.
        for name in img_names:
            image_path = os.path.join(class_path, name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (img_size, img_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        
        # Extracting probabilities, activation maps, gradients.
        model_out = model.TargetModel(images, arg_model,layer_num, class_name)
        result = model_out.run()
        data = [result[1],result[2],]
        prob = result[0]
        
        # Discovering concepts.
        clusters = ct.Clustering(data, images, arg_model, layer_num, class_name, gamma)
        concepts = clusters.run()
        
        # Evaluating and summarizing concepts.
        ev.Evaluator(concepts, images, prob, arg_model, layer_num, class_name, alpha, beta).run()

if __name__ == '__main__':
    main()
