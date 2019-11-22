import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET

# Instantiating list of images, one hots
images = []
one_hots = []

# Reading images, annotations in each class directory
annotation_class_paths = os.listdir("Annotation")
images_class_paths = os.listdir("Images")
zipped = zip(annotation_class_paths, images_class_paths)
for i, (images_class_path, annotation_class_path) in enumerate(zipped):
    images_class_path = "Images/" + images_class_path
    annotation_class_path = "Annotation/" + annotation_class_path
    image_paths = os.listdir(images_class_path)
    annotation_paths = os.listdir(annotation_class_path)
    zipped = zip(image_paths[:10], annotation_paths[:10])
    for image_path, annotation_path in zipped:
        image_path = images_class_path + "/" + image_path
        annotation_path = annotation_class_path + "/" + annotation_path
        # Reading image
        image = cv2.imread(image_path)
        # Getting bounding box coordinates
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        x_min = int(root.find('object').find('bndbox').find('xmin').text)
        y_min = int(root.find('object').find('bndbox').find('ymin').text)
        x_max = int(root.find('object').find('bndbox').find('xmax').text)
        y_max = int(root.find('object').find('bndbox').find('ymax').text)
        # Cropping image to bounding box
        image = image[y_min:y_max,x_min:x_max]
        # Omitting images where either valid region dim < 256
        if image.shape[0] > 255 and image.shape[1] > 255:
            # Cropping to top left 256x256 square 
            images.append(image[:256,:256,:])
            # Adding corresponding one hot vector
            one_hot = np.zeros(120)
            one_hot[i] = 1
            one_hots.append(one_hot)

images = np.array(images)
one_hots = np.array(one_hots)
images_per_class = map(len, images)

print("shape of one hots: ", one_hots.shape)
print("Shape of imaegs: ", images.shape)
print("image shape: ", images[0].shape)
'''
for c in class_images:
    print(c.shape)
'''