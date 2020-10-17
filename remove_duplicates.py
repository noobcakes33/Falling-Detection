import os
import cv2

path = "frames/"
images = []

for img in os.listdir(path):
    image = cv2.imread(path + img)
    images.append(image)

print(len(images))

for i in range(len(images)):
    for j in range(i+1,len(images)):
        if (images[i] == images[j]).all():
            images.pop(j)

print(len(images))
