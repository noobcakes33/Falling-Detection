import os
import random
from shutil import copyfile


path = "Dataset/Training/"

falling_imgs = os.listdir(path + "falling")
non_falling_imgs = os.listdir(path + "non_falling")

random_fall = random.sample(falling_imgs, int(len(falling_imgs)*0.25))
random_non_fall = random.sample(falling_imgs, int(len(non_falling_imgs)*0.25))

for rand_img in random_fall:
    print("[falling image] ", rand_img)
    if rand_img not in os.listdir("Dataset/Validation/falling"):
        copyfile(src=path+"falling/"+rand_img, dst="Dataset/Validation/falling")

for rand_img in random_non_fall:
    print("[non-falling image] ", rand_img)
    if rand_img not in os.listdir("Dataset/Validation/non_falling"):
        copyfile(src=path+"non_falling/"+rand_img, dst="Dataset/Validation/non_falling")
