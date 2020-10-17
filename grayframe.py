import cv2
import os

def convert2gray():
    paths = ["Dataset/Training/falling",
             "Dataset/Training/non_falling",
             "Dataset/Validation/falling",
             "Dataset/Validation/non_falling"
            ]
    for imgs_path in paths:
        imgs = os.listdir(imgs_path)
        for idx, img in enumerate(imgs):
            im = cv2.imread(imgs_path+"/"+img)
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"{imgs_path}/gray_{idx}.jpg", gray_im)
            print(f"[W] {imgs_path} gray_{idx}.jpg")
            os.remove(f"{imgs_path}/{img}")
            print(f"[RM] {img}")

if __name__ == "__main__":
    convert2gray()
