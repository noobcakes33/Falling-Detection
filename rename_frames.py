import os

def rename_frames(dir_name, class_name):
    imgs = os.listdir(dir_name)
    for i in range(len(imgs)):
        os.rename(dir_name+imgs[i], dir_name+class_name+"_"+str(i)+".jpg")


if __name__ == "__main__":
    #rename_frames("non_falling_dataset/", class_name="non_fall")
    #rename_frames("falling_dataset/", class_name="fall")
    rename_frames("frames/", class_name="nonfall_frame")
