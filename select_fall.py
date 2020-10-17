import os
import pandas as pd


path = "Fall/"
fall_folders = os.listdir(path) # Fall1 , ... , Fall35

labels = pd.read_csv("Labels.csv")

name = labels["Unnamed: 0"]
start = labels["Fall Start"]
end = labels["Fall Stop"]

for folder_name in fall_folders:
    # print(os.listdir(path + folder_name + "/" + folder_name))
    images = os.listdir(path + folder_name + "/" + folder_name)
 
