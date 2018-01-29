import os
import random
from random import shuffle
from shutil import copyfile

def create_actor_directories():
    for filename in os.listdir("./data/clean"):
        filename_split = filename.split("-")
        actor = filename_split[0]

        if not os.path.exists("./data/organized/" + actor):
            os.makedirs("./data/organized/" + actor)
            os.makedirs("./data/organized/" + actor + "/test")
            os.makedirs("./data/organized/" + actor + "/train")
            os.makedirs("./data/organized/" + actor + "/validation")

def shuffle_actor_images():
    random.seed(42) # The answer to life, the universe and everything

    files_list = {"Alec Baldwin" : [], "Bill Hader" : [], "Daniel Radcliffe" : [], "Gerard Butler" : [], "Michael Vartan" : [], "Steve Carell" : [], "Lorraine Bracco" : [], "Kristin Chenoweth" : [], "Fran Drescher" : [], "America Ferrera" : [], "Peri Gilpin" : [], "Angie Harmon" : []}
    for filename in os.listdir("./data/clean"):
        filename_split = filename.split("-")
        actor = filename_split[0]
        id = filename_split[1]
        hashcode = filename_split[2].split(".")[0]
        extension = filename_split[2].split(".")[1]

        files_list[actor].append(filename)
    
    print(files_list["Alec Baldwin"][0])

    for actor in files_list:
        actors_list = files_list[actor]
        random.shuffle(actors_list)
        for i in range(70):
            copyfile("./data/clean/" + actors_list[i], "./data/organized/" + actor + "/train/" + actors_list[i])
        for i in range(70, 80):
            copyfile("./data/clean/" + actors_list[i], "./data/organized/" + actor + "/validation/" + actors_list[i])
        for i in range(80, 90):
            copyfile("./data/clean/" + actors_list[i], "./data/organized/" + actor + "/test/" + actors_list[i])

if __name__ == "__main__":
    create_actor_directories()
    shuffle_actor_images()