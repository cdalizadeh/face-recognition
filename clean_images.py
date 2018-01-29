import os
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave

if __name__ == "__main__":
    actors_count = {"Alec Baldwin" : 0, "Bill Hader" : 0, "Daniel Radcliffe" : 0, "Gerard Butler":0, "Michael Vartan":0, "Steve Carell":0, "Lorraine Bracco":0, "Kristin Chenoweth":0, "Fran Drescher":0, "America Ferrera":0, "Peri Gilpin":0, "Angie Harmon":0}
    
    for filename in os.listdir("./data/raw"):
        filename_split = filename.split("-")
        actor = filename_split[0]
        coords = filename_split[1]
        hashcode = filename_split[2].split(".")[0]
        extension = filename_split[2].split(".")[1]

        im = Image.open("./data/raw/" + filename)
        coords = map(int, coords.split(","))    # converts coords from string to list of ints
        im = im.crop(coords)
        im = im.convert("L")
        im = imresize(im, size = (32,32))
        imsave("./data/clean/" + actor + "-" + str(actors_count[actor]) + "-" + hashcode + "." + extension, im)
        actors_count[actor] += 1
    print(4)