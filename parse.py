import os
import re
import urllib2
import hashlib

def clean_text(gender):
    #input: 'males' or 'females'
    text_raw = open("./data/" + gender + "_raw.txt", "r")
    lines = text_raw.readlines()
    clean = []              #format is [name, url, coordinates, hash]
    for line in lines:
        line_split = re.split(r'\t+', line)
        print line_split
        new_line = [line_split[0], line_split[3], line_split[4], line_split[5]]
        clean.append(new_line)
    return clean

def download_images(clean):
    for image_line in clean:
        name = image_line[0]
        url = image_line[1]
        coordinates = image_line[2]
        hashcode = image_line[3].rstrip("\n\r") #rstrip required to remove newline symbol

        try:
            request = urllib2.urlopen(url, timeout=10)
        except:
            print("error opening url")
            continue
        
        try:
            filedata = request.read()
        except:
            print("error reading file")
            continue

        if hashlib.sha256(filedata).hexdigest() != hashcode:
            print("hashes did not match")
            continue

        filename = name + "-" + coordinates + "-" + hashcode
        imgfile = open("./data/raw/" + filename, "wb")
        try:
            imgfile.write(filedata)
        except:
            print("error writing file data")
            continue
        
        print "success"

if __name__ == "__main__":
    clean = clean_text("females")
    download_images(clean)