import os
import re
import urllib

def clean_text(gender):
    #input: 'males' or 'females'
    text_raw = open("./data/" + gender + "_raw.txt", "r")
    lines = text_raw.readlines()
    clean = []              #format is [name, url, coordinates, hash]
    for line in lines:
        line_split = re.split(r'\t+', line)
        new_line = [line_split[0], line_split[3], line_split[4], line_split[5]]
        clean.append(new_line)
    return clean

if __name__ == "__main__":
    #try:
    #    print urllib.URLopener().retrieve(urls[1], "./test.jpg")
    #except:
    #    print "failure"
    #print re.split(r'\t+', males_list[0])
    clean = clean_text("males")
    print clean[0]