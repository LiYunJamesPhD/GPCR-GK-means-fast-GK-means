import os, sys, json
import numpy as np

# declare variables
file_path = sys.argv[1]
label_file_path = sys.argv[2]
json_file_path = '/home/li-yun/ism_github/preprocessed_file/imagenet_class_index.json'
label_dic = {}
json_dic = {}

# read a json and a text file
with open(label_file_path) as f:
    label_dic = dict(x.rstrip().split(None, 1) for x in f)
with open(json_file_path) as f:
    json_dic = json.load(f)

# re-name files
folder_path = os.path.dirname(file_path)
file_name = os.path.basename(file_path)
label_id = label_dic[file_name]
truth_label = json_dic[str(int(label_id))][1]
new_file_name = truth_label + '.v.' + file_name.split('_')[2]
os.rename(file_path, os.path.join(folder_path, new_file_name))


