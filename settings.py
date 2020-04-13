import os

# global variable storage
root_dir = os.path.realpath(__file__)[0:-(len(os.path.basename(__file__)))]
res_dir = os.path.join(root_dir, "resources")
img_dir = os.path.join(res_dir, "images")
dataset_dir = os.path.join(res_dir, "datasets")
googled_dir = os.path.join(dataset_dir, "googled")
result_dir = os.path.join(res_dir, "results")
third_party_dir = os.path.join(root_dir, "third_party")