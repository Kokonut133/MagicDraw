import settings
import os

if __name__ == '__main__':
    try:
        os.makedirs(settings.root_dir+"resources")
        os.makedirs(settings.root_dir+"resources/images")
        os.makedirs(settings.root_dir+"resources/datasets")
        os.makedirs(settings.root_dir+"resources/results")
    except:
        print("Folders already exist")

#######################################################################################################################

# dataset links
#
# COCO
# images http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# labels http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
#
# pix2pix
# https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/
