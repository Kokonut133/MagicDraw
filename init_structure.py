import settings
import os

if __name__ == '__main__':
    os.makedirs(settings.root_dir+"resources")
    os.makedirs(settings.root_dir+"resources/images")
    os.makedirs(settings.root_dir+"resources/datasets")