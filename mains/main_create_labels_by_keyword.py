
import settings

__author__="cstur"

from pipeline.data_gatherers.imageset_creator import ImagesetCreator

if __name__ == '__main__':
    keyword = "ocean"
    data_creator = ImagesetCreator()
    data_creator.create_image_dataset(keyword=keyword, height=128, width=128, amount=5)

