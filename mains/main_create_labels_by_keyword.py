
import settings

__author__="cstur"

from pipeline.data_gatherers.imageset_creator import ImagesetCreator

if __name__ == '__main__':
    data_creator = ImagesetCreator()
    # data_creator.create_image_dataset(keyword="stones -rolling", size="medium", amount=1000)        # -rolling to exclude search results for the rolling stones
    # data_creator.create_image_dataset(keyword="grass", size="medium", amount=1000)
    # data_creator.create_image_dataset(keyword="clouds", size="medium", amount=1000)



