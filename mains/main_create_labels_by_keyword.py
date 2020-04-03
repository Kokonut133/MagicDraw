
import settings

__author__="cstur"

from pipeline.data_gatherers.imageset_creator import ImagesetCreator

if __name__ == '__main__':
    data_creator = ImagesetCreator()
    data_creator.create_image_dataset(keyword="forest", size="medium", amount=1000)
    data_creator.create_image_dataset(keyword="stones", size="medium", amount=1000)
    data_creator.create_image_dataset(keyword="clouds", size="medium", amount=1000)
    data_creator.create_image_dataset(keyword="sky", size="medium", amount=1000)



