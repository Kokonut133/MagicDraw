
import settings

__author__="cstur"

from pipeline.data_gatherers.imageset_creator import ImagesetCreator

if __name__ == '__main__':
    data_creator = ImagesetCreator()
<<<<<<< HEAD
    data_creator.create_image_dataset(keyword="forest", size="medium", amount=1000)
    data_creator.create_image_dataset(keyword="stones", size="medium", amount=1000)
    data_creator.create_image_dataset(keyword="clouds", size="medium", amount=1000)
    data_creator.create_image_dataset(keyword="sky", size="medium", amount=1000)
=======
    keywords_and_amount = {"grass": 200,
                           "stone": 200,
                           "sand": 200,
                           "sky": 200,
                           "clouds": 200}

    data_creator.create_image_dataset(keyword=keywords_and_amount, size="medium", image_dir="landscapemix0", prefix="landscape ")
>>>>>>> 5897f51d115bcec6cc425a8fce7bd4bd89aa177b



