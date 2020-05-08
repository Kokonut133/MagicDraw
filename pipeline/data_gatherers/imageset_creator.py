<<<<<<< HEAD
from google_images_download import google_images_download       # currently broken 4.2.2020
from pipeline.data_gatherers.bing_scraper import ImageDownloader
from icrawler.builtin import GoogleImageCrawler                 # currently broken 4.2.2020
# from pipeline.data_gatherers.image_downloader import ImageDownloader    # urllib.error.URLError: <urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>; cant fix it
=======
# from google_images_download import google_images_download       # currently broken 4.2.2020
# from icrawler.builtin import GoogleImageCrawler                 # currently broken 4.2.2020
# from pipeline.data_gatherers.image_downloader import ImageDownloader    # urllib.error.URLError: <urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>; cant fix it
from typing import overload
>>>>>>> 5897f51d115bcec6cc425a8fce7bd4bd89aa177b

from pipeline.data_gatherers.bing_scraper import ImageDownloader
from pipeline.aws_services.aws_image_processor import AWS_Imageprocessor
import os

import settings

__author__ = "cstur"


class ImagesetCreator():
    def __init__(self):
        pass

<<<<<<< HEAD
    def create_image_dataset(self, keyword, size, amount, height=None, width=None):
        output_dir = settings.googled_dir
        os.makedirs(output_dir, exist_ok=True)
        image_downloader = ImageDownloader()
        if height and width:
            exact_size = str(height)+","+str(width)
            image_downloader.download_from_bing(keyword=keyword, output_dir=output_dir, amount=amount, exact_size=exact_size, chromedriver="D:\code\PycharmProjects\magic_draw\pipeline\data_gatherers\chromedriver.exe")
        else:
            image_downloader.download_from_bing(keyword=keyword, output_dir=output_dir, amount=amount, size=size, chromedriver="D:\code\PycharmProjects\magic_draw\pipeline\data_gatherers\chromedriver.exe")
=======
    def create_image_dataset(self, keyword, image_dir, size, amount=None, prefix=None, height=None, width=None):
        image_downloader = ImageDownloader()
        if type(keyword)==str:
            if height and width:
                exact_size = str(height) + "," + str(width)
                image_downloader.download_from_bing(keyword=keyword, image=image_dir, amount=amount,
                    exact_size=exact_size, prefix=prefix,
                    chromedriver=os.path.join(settings.third_party_dir, "chromedriver.exe"))
            else:
                image_downloader.download_from_bing(keyword=keyword, image_dir=image_dir, amount=amount, size=size,
                    prefix=prefix, chromedriver=os.path.join(settings.third_party_dir, "chromedriver.exe"))

        elif type(keyword)==dict:
            if height and width:
                exact_size = str(height) + "," + str(width)
                for key, amount in keyword.items():
                    image_downloader.download_from_bing(keyword=key, image=image_dir, amount=amount,
                        exact_size=exact_size, prefix=prefix,
                        chromedriver=os.path.join(settings.third_party_dir, "chromedriver.exe"))
            else:
                for key, amount in keyword.items():
                    image_downloader.download_from_bing(keyword=key, image_dir=image_dir, amount=amount, size=size,
                        prefix=prefix, chromedriver=os.path.join(settings.third_party_dir, "chromedriver.exe"))
        else:
            print("Fix your input keyword.")

        # kicks out all False Positives from query. Beware paid service!
        aws_image_processor = AWS_Imageprocessor(testing=True)
        aws_image_processor.upload_folder_to_s3(bucket_name=image_dir, local_dir=os.path.join(settings.img_dir, image_dir))
        aws_image_processor.create_task_with_batch(title="Is the label in the picture?",
            instructions="Confirm if the Image Title really fits the image.", reward="0.01", workers_per_hit=1,
            process_time_in_s=30,
           parameter_file=os.path.join(settings.root_dir, "pipeline", "aws_services", "s3_references", image_dir+".csv"))
>>>>>>> 5897f51d115bcec6cc425a8fce7bd4bd89aa177b
