# from google_images_download import google_images_download       # currently broken 4.2.2020
# from icrawler.builtin import GoogleImageCrawler                 # currently broken 4.2.2020
# from pipeline.data_gatherers.image_downloader import ImageDownloader    # urllib.error.URLError: <urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>; cant fix it
from typing import overload

from pipeline.data_gatherers.bing_scraper import ImageDownloader
from pipeline.aws_services.aws_image_processor import AWS_Imageprocessor
import os

import settings

__author__ = "cstur"


class ImagesetCreator():
    def __init__(self):
        pass

    def create_image_dataset(self, keyword, image_dir, size, amount=None, prefix=None, height=None, width=None):
        image_downloader = ImageDownloader()
        # if type(keyword)==str:
        #     if height and width:
        #         exact_size = str(height) + "," + str(width)
        #         image_downloader.download_from_bing(keyword=keyword, image=image_dir, amount=amount,
        #             exact_size=exact_size, prefix=prefix,
        #             chromedriver=os.path.join(settings.third_party_dir, "chromedriver.exe"))
        #     else:
        #         image_downloader.download_from_bing(keyword=keyword, image_dir=image_dir, amount=amount, size=size,
        #             prefix=prefix, chromedriver=os.path.join(settings.third_party_dir, "chromedriver.exe"))
        #
        # elif type(keyword)==dict:
        #     if height and width:
        #         exact_size = str(height) + "," + str(width)
        #         for key, amount in keyword.items():
        #             image_downloader.download_from_bing(keyword=key, image=image_dir, amount=amount,
        #                 exact_size=exact_size, prefix=prefix,
        #                 chromedriver=os.path.join(settings.third_party_dir, "chromedriver.exe"))
        #     else:
        #         for key, amount in keyword.items():
        #             image_downloader.download_from_bing(keyword=key, image_dir=image_dir, amount=amount, size=size,
        #                 prefix=prefix, chromedriver=os.path.join(settings.third_party_dir, "chromedriver.exe"))
        # else:
        #     print("Fix your input keyword.")

        # kicks out all False Positives from query. Beware paid service!
        image_cleaner = AWS_Imageprocessor(testing=True)
        image_cleaner.create_task(title="Is the label in the picture?",
            instructions="Confirm if the Image Title really fits the image.", reward="0.01", workers_per_hit=1,
            process_time_in_s=30,
            question_file=os.path.join(settings.root_dir, "pipeline", "aws_services", "question_cleanup_images.xml"))
        image_cleaner.upload_folder_to_s3(bucket_name=image_dir, local_dir=os.path.join(settings.img_dir, image_dir))

