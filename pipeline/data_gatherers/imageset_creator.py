# from google_images_download import google_images_download       # currently broken 4.2.2020
# from icrawler.builtin import GoogleImageCrawler                 # currently broken 4.2.2020
# from pipeline.data_gatherers.image_downloader import ImageDownloader    # urllib.error.URLError: <urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>; cant fix it
from pipeline.data_gatherers.bing_scraper import ImageDownloader
from pipeline.aws_services.aws_image_processor import AWS_Imageprocessor
import os

import settings

__author__ = "cstur"


class ImagesetCreator():
    def __init__(self):
        pass

    def create_image_dataset(self, keyword, image_dir, size, amount, prefix=None, height=None, width=None):
        image_downloader = ImageDownloader()
        # if height and width:
        #     exact_size = str(height)+","+str(width)
        #     image_downloader.download_from_bing(keyword=keyword, image=image_dir, amount=amount, exact_size=exact_size,
        #                                         prefix=prefix,
        #                                         chromedriver=os.path.join(settings.third_party_dir, "chromedriver.exe"))
        # else:
        #     image_downloader.download_from_bing(keyword=keyword, image_dir=image_dir, amount=amount, size=size,
        #                                         prefix=prefix,
        #                                         chromedriver=os.path.join(settings.third_party_dir, "chromedriver.exe"))

        # AWS Clean Image-set
        image_cleaner = AWS_Imageprocessor(testing=True)





