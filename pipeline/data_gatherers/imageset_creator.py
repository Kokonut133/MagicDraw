from google_images_download import google_images_download       # currently broken 4.2.2020
from pipeline.data_gatherers.bing_scraper import ImageDownloader
from icrawler.builtin import GoogleImageCrawler                 # currently broken 4.2.2020
# from pipeline.data_gatherers.image_downloader import ImageDownloader    # urllib.error.URLError: <urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>; cant fix it

import os
from PIL import Image

__author__ = "cstur"

import settings


class ImagesetCreator():
    def __init__(self):
        pass

    def create_image_dataset(self, keyword, size, amount, height=None, width=None):
        output_dir = settings.googled_dir
        os.makedirs(output_dir, exist_ok=True)
        image_downloader = ImageDownloader()
        if height and width:
            exact_size = str(height)+","+str(width)
            image_downloader.download_from_bing(keyword=keyword, output_dir=output_dir, amount=amount, exact_size=exact_size, chromedriver="D:\code\PycharmProjects\magic_draw\pipeline\data_gatherers\chromedriver.exe")
        else:
            image_downloader.download_from_bing(keyword=keyword, output_dir=output_dir, amount=amount, size=size, chromedriver="D:\code\PycharmProjects\magic_draw\pipeline\data_gatherers\chromedriver.exe")
