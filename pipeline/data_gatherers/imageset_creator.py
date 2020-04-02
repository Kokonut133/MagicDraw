from google_images_download import google_images_download       # currently broken 4.2.2020
import pipeline.data_gatherers.bing_scraper
from icrawler.builtin import GoogleImageCrawler                 # currently broken 4.2.2020
from pipeline.data_gatherers.image_downloader import ImageDownloader    # urllib.error.URLError: <urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>; cant fix it

import os
from PIL import Image

__author__ = "cstur"

import settings


class ImagesetCreator():
    def __init__(self):
        pass

    def create_image_dataset(self, keyword, height, width, amount):
        path = os.path.join(settings.googled_dir, keyword)
        os.makedirs(path, exist_ok=True)
        self.download_images(keyword=keyword, path=path, amount=amount)

    def download_images(self, keyword, path, amount):
        image_downloader = ImageDownloader()
        image_downloader.get_google_images(keyword=keyword, path=path, amount=amount)