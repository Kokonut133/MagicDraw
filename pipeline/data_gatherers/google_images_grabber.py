from google_images_download import google_images_download
import os
from PIL import Image

__author__="cstur"

class DataGatherer():
    def __init__(self):
        pass

    def resizeFolder(self, input_dir, new_width=256):
        imgs = os.listdir(input_dir)
        print("Found files:", len(imgs), "; Resizing all images be %d pixels wide" % new_width)

        for img in imgs:
            pic = Image.open(input_dir + "/" + img)
            width, height = (pic).size
            print(width, " ", height)
            new_pic = pic.resize((new_width, new_width))
            new_pic.save(input_dir + "/resized/" + img)

    def download_images(self, keyword):
        arguments = {"keywords": keyword, "format": "jpg", "limit": 4,
                     "print_urls": True, "size": "medium", "aspect_ratio": "panoramic"}

        response = google_images_download.googleimagesdownload()

        try:
            response.download(arguments)

        except FileNotFoundError:
            arguments = {"keywords": keyword, "format": "jpg", "limit": 4,
                         "print_urls": True, "size": "medium"}
            try:
                response.download(arguments)
            except:
                pass