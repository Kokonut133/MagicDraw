from google_images_download import google_images_download
import os
from PIL import Image

__author__="cstur"

class DataGatherer():
    def resizeFolder(input_dir, new_width=256):
        imgs = os.listdir(input_dir)
        print("Found files:", len(imgs), "; Resizing all images be %d pixels wide" % new_width)

        # if not os.path.exists(input_dir+"/resized/"):
        #    os.makedirs(input_dir+"/resized/")

        for img in imgs:
            pic = Image.open(input_dir + "/" + img)
            width, height = (pic).size
            print(width, " ", height)
            new_pic = pic.resize((new_width, new_width))
            new_pic.save(input_dir + "/resized/" + img)

    keywords = ["water", "fire"]
    quantity = 3

    response = google_images_download.googleimagesdownload()
    for word in keywords:
        absolute_image_paths = response.download(
            {"keywords": word, "limit": quantity, "chromedriver": '/home/christian/myPrograms/chromedriver'})
        resizeFolder(os.getcwd() + "/downloads/" + word)