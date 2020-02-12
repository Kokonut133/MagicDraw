import os
import skimage
import numpy as np
import tifffile
from PIL import Image


# applies global positive standardization (range 0:1)
def process(input_dir, output_dir):
    if len(os.listdir(input_dir)) == len(os.listdir(output_dir)):
        print("Already normalized dataset")
        return
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]

    # calculating mean and stddev
    means = [[], [], []]
    stddevs = [[], [], []]
    for file in files:
        image = skimage.io.imread(file)
        for i in range(0, 3):
            means[i].append(np.mean(image[:, :, i]))
            stddevs[i].append(np.std(image[:, :, i]))

    mean = []
    stddev = []
    for i in range(0, 3):
        mean.append(int(np.mean(means[i])))
        stddev.append(int(np.mean(stddevs[i])))
    print("Calculated means: " + str(mean) + " and stddevs: " + str(stddev))

    files = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    for file in files:
        input_file = os.path.join(input_dir, file)
        image = np.asarray(Image.open(input_file)).astype("float32")
        for i in range(0, 3):
            image[:, :, i] -= mean[i]
        image = np.interp(image, (image.min(), image.max()), (0, +1))
        output_file = os.path.join(output_dir, (str(file.split(".")[0])+".tif"))
        tifffile.imsave(output_file, image)

    print("Normalized dataset")


