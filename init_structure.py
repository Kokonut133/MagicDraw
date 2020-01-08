import settings
import os
import tarfile
import requests
import zipfile
import gc


def download_and_extract(path, url):
    gc.collect()
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        temp = "/tmp/" + url.split("/")[-1]
        with open(temp, 'wb') as f:
            f.write(response.raw.read())

        if temp.endswith("tar.gz"):
            tar = tarfile.open(temp, "r:gz")
            tar.extractall(path=path)
            tar.close()
        elif temp.endswith("tar"):
            tar = tarfile.open(temp, "r:")
            tar.extractall(path=path)
            tar.close()
        elif temp.endswith("zip"):
            with zipfile.ZipFile(temp, 'r') as zip_ref:
                zip_ref.extractall(path=path)
        print("Done downloading and extracting: " + url)
    else:
        print("Request failed")


if __name__ == '__main__':
    os.makedirs(os.path.join(settings.res_dir), exist_ok=True)
    os.makedirs(os.path.join(settings.img_dir), exist_ok=True)
    os.makedirs(os.path.join(settings.dataset_dir), exist_ok=True)

    get_coco = False
    get_pix2pix = False


    if "coco" not in os.listdir(settings.dataset_dir):
        get_coco = True if input("Get coco dataset? (y/n)") == "y" else False
    if "pix2pix" not in os.listdir(settings.dataset_dir):
        get_pix2pix = True if input("Get pix2pix dataset? (y/n)") == "y" else False

    if get_coco:
        download_and_extract(path=os.path.join(settings.dataset_dir,"coco", "images"), url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        download_and_extract(path=os.path.join(settings.dataset_dir,"coco","labels"), url="http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip")

    if get_pix2pix:
        tmp_path = os.path.join(settings.dataset_dir, "pix2pix")
        download_and_extract(path=tmp_path, url="https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz")
        download_and_extract(path=tmp_path, url="https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz")
        download_and_extract(path=tmp_path, url="https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz")
        download_and_extract(path=tmp_path, url="https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2handbags.tar.gz")
        download_and_extract(path=tmp_path, url="https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz")

