


__author__="cstur"

import datetime
import logging
import os
from pathlib import Path

import settings


class Network_handler():

    @staticmethod
    def create_result_dir(data_dir):
        result_dir = os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem,
                                  str(datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")))
        os.makedirs(result_dir, exist_ok=True)
        checkpoint_path = os.path.join(result_dir, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        logging.basicConfig(filename=os.path.join(result_dir, "log.txt"), level=logging.INFO, filemode="w")
        return checkpoint_path, result_dir

    @staticmethod
    def get_d_g_paths(data_dir):
        list_of_folders = os.listdir(os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem))
        potential_folders = []
        for folder in list_of_folders:  # remove with no checkpoints
            if len(os.listdir(os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem, folder,
                    "checkpoints")))!=0:
                potential_folders.append(folder)
        if not potential_folders:
            print("No previous weights found!")
        latest_folder = max([datetime.datetime.strptime(i, "%Y-%m-%d-%H-%M-%S") for i in potential_folders])
        goal_folder = os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem,
            latest_folder.strftime("%Y-%m-%d-%H-%M-%S"), "checkpoints")
        discriminator_path = max(
            [os.path.join(goal_folder, d) for d in os.listdir(goal_folder) if "discriminator" in d],
            key=os.path.getctime)
        generator_path = max([os.path.join(goal_folder, d) for d in os.listdir(goal_folder) if "generator" in d],
            key=os.path.getctime)

        return discriminator_path, generator_path
