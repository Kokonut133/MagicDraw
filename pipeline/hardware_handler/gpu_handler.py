from tensorflow.python.client import device_lib

__author__="cstur"


def list_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    for device in local_device_protos:
        if device.device_type=="GPU":
            print(device.name+" "+device.physical_device_desc)
