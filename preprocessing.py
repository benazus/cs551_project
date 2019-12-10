import os
import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from PIL import Image

root_dir = "/home/bb/BilkentUniversity/Fall2019/cs551/project/dataset/"
data_dir = root_dir + "files/"

"""
PIL image I/O
------------------
depth -> 240 x 320
    Image.fromarray(np.reshape(np.array(np.fromfile("/home/bb/BilkentUniversity/Fall2019/cs551/project/dataset/files/0-depth.bin", dtype = np.int16), dtype = np.int32), (240, 320)))

conf -> 240 x 320
    Image.fromarray(np.reshape(np.array(np.fromfile("/home/bb/BilkentUniversity/Fall2019/cs551/project/dataset/files/0-conf.bin", dtype = np.int16), dtype = np.int32), (240, 320)))

color -> 480 x 640 x 3
    img = Image.open("/home/bb/BilkentUniversity/Fall2019/cs551/project/dataset/files/0-color.png").convert('LA')
"""


# Reformat the folder structure of the dataset & rename samples with a global id
def transfer_dataset():
    global_id = 0
    new_path = data_dir
    labels = open(root_dir + "labels.csv", "w")

    for subject in range(1,5):
        for gesture in range(1,12):
            for sample in range(1,31):
                path = root_dir + "acquisitions/S" + str(subject) + "/G" + str(gesture) + "/"
                rgb = path + str(sample) + "-color.png"
                conf = path + str(sample) + "-conf.bin"
                depth = path + str(sample) + "-depth.bin"
                os.rename(rgb, new_path + "/" + str(global_id) + "-color.png")
                os.rename(conf, new_path + "/" + str(global_id) + "-conf.bin")
                os.rename(depth, new_path + "/" + str(global_id) + "-depth.bin")
                labels.write(str(global_id) + "\t" + str(gesture) + "\n")
                global_id += 1

    labels.close()

# Calculates the mean & stddev of the dataset, remember NOT to run this with normalized data
def dataset_mean_std():
    dataset = PatternDataset(data_dir, root_dir + "labels.csv")
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True
    )
    mean_rgb = 0.
    std_rgb = 0.

    mean_depth = 0.
    std_depth = 0.

    mean_conf = 0.
    std_conf = 0.

    nb_samples = 0.
    for data in loader:
        batch_size = data[0].size(0)
        rgb = data[0].view(batch_size, data[0].size(2), data[0].size(3))
        depth = data[1].view(batch_size, data[1].size(2), data[1].size(3))
        conf = data[2].view(batch_size, data[2].size(2), data[2].size(3))

        mean_rgb += rgb.mean()
        std_rgb += rgb.std()
        mean_depth = depth.mean()
        std_depth = depth.std()
        mean_conf = conf.mean()
        std_conf = conf.std()
        nb_samples += batch_size

    mean_rgb /= nb_samples
    std_rgb /= nb_samples
    mean_depth /= nb_samples
    std_depth /= nb_samples
    mean_conf /= nb_samples
    std_conf /= nb_samples
    return mean_rgb, std_rgb, mean_depth, std_depth, mean_conf, std_conf

class PatternDataset(Dataset):
    def __init__(self, data_dir, labels, rgb_mean = 3.5167, rgb_std = 1.0562, depth_mean = 0.7755, 
        depth_std = 0.2604, conf_mean = 0.0523, conf_std = 0.1067, concatenate_channels = False):
        tmp = np.loadtxt(labels)
        self.ids = tmp[:, 0]
        self.labels = tmp[:, 1]
        self.data_dir = data_dir
        self.concatenate_channels = concatenate_channels
        self.transform_rgb = transforms.Compose([transforms.Resize((60,80)), transforms.ToTensor(), transforms.Normalize((rgb_mean,), (rgb_std,))])
        self.transform_depth = transforms.Compose([transforms.Resize((60,80)), transforms.ToTensor(), transforms.Normalize((depth_mean,), (depth_std,))])
        self.transform_conf = transforms.Compose([transforms.Resize((60,80)), transforms.ToTensor(), transforms.Normalize((conf_mean,), (conf_std,))])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        fname = self.data_dir + str(index)
        rgb = Image.fromarray(np.array(np.mean(np.asarray(Image.open(fname + "-color.png")), 1), dtype = np.float64))
        depth = Image.fromarray(np.reshape(np.array(np.fromfile(fname + "-depth.bin", dtype = np.int16), dtype = np.float64), (240, 320)))
        conf = Image.fromarray(np.reshape(np.array(np.fromfile(fname + "-conf.bin", dtype = np.int16), dtype = np.float64), (240, 320)))
        return self.transform_rgb(rgb), self.transform_depth(depth), self.transform_conf(conf), torch.from_numpy(np.asarray(self.labels[index]))