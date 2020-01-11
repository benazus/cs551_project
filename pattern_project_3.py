import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from torchvision import transforms, utils
from PIL import Image
import shutil

data_dir = ""
root_dir = ""
all_labels_file = root_dir + "labels.csv"
train_data_dir = root_dir + "files/train/"
train_labels_path = root_dir + "files/train_labels.csv"
test_data_dir = root_dir + "files/test/"
test_labels_path = root_dir + "files/test_labels.csv"
model_path = root_dir + "models3/"
statistics_path = root_dir + "statistics3/"
figure_path = root_dir + "figures3/"
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

# Util functions
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
                labels.write(str(global_id) + "\t" + str(gesture - 1) + "\n")
                global_id += 1

    labels.close()

# Calculates the mean & stddev of the dataset, remember NOT to run this with normalized data
def dataset_mean_std():
    dataset = PatternDataset(data_dir, root_dir + "labels.csv")
    loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
    mean_rgb, std_rgb, mean_depth, std_depth, mean_conf, std_conf, nb_samples = 0., 0., 0., 0., 0., 0., 0.
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

# Split the dataset into train & test, since the size dataset size is small no validation set is formed
def train_test_split():
    test_ratio = 0.2
    file_content = np.loadtxt(all_labels_file, dtype=np.int32)
    all_ids, all_labels = file_content[:,0], file_content[:,1]
    train_sample_ids, train_sample_labels, test_sample_ids, test_sample_labels = [], [], [], []
    for label in range(11):
        label_sample_ids = all_ids[np.argwhere(all_labels == label)]
        label_sample_labels = all_labels[np.argwhere(all_labels == label)]

        test_indices = np.random.choice(label_sample_ids.shape[0], int(len(label_sample_ids) * test_ratio), replace=False)
        train_indices = np.array([i for i in range(len(label_sample_ids)) if i not in test_indices])
        train_sample_ids.extend(label_sample_ids[train_indices])
        train_sample_labels.extend(label_sample_labels[train_indices])
        test_sample_ids.extend(label_sample_ids[test_indices])
        test_sample_labels.extend(label_sample_labels[test_indices])

    train_sample_ids, train_sample_labels = np.array(train_sample_ids), np.array(train_sample_labels)
    test_sample_ids, test_sample_labels = np.array(test_sample_ids), np.array(test_sample_labels)

    with open(train_labels_path, "w") as f:
        for _, (sample_id, sample_label) in enumerate(zip(train_sample_ids, train_sample_labels)):
            sample_id = sample_id[0]
            sample_label = sample_label[0]
            src = root_dir + "files/"
            shutil.move(src + "{}-color.png".format(sample_id), train_data_dir + "{}-color.png".format(sample_id))
            shutil.move(src + "{}-conf.bin".format(sample_id), train_data_dir + "{}-conf.bin".format(sample_id))
            shutil.move(src + "{}-depth.bin".format(sample_id), train_data_dir + "{}-depth.bin".format(sample_id))
            f.write(str(sample_id) + " " + str(sample_label) + "\n")
    
    with open(test_labels_path, "w") as f:
        for _, (sample_id, sample_label) in enumerate(zip(test_sample_ids, test_sample_labels)):
            sample_id = sample_id[0]
            sample_label = sample_label[0]
            src = root_dir + "files/"
            shutil.move(src + "{}-color.png".format(sample_id), test_data_dir + "{}-color.png".format(sample_id))
            shutil.move(src + "{}-conf.bin".format(sample_id), test_data_dir + "{}-conf.bin".format(sample_id))
            shutil.move(src + "{}-depth.bin".format(sample_id), test_data_dir + "{}-depth.bin".format(sample_id))
            f.write(str(sample_id) + " " + str(sample_label) + "\n")

# Classes
class PatternDataset(Dataset):
    def __init__(self, data_dir, labels, rgb_mean = 3.5167, rgb_std = 1.0562, depth_mean = 0.7755, 
        depth_std = 0.2604, conf_mean = 0.0523, conf_std = 0.1067):
        tmp = np.loadtxt(labels)
        self.ids = np.array(tmp[:, 0], dtype = np.int32)
        self.labels = np.array(tmp[:, 1], dtype = np.int32)
        self.data_dir = data_dir
        self.transform_rgb = transforms.Compose([transforms.Resize((30,40)), transforms.ToTensor(), transforms.Normalize((rgb_mean,), (rgb_std,))])
        self.transform_depth = transforms.Compose([transforms.Resize((30,40)), transforms.ToTensor(), transforms.Normalize((depth_mean,), (depth_std,))])
        self.transform_conf = transforms.Compose([transforms.Resize((30,40)), transforms.ToTensor(), transforms.Normalize((conf_mean,), (conf_std,))])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        fname = self.data_dir + str(self.ids[index])
        rgb = Image.fromarray(np.array(np.mean(np.asarray(Image.open(fname + "-color.png")), 1), dtype = np.float64))
        depth = Image.fromarray(np.reshape(np.array(np.fromfile(fname + "-depth.bin", dtype = np.int16), dtype = np.float64), (240, 320)))
        conf = Image.fromarray(np.reshape(np.array(np.fromfile(fname + "-conf.bin", dtype = np.int16), dtype = np.float64), (240, 320)))
        return torch.cat((self.transform_rgb(rgb), self.transform_depth(depth), self.transform_conf(conf))).double(), torch.from_numpy(np.asarray(self.labels[index])).long()

class PatternModel(nn.Module):
    def __init__(self):
        super(PatternModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, padding=2)
        self.batchnorm3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, 5, padding=2)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 5, padding=2)
        self.batchnorm5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 5, padding=2)
        self.batchnorm6= nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 128, 5, padding=2)
        self.batchnorm7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 5, padding=2)
        self.batchnorm8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, 5, padding=2)
        self.batchnorm9 = nn.BatchNorm2d(128)
        
        self.conv10 = nn.Conv2d(128, 256, 5, padding=2)
        self.batchnorm10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, 5, padding=2)
        self.batchnorm11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, 5, padding=2)
        self.batchnorm12 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 11)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1,3,30,40)
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.max_pool2d(x, kernel_size = (2,2), stride=(2,2))
        x = F.dropout2d(x, training=True, p = 0.5)

        x = F.relu(self.conv4(x))
        x = self.batchnorm4(x)
        x = F.relu(self.conv5(x))
        x = self.batchnorm5(x)
        x = F.relu(self.conv6(x))
        x = self.batchnorm6(x)
        x = F.max_pool2d(x, kernel_size = (2,2), stride=(2,2))
        x = F.dropout2d(x, training=True, p = 0.5)

        x = F.relu(self.conv7(x))
        x = self.batchnorm7(x)
        x = F.relu(self.conv8(x))
        x = self.batchnorm8(x)
        x = F.relu(self.conv9(x))
        x = self.batchnorm9(x)
        x = F.max_pool2d(x, kernel_size = (2,2), stride=(2,2))
        x = F.dropout2d(x, training=True, p = 0.5)        

        x = F.relu(self.conv10(x))
        x = self.batchnorm10(x)
        x = F.relu(self.conv11(x))
        x = self.batchnorm11(x)
        x = F.relu(self.conv12(x))
        x = self.batchnorm12(x)
        x = F.max_pool2d(x, kernel_size = (2,2), stride=(2,2))
        x = F.dropout2d(x, training=True, p = 0.5)

        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True, p = 0.5)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=True, p = 0.5)

        x = self.softmax(self.fc3(x))
        return x

# Network functions
def train(lr, momentum, batch_size, epochs):   
    train_dataset = PatternDataset(train_data_dir, train_labels_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = PatternDataset(test_data_dir, test_labels_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = PatternModel()
    model = model.double()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    data_size = len(train_dataset)      
    print("Training starts...")
    for i in range(epochs):
        print("Epoch " + str(i))
        epoch_loss = 0
        running_loss = 0
        for j, data in enumerate(train_loader):
            pixels, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(pixels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += running_loss

            if j % 50 == 49:
                print('[%d, %5d] loss: %.3f' % (i + 1, j + 1, running_loss / 50))
                running_loss = 0
        losses.append(epoch_loss / data_size)

    print("Training complete.")
    path_to_model = model_path + "model_{}_{}_{}.pth".format(lr, momentum, epochs)
    torch.save(model.state_dict(), path_to_model)
    plt.figure(figsize=(20,10))
    plt.plot(list(range(1,epochs+1)), losses)
    plt.title("model_{}_{}_{}".format(lr, momentum, epochs))
    plt.savefig(figure_path + "model_{}_{}_{}.png".format(lr, momentum, epochs))
    return path_to_model

def model_statistics(path_to_model, lr, momentum, epochs):
    model = PatternModel()
    model.load_state_dict(torch.load(path_to_model))
    model = model.double()
    criterion = nn.CrossEntropyLoss()
    test_dataset = PatternDataset(test_data_dir, test_labels_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loss = 0
    data_size = len(test_dataset)
    correct, total = 0, 0
    labels_top, preds_top = [], []
    with torch.no_grad():
        for data in test_loader:
            pixels, labels = data[0], data[1]
            outputs = model(pixels)
            _, predicted = torch.max(outputs.data, 1)
            test_loss += criterion(outputs, labels).item()
            labels_top.extend(labels.numpy())
            preds_top.extend(predicted.numpy())
    overall_accuracy = accuracy_score(labels_top, preds_top, normalize=True)
    cms = multilabel_confusion_matrix(np.array(labels_top, dtype=np.int32), np.array(preds_top, dtype=np.int32), labels=list(range(7))) # tn fp fn tp
    tn, tp, fn, fp = cms[:, 0, 0], cms[:, 1, 1], cms[:, 1, 0], cms[:, 0, 1]
    true_positive_rate = np.around(tp / (tp + fn), 3)
    true_negative_rate = np.around(tn / (tn + fp), 3)
    false_positive_rate = np.around(fp / (fp + tn), 3)
    false_negative_rate = np.around(fn / (fn + tp), 3)
    classwise_accuracies = np.around((tp + tn) / (tp + tn + fp + fn), 3)
    stat_file = statistics_path + "statistics_{}_{}_{}.txt".format(lr, momentum, epochs)
    with open(stat_file, "w") as f:
        f.write("True Positives: {} {} {} {} {} {} {}\n".format(*tp))
        f.write("False Negatives: {} {} {} {} {} {} {}\n".format(*fn))
        f.write("False Positives: {} {} {} {} {} {} {}\n".format(*fp))
        f.write("True Negatives: {} {} {} {} {} {} {}\n".format(*tn))
        f.write("Test Accuracy: {}\n".format(round(overall_accuracy, 3)))
        f.write("Test Loss: {}\n".format(round(test_loss / data_size, 3)))
        f.write("Classwise Accuracies: {} {} {} {} {} {} {}\n".format(*classwise_accuracies))
        f.write("True Positive Rates: {} {} {} {} {} {} {}\n".format(*true_positive_rate))
        f.write("True Negative Rates: {} {} {} {} {} {} {}\n".format(*true_negative_rate))
        f.write("False Positive Rates: {} {} {} {} {} {} {}\n".format(*false_positive_rate))
        f.write("False Negative Rates: {} {} {} {} {} {} {}\n".format(*false_negative_rate))
    return overall_accuracy

def experiments():
    lrs = [0.005, 0.001, 0.0005] #[0.005, 0.001, 0.0005]
    momentums = [0.85, 0.9, 0.95]
    epochs = [50, 60, 70, 80, 90, 100]
    best_accuracy = -999
    best_lr = 0
    best_epochs = 0
    best_momentum = 0
    for lr in lrs:
        for momentum in momentums:
            for epoch in epochs:
                path_to_model = train(lr, momentum, 32, epoch)
                accuracy = model_statistics(path_to_model, lr, momentum, epoch)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_lr = lr
                    best_epochs = epoch
                    best_momentum = momentum
    print("accuracy, lr, epochs, momentum -> {}, {}, {}, {}".format(best_accuracy, best_lr, best_epochs, best_momentum))
experiments()