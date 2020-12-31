import torch
import os
import cv2
import numpy as np
from torch.utils import data
import torchvision
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

np.random.seed(555)
torch.manual_seed(555)
torch.cuda.manual_seed(555)


def load_dataset_indices(data_dir):
    normal_path = os.path.join(data_dir, 'normal')
    abnormal_path = os.path.join(data_dir, 'abnormal')
    normal_files = glob.glob(os.path.join(normal_path, '0*'))
    abnormal_files = glob.glob(os.path.join(abnormal_path, '0*'))

    normal_indices, abnormal_indices = [], []
    for normal_file in normal_files:
        normal_indices.append({'file_dir': normal_file, 'label': 1})
    for abnormal_file in abnormal_files:
        abnormal_indices.append({'file_dir': abnormal_file, 'label': 0})
    return normal_indices, abnormal_indices


class load_data(Dataset):
    
    def __init__(self, indices) -> None:
        super().__init__()
        self.indices = indices
        self.transform = ToTensor()

    def __getitem__(self, index):
        img_index = self.indices[index]
        img_dir, label = img_index['file_dir'], img_index['label']
        label = img_index['label']
        img = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)

        return img_tensor, label

    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    normal_indices, abnormal_indices = load_dataset_indices('../data')
    ratio = 0.7
    train_indices = np.random.choice(abnormal_indices, size=len(abnormal_indices) * ratio, replace=False)
    test_indices = list(set(abnormal_indices) - set(train_indices))
    train_indices = train_indices + normal_indices
    
    # data
    train_set = load_data(train_indices)
    test_set = load_data(test_indices)
    train_loader = DataLoader(train_set, batch_size=32, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=4)

    # model
    loss_fn = torch.nn.CrossentropyLoss()
    model = torchvision.models.resnet50(pretrained=False)
    model.to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # train
    for epoch in range(50):
        running_loss = 0
        batch = 0
        for _, data in enumerate(train_loader):
            imgs, labels = data
            optimizer.zero_grad()
            logits = model(imgs.cuda())
            loss = loss_fn(logits, labels.long().cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch += 1
        loss_avg = running_loss / batch
        print('epoch: {} loss: {:.4f}'.format(epoch, loss_avg))
