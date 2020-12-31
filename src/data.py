import torch
import os
import cv2
import numpy as np
from torch.utils import data
import torchvision
import glob
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch.cuda.amp import autocast, GradScaler

np.random.seed(555)
torch.manual_seed(555)
torch.cuda.manual_seed(555)


def load_dataset_indices(data_dir):
    normal_path = os.path.join(data_dir, 'normal')
    abnormal_path = os.path.join(data_dir, 'abnormal')
    normal_files = glob.glob(os.path.join(normal_path, '*.jpg'))
    abnormal_files = glob.glob(os.path.join(abnormal_path, '*.jpg'))

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

        return img_tensor, label, img_dir

    def __len__(self):
        return len(self.indices)


def evaluate(model, test_loader, epoch, name):
    with torch.no_grad():
        result = []
        correct = 0
        total = 0
        for data in test_loader:
            imgs, labels, img_dirs = data
            logits = model(imgs.cuda()).cpu()
            _, predicted_logits = torch.max(logits, 1)
            correct += (predicted_logits == labels).sum().item()
            total += imgs.shape[0]
            for idx in range(labels.shape[0]):
                result.append({'img_dir': str(img_dirs[idx]), 'logits': int(predicted_logits[idx]), 'label': int(labels[idx])})
        with open('../{}_result.json'.format(name), 'w') as f:
            config = json.dumps(result, indent=4)
            f.write(config)
        print('epoch {}: {} acc {:.2f} %'.format(epoch, name, (correct / total) * 100))


if __name__ == '__main__':
    normal_indices, abnormal_indices = load_dataset_indices('../data')
    ratio = 0.7
    good_idx = list(range(len(abnormal_indices)))
    t_indices = list(np.random.choice(good_idx, size=int(len(good_idx) * ratio), replace=False))
    te_indices = list(set(good_idx) - set(t_indices))
    train_indices = [abnormal_indices[i] for i in t_indices]
    test_indices = [abnormal_indices[i] for i in te_indices]
    train_indices = train_indices + normal_indices
    
    # data
    train_set = load_data(train_indices)
    test_set = load_data(test_indices)
    train_loader = DataLoader(train_set, batch_size=16, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=16, num_workers=4)

    # model
    loss_fn = torch.nn.CrossEntropyLoss()
    model = torchvision.models.resnet50(pretrained=False)
    model.to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    scaler = GradScaler()
    print('training ...')
    # train
    for epoch in range(50):
        running_loss = 0
        batch = 0
        for _, data in enumerate(train_loader):
            imgs, labels = data[0], data[1]
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs.cuda())
                loss = loss_fn(logits, labels.long().cuda())
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            batch += 1
        loss_avg = running_loss / batch
        print('epoch: {} loss: {:.4f}'.format(epoch, loss_avg))
        evaluate(model, train_loader, epoch, 'train')
        evaluate(model, test_loader, epoch, 'test')
    
    torch.save(model, '../output/model.pth')
    print('Parameters saved.')
