import cv2
import os
import json
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch.cuda.amp import autocast, GradScaler


# coarse class
# class_index = {'背面白斑': [1], '背面附金': list(range(2, 6)), '背面基片混': [6], 
#                '背面漏镀': list(range(7, 10)), '背面缺件': list(range(10, 22)), 
#                '背面熵切引损伤': [22], 
#                '背面损伤': list(range(23, 60)), '背面锈点': list(range(60, 65)) + list(range(66, 75)), 
#                '背面虚焊': [75], '背面引纯低': [76], '背面引浅伤': list(range(77, 82)), 
#                '背面粘上膜': list(range(82, 101)),
#                '背面粘珠': list(range(101, 110)) + list(range(120, 172)), 
#                '反面打扁不良': list(range(172, 216)), '反面伤散热器': list(range(216, 222)), 
#                '反面污渍': list(range(222, 257)), '反面引线长打扁': list(range(257, 260)), 
#                '正面白斑': [260],
#                '正面打扁不良': list(range(261, 305)), '正面附金': list(range(305, 308)), 
#                '正面基片混': [308], '正面漏镀': list(range(309, 312)), '正面缺件': list(range(312, 324)), 
#                '正面伤散热器': list(range(324, 330)),
#                '正面熵切引损伤': [330], '正面损伤': list(range(331, 368)), 
#                '正面污渍': list(range(368, 404)), '正面锈点': list(range(404, 418)), 
#                '正面虚焊': [418], '正面引纯低': [419],
#                '正面引浅伤': list(range(420, 425)), '正面引线长打扁': list(range(425, 428)), 
#                '正面粘上膜': list(range(428, 447)), '正面粘珠': list(range(447, 494)), 
#                '正常': list(range(494, 538))}

# modified class
class_index = {'背面漏镀': list(range(7, 10)), 
               '背面损伤': list(range(23, 60)), '背面锈点': list(range(60, 65)) + list(range(66, 75)), 
               '背面虚焊': [75] + list(range(538, 548)),
               '背面粘珠': list(range(101, 110)) + list(range(120, 172)), 
               '反面污渍': list(range(222, 257)), 
               '正面白斑': [260],
               '正面打扁不良': list(range(261, 305)), '正面附金': list(range(305, 308)), 
               '正面基片混': [308], '正面漏镀': list(range(309, 312)), '正面缺件': list(range(312, 324)), 
               '正面伤散热器': list(range(324, 330)),
               '正面熵切引损伤': [330], '正面损伤': list(range(331, 368)), 
               '正面污渍': list(range(368, 404)), '正面锈点': list(range(404, 418)), 
               '正面虚焊': [418], '正面引纯低': [419],
               '正面引线损伤': list(range(420, 425)), '正面引线未打扁': list(range(425, 428)), 
               '正面粘上膜': list(range(428, 447)), '正面粘珠': list(range(447, 494)), 
               '正常': list(range(494, 538))}


def prepare_label(data_folder, class_index):
    dir_with_label = []
    cls_count = 0
    for cls in class_index.keys():
        file_range = class_index[cls]
        for f in file_range:
            dir_with_label.append({'file_dir': os.path.join(data_folder, '{:04d}.jpg'.format(f)),
                                   'label': cls_count})
        cls_count += 1
    return dir_with_label, list(class_index.keys())


class Det(nn.Module):
    def __init__(self, backbone, num_class):
        super().__init__()
        self.backbone = backbone
        self.linear = nn.Linear(1000, num_class)
    
    def forward(self, inputs):
        hidden = F.relu(self.backbone(inputs))
        outputs = self.linear(hidden)
        return outputs


class load_data(Dataset):

    def __init__(self, indices) -> None:
        super().__init__()
        self.indices = indices
        self.transforms = ToTensor()
        print('length: {}'.format(len(indices)))
    
    def __getitem__(self, index):
        annotation = self.indices[index]
        img_dir, label = annotation['file_dir'], annotation['label']
        img = cv2.imread(img_dir)
        img = self.transforms(img)

        return img, label, img_dir
    
    def __len__(self):
        return len(self.indices)


def evaluate(model, test_loader, epoch, name, is_final):
    with torch.no_grad():
        result = []
        mistake_results = {}
        correct = 0
        total = 0
        labels_all = {}
        mis_count = {}
        for data in test_loader:
            imgs, labels, img_dirs = data
            
            for item in labels:
                if int(item) not in labels_all:
                    labels_all[int(item)] = 1
                else:
                    labels_all[int(item)] += 1

            logits = model(imgs.cuda()).cpu()
            _, predicted_logits = torch.max(logits, 1)
            correct += (predicted_logits == labels).sum().item()
            total += imgs.shape[0]
            for idx in range(labels.shape[0]):
                if predicted_logits[idx] != labels[idx]:
                    if int(labels[idx]) not in mistake_results.keys():
                        mistake_results[int(labels[idx])] = []
                        mis_count[int(labels[idx])] = []
                    mistake_results[int(labels[idx])].append({'img_dir': str(img_dirs[idx]), 
                                                              'logits': int(predicted_logits[idx])})
                    mis_count[int(labels[idx])].append(int(predicted_logits[idx]))
            for idx in range(labels.shape[0]):
                result.append({'img_dir': str(img_dirs[idx]), 
                               'logits': int(predicted_logits[idx]), 
                               'label': int(labels[idx])})
        with open('../{}_result.json'.format(name), 'w') as f:
            config = json.dumps(result, indent=4)
            f.write(config)
        with open('../{}_misktake_result.json'.format(name), 'w') as f:
            config = json.dumps(mistake_results, indent=4)
            f.write(config)
        print('epoch {}: {} acc {:.2f} %'.format(epoch, name, (correct / total) * 100))
        if is_final:
            data_sorted = sorted(zip(labels_all.keys(), labels_all.values()))
            for item in data_sorted:
                print(item)
            for item in zip(mis_count.keys(), mis_count.values()):
                print(item)


if __name__ == "__main__":
    seed = np.random.randint(20000)
    print("random seed: {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
    data_folder = '../img'
    BATCH_SIZE = 32 
    LR = 0.0001
    L2 = 0.0001
    ratio = 0.7
    EPOCHS = 100

    data_anno, real_labels = prepare_label(data_folder, class_index)
    device = torch.device('cuda:0')

    backbone = models.resnet18(pretrained=False)
    model = Det(backbone, len(class_index))
    model.to(device)
    
    indices = list(range(len(data_anno)))
    
    train_indices = np.random.choice(indices, size=int(len(data_anno) * ratio), replace=False)
    test_indices = list(set(indices) - set(train_indices))

    train_anno = [data_anno[i] for i in train_indices]
    test_anno = [data_anno[i] for i in test_indices]

    print('loading train set and test set')
    train_set = load_data(train_anno)
    test_set = load_data(test_anno)

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), LR, weight_decay=L2)
    loss_fn = nn.CrossEntropyLoss()
    
    scaler = GradScaler()
    # training stage
    is_final = False
    for epoch in range(EPOCHS):
        running_loss = []
        
        for data in train_loader:
            images, labels, _ = data
            optimizer.zero_grad()
            with autocast():
                logits = model(images.to(device))
                loss = loss_fn(logits, labels.long().to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss.append(loss.item())
        print('epoch {}: loss {:.4f}'.format(epoch, np.array(running_loss).mean()))
        if epoch == EPOCHS - 1:
            is_final = True
        evaluate(model, train_loader, epoch, 'train', False)
        evaluate(model, test_loader, epoch, 'test', is_final)
    print("random seed: {}".format(seed))
    print(real_labels)
