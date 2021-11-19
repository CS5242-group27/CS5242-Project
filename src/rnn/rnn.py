import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch.nn as nn
import os, random
from PIL import Image as Img
from Utils import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import numpy as np

cls = ['DriveDowntown', 'forest', 'snow', 'VolcanicEruption', 'Waves']


class MyDataset(Dataset):
    def __init__(self, data_path, train=True):
        self.folder = data_path
        self.filenames = []
        self.train = train
        # loop over the dataset and save the paths of frame sequences and their labels
        for root, dirs, files in os.walk(data_path):
            y = -1
            for c_name in dirs:
                for i in range(len(cls)):
                    if cls[i] == c_name:
                        y = i
                        break
                c_path = os.path.join(root, c_name)
                for r, ds, fs in os.walk(c_path):
                    for d in ds:
                        lst = []
                        d_path = os.path.join(c_path, d)
                        imgs = os.listdir(d_path)
                        # make sure the order of frames is the same as the video
                        imgs = sorted(imgs, key=lambda x: int(x.split('.')[0]))
                        for img in imgs:
                            lst.append(os.path.join(d_path, img))
                        self.filenames.append([lst[:], y])
                    break
            break
        self.trans = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) 

    def __getitem__(self, idx):
        filename, label = self.filenames[idx]
        imgs = []
        for fi in filename[:5]:
            img = Img.open(fi)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # resize the image to the fixed size
            lr_img = img.resize((48, 27))
            h, w = lr_img.size
            lr_img = self.trans(lr_img)
            imgs.append(lr_img[np.newaxis, :, :, :])
        # if the number of frames is less than the maximum, supplement zeros
        if len(filename) < 5:
            n = 5-len(filename)
            temp = Img.new('RGB', (h, w), (0,0,0))
            temp = tv.transforms.ToTensor()(temp)[np.newaxis, :, :, :]
            imgs.append(temp.repeat(n, 1, 1, 1))
        # concat 5 images
        return torch.cat(imgs, 0), label

    def __len__(self):
        return len(self.filenames)

class RNN(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=3888,
            hidden_size=200,        
            num_layers=1,       
            batch_first=True,
        )
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, num_cls)
    def forward(self, input):
        b, t, c, h, w = input.shape
        x = input.view(b, t, 3888)
        x, (h_n, h_c) = self.lstm(x, None)
        # FC layers
        # choose RNN_out at the last time step
        x = self.fc1(x[:, -1, :])
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x

CLASS_NUM = 5
EPOCH = 50
BATCH_SIZE = 2
LR = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.
PATIENCE = 6

def train(model_name, ctn=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter('log/' + model_name + '/')

    # should modify data path if you want to run it.
    dataset_train = MyDataset('../../data/train')
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    # should modify data path if you want to run it.
    dataset_valid = MyDataset('../../data/valid', train=False)
    valid_loader = DataLoader(
        dataset=dataset_valid,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Model, optimizer, loss function, earlystopping
    net = RNN(CLASS_NUM).to(device)
    if ctn == True:
        net.load_state_dict(torch.load('models/' + model_name + '.pth'))
    print(net)
    for i in net.modules():
        print('-', i)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_func = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(PATIENCE, verbose=True)

    # Training and validating
    for epoch in range(EPOCH):
        running_loss = 0.0
        step_cnt = 0
        corr_cnt = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            out = net(b_x)
            loss = loss_func(out, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step_cnt += 1

            # Compute Accuracy
            pred_out = torch.max(torch.softmax(out, dim=1), 1)[1]
            pred_y = pred_out.data.cpu().numpy().squeeze()
            target_y = b_y.data.cpu().numpy()
            accuracy = sum(pred_y == target_y) / BATCH_SIZE
            corr_cnt += sum(pred_y == target_y)

            # print result
            print('Epoch: {} | Step: {} / {} | Accuracy: {:.4f} | Loss: {:.4f}'.format(
                epoch, step, len(train_loader), accuracy, loss.item()
            ))

        avg_train_loss = running_loss / step_cnt
        train_acc = corr_cnt / len(dataset_train)

        # Validating
        running_loss = 0.0
        step_cnt = 0
        corr_cnt = 0  
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(valid_loader):

                # Compute Loss
                out = net(batch_x.to(device))
                loss = loss_func(out, batch_y.to(device))
                running_loss += loss.item()
                step_cnt += 1

                # Compute Accuracy
                pred = torch.max(torch.softmax(out, dim=1), dim=1)[1]
                pred = pred.data.cpu().numpy().squeeze()
                batch_y = batch_y.data.cpu().numpy()
                corr_cnt += sum(pred == batch_y)
        valid_acc = corr_cnt / len(dataset_valid)
        avg_valid_loss = running_loss / step_cnt
        print('Validation Loss: {} | Accuracy: {}'.format(avg_valid_loss, valid_acc))

        # Visualization
        writer.add_scalars('Training Loss Graph', {'train_loss': avg_train_loss,
                                                   'validation_loss': avg_valid_loss}, epoch)
        writer.add_scalars('Training Acc Graph', {'train_acc': train_acc,
                                                  'validation_acc': valid_acc}, epoch)

        early_stopping(avg_valid_loss, net)
        if early_stopping.early_stop == True:
            print("Early Stopping!")
            break

    # save models in designated location
    import shutil
    shutil.move('checkpoint.pth', 'models/' + model_name + '.pth')


train('rnn')

def test(test_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNN(CLASS_NUM).to(device)
    model.load_state_dict(torch.load(test_model_path))

    dataset_test = MyDataset('../../data/test', train=False)
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    loss_func = nn.CrossEntropyLoss()

    corr_cnt = 0
    step_cnt = 0
    running_loss = 0.0
    with torch.no_grad():
        for step, (b_x, b_y) in enumerate(test_loader):
            # Compute loss
            out = model(b_x.to(device))
            loss = loss_func(out, b_y.to(device))
            running_loss += loss.item()
            step_cnt += 1

            # Compute Accuracy
            pred = torch.max(torch.softmax(out, dim=1), 1)[1]
            pred = pred.data.cpu().numpy().squeeze()
            b_y = b_y.data.cpu().numpy()
            corr_cnt += sum(pred == b_y)

    test_acc = corr_cnt / len(dataset_test)
    avg_test_loss = running_loss / step_cnt
    print('Tesing Dataset Loss: {} | Accuracy: {}'.format(avg_test_loss, test_acc))

    return avg_test_loss, test_acc



test('models/rnn.pth')
