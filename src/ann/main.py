import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import os, random, time
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image as Img
from Utils import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from transformer_classifier import make_cls

random.seed(10)
torch.manual_seed(10)

cls = ['DriveDowntown', 'forest', 'snow', 'VolcanicEruption', 'Waves']
CLASS_NUM = 5
EPOCH = 60
BATCH_SIZE = 8
LR = 1e-5  # improve: 5e-5   basic: 1e-5
WEIGHT_DECAY = 0.
EARLY_STOP = True
use_tensorboard = True
PATIENCE = 60
dataset_root = '../../dataset/'
h, w = 200, 320

N = 4  # basic: 2
D_MODEL = 128  # basic: 128
D_FF = 256  # basic: 256

device = torch.device("cuda")

check_only = False

MODEL_NAME = 'improve_v2'


# MODEL_NAME = 'basic_model'


def main():
    train(MODEL_NAME)
    # test('./models/{}.pth'.format(MODEL_NAME))


def model_2():
    net = ANN2()
    return net


def get_model(model_name):
    model_type = model_name.split('_')[0]
    if model_type == 'basic':
        transformer = make_cls(src_dim=3 * w, seq_len=5 * h, cls_num=CLASS_NUM, N=N, d_model=128, d_ff=D_FF, h=2,
                               dropout=0.1)
        model = ANN_Basic(transformer=transformer)
    elif model_type == 'improve':
        model = ANN2()
    return model


def train(model_name, ctn=False):
    writer = SummaryWriter('log/' + model_name + '/')

    if not os.path.exists('./ckpt/' + model_name):
        os.mkdir('./ckpt/' + model_name)

    dataset_train = MyDataset(dataset_root + 'train', train=True)
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    dataset_valid = MyDataset(dataset_root + 'valid', train=False)
    valid_loader = DataLoader(
        dataset=dataset_valid,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    dataset_test = MyDataset(dataset_root + 'test', train=False)
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # Model, optimizer, loss function, earlystopping
    # net = model_basic().to(device)
    net = get_model(model_name).to(device)
    if ctn == True:
        net.load_state_dict(torch.load('models/' + model_name + '.pth'))
    print(net)
    # optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.NLLLoss()
    early_stopping = EarlyStopping(PATIENCE, verbose=False)

    check_model(net)
    if check_only == True:
        return

    # Training and validating
    for epoch in range(EPOCH):
        # if epoch > 0 and epoch % 10 == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.5

        net.train()
        running_loss = 0.0
        step_cnt = 0
        corr_cnt = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # print(b_x.shape)
            out = net(b_x)
            # print(out.shape)
            loss = loss_func(out, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step_cnt += 1

            # training acc
            pred_out = torch.max(torch.softmax(out, dim=1), 1)[1]
            pred_y = pred_out.data.cpu().numpy().squeeze()
            target_y = b_y.data.cpu().numpy()
            accuracy = sum(pred_y == target_y) / BATCH_SIZE
            corr_cnt += sum(pred_y == target_y)

            # print('Epoch: {} | Step: {} / {} | Accuracy: {:.4f} | Loss: {:.4f}'.format(
            #     epoch+1, step+1, len(train_loader), accuracy, loss.item()
            # ))

        avg_train_loss = running_loss / step_cnt
        train_acc = corr_cnt / len(dataset_train)
        print('Epoch: {} | Accuracy: {:.4f} | Training Loss: {:.4f}'.format(epoch + 1, train_acc, avg_train_loss),
              end='')

        # Validating
        running_loss = 0.0
        step_cnt = 0
        corr_cnt = 0
        net.eval()
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(valid_loader):
                out = net(batch_x.to(device))
                loss = loss_func(out, batch_y.to(device))
                running_loss += loss.item()
                step_cnt += 1

                pred = torch.max(torch.softmax(out, dim=1), dim=1)[1]
                pred = pred.data.cpu().numpy().squeeze()
                batch_y = batch_y.data.cpu().numpy()
                corr_cnt += sum(pred == batch_y)
        valid_acc = corr_cnt / len(dataset_valid)
        avg_valid_loss = running_loss / step_cnt
        print(' | Validation Loss: {:.4f} | Accuracy: {:.4f}'.format(avg_valid_loss, valid_acc), end='')

        # Validating by test data
        running_loss = 0.0
        step_cnt = 0
        corr_cnt = 0
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(test_loader):
                out = net(batch_x.to(device))
                loss = loss_func(out, batch_y.to(device))
                running_loss += loss.item()
                step_cnt += 1

                pred = torch.max(torch.softmax(out, dim=1), dim=1)[1]
                pred = pred.data.cpu().numpy().squeeze()
                batch_y = batch_y.data.cpu().numpy()
                corr_cnt += sum(pred == batch_y)
        test_acc = corr_cnt / len(dataset_valid)
        avg_test_loss = running_loss / step_cnt
        print(' | Test Loss: {:.4f} | Accuracy: {:.4f}'.format(avg_test_loss, test_acc))

        # Visualization
        if use_tensorboard:
            writer.add_scalars('Training Loss Graph', {'train_loss': avg_train_loss,
                                                       'validation_loss': avg_valid_loss,
                                                       'test_loss': avg_test_loss}, epoch + 1)
            writer.add_scalars('Training Acc Graph', {'train_acc': train_acc,
                                                      'validation_acc': valid_acc,
                                                      'test_acc': test_acc}, epoch + 1)

        # Save checkpoints:
        # if (epoch + 1) % 10 == 0:
        #     torch.save(net, './ckpt/' + model_name + '/' + str(epoch + 1) + '.pth')

        time.sleep(0.5)

        if EARLY_STOP:
            early_stopping(avg_valid_loss, net)
            if early_stopping.early_stop == True:
                print("Early Stopping!")
                break

    # save models in designated location
    if EARLY_STOP:
        import shutil
        shutil.move('checkpoint.pth', 'models/' + model_name + '.pth')


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
            tv.transforms.Normalize([0.3840, 0.3828, 0.3885], [0.2694, 0.2637, 0.2758]),
        ])

    def __getitem__(self, idx):
        filename, label = self.filenames[idx]
        imgs = []
        t = random.random()
        for fi in filename[:5]:
            img = Img.open(fi)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # use data augmentation if needed
            # if t > 0.5 and self.train:
            #    img = img.transpose(Img.FLIP_LEFT_RIGHT)

            # resize the image to the fixed size
            lr_img = img.resize((w, h))
            lr_img = self.trans(lr_img)
            imgs.append(lr_img[np.newaxis, :, :, :])

        # if the number of frames is less than the maximum, supplement zeros
        if len(filename) < 5:
            n = 5 - len(filename)
            temp = Img.new('RGB', (w, h), (0, 0, 0))
            temp = tv.transforms.ToTensor()(temp)[np.newaxis, :, :, :]
            imgs.append(temp.repeat(n, 1, 1, 1))

        return torch.cat(imgs, 0), label

    def __len__(self):
        return len(self.filenames)


class MyDataset_1(Dataset):
    def __init__(self, train_data_path, train=True):
        '''
        构造方法
        '''
        self.folder = train_data_path
        # self.filenames = os.listdir(train_data_path)
        # self.img_nums = len(self.filenames)
        self.filenames = []
        self.train = train
        for root, dirs, files in os.walk(train_data_path):
            y = -1
            for c_name in dirs:
                if not c_name in ['BoilingWater', 'DriveCountryside', 'DriveDowntown', 'ForestFire']:
                    # continue
                    pass
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
                        imgs = sorted(imgs, key=lambda x: int(x.split('.')[0]))
                        for img in imgs:
                            lst.append(os.path.join(d_path, img))
                        self.filenames.append([lst[:], y])
                    break
            break
        self.trans = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            # DEBUG
            # tv.transforms.RandomHorizontalFlip(0.5),

            # DEBUG
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) if train else tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__old(self, idx):
        filename, label = self.filenames[idx]
        # print(filename, label)
        imgs = []
        for fi in filename[:15]:
            img = Img.open(fi)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            lr_img = img.resize((224, 224))
            h, w = lr_img.size

            # Data augmentation
            lr_img = self.trans(lr_img)

            imgs.append(lr_img[np.newaxis, :, :, :])
        if len(filename) < 15:
            n = 15 - len(filename)
            temp = Img.new('RGB', (h, w), (0, 0, 0))

            temp = TF.to_tensor(temp)[np.newaxis, :, :, :]
            imgs.append(temp.repeat(n, 1, 1, 1))
        return torch.cat(imgs, 0), label

    def __getitem__(self, idx):
        filename, label = self.filenames[idx]
        # print(filename, label)
        img_seq_list = []
        for fi in filename[:15]:
            img = Img.open(fi)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize((w, h))
            # h, w = img.size

            # Convert to tensor, value in range [0, 1]
            t = TF.to_tensor(img)

            img_seq_list.append(t)
        if len(filename) < 5:
            n = 5 - len(filename)
            temp = Img.new('RGB', (w, h), (0, 0, 0))
            for i in range(n):
                t = TF.to_tensor(temp)
                img_seq_list.append(t)

        # Augmentation
        if self.train:
            self.augment(img_seq_list)
        # else:
        #     for i in range(len(img_seq_list)):
        #         img_seq_list[i] = T.RandomCrop(size=(224, 224))(img_seq_list[i])

        # Normalization
        # for img in img_seq_list:
        #     TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

        imgs_tensor = torch.stack(img_seq_list)
        return imgs_tensor, label

    def augment(self, imgs):
        '''
        Apply the same data augmentation methods to a list of pictures
        In place operation for all images
        imgs: a list of images, from a save video
        '''
        # Random horizontal flipping
        if random.random() > 0.5:
            for i in range(len(imgs)):
                imgs[i] = TF.hflip(imgs[i])

        # # Rotate
        # d = random.uniform(-20, 20)
        # for i in range(len(imgs)):
        #     imgs[i] = TF.rotate(imgs[i], d)

        # # Random crop
        # i, j, h, w = T.RandomCrop.get_params(imgs[0], output_size=(224, 224))  # original: (270, 480)
        # for i in range(len(imgs)):
        #     imgs[i] = TF.crop(imgs[i], i, j, h, w)

        # #
        # i, j, h, w = T.RandomResizedCrop.get_params(imgs[0], scale=[0.5, 1.0], ratio=[0.75, 1.33])  # original: (270, 480)
        # for i in range(len(imgs)):
        #     imgs[i] = TF.resized_crop(imgs[i], i, j, h, w, (224,224))

        # # Adjust contrast
        # factor = random.uniform(0.5, 1.5)
        # for i in range(len(imgs)):
        #     imgs[i] = TF.adjust_contrast(imgs[i], contrast_factor=factor)

        # # Adjust brightness
        # factor = random.uniform(0.5, 1.5)
        # for i in range(len(imgs)):
        #     imgs[i] = TF.adjust_brightness(imgs[i], factor)

        # # Adjust saturation
        # factor = random.uniform(0.5, 1.5)
        # for i in range(len(imgs)):
        #     imgs[i] = TF.adjust_saturation(imgs[i], factor)

        # # Guassian blur
        # p = random.random()
        # if p < 5:
        #     for i in range(len(imgs)):
        #         imgs[i] = TF.gaussian_blur(imgs[i], kernel_size=[5, 5])

        # # Random erase
        # t1, t2, t3, t4, t5 = T.RandomErasing.get_params(imgs[0], scale=[0.02, 0.1], ratio=[0.3, 1.3])
        # for i in range(len(imgs)):
        #     TF.erase(imgs[i], t1, t2, t3, t4, t5, inplace=True)

    def __len__(self):
        return len(self.filenames)


class ANN_Basic(nn.Module):
    def __init__(self, transformer):
        super(ANN_Basic, self).__init__()
        self.transformer = transformer

    def forward(self, x):
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(x.shape[0], 5 * h, 3, w)
        x = x.view(x.shape[0], 5 * h, 3 * w)

        x = self.transformer(x)
        return x

    # def load_state_dict(self, state_dict, strict: bool = True):
    #     self.transformer.load_state_dict(state_dict, strict=strict)


class ANN2(nn.Module):
    def __init__(self):
        super(ANN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)

        self.transformer = make_cls(src_dim=960, seq_len=250, cls_num=CLASS_NUM, N=N, d_model=D_MODEL, d_ff=D_FF, h=2)

    def forward(self, x):
        "Take in and process masked src and target sequences."
        x = x.permute(1, 0, 2, 3, 4)
        t = []
        for y in x:
            y = F.relu(self.conv1(y))
            y = F.relu(self.conv2(y))
            y = F.max_pool2d(y, kernel_size=2)
            y = F.relu(self.conv3(y))
            y = F.relu(self.conv4(y))
            y = F.max_pool2d(y, kernel_size=2)
            y = y.permute(0, 2, 3, 1)
            y = y.reshape(y.shape[0], y.shape[1], -1)
            t.append(y)
        x = torch.cat(t, dim=1)
        x = F.dropout(x, p=0.5)
        x = self.transformer(x)
        return x


def check_model(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Totalparams:', format(pytorch_total_params, ','))
    print('Trainableparams:', format(pytorch_train_params, ','))


def test(test_model_path):
    model_name = test_model_path.split('/')[-1].split('.')[0]
    model = get_model(model_name).to(device)
    # model = torch.load(test_model_path)
    model.load_state_dict(torch.load(test_model_path), strict=True)

    dataset_test = MyDataset(dataset_root + 'test', train=False)
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.NLLLoss()

    corr_cnt = 0
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for step, (b_x, b_y) in enumerate(test_loader):
            out = model(b_x.to(device))
            print(out)
            print(b_y)
            loss = loss_func(out, b_y.to(device))
            print(loss)
            running_loss += loss.item()

            pred = torch.max(torch.softmax(out, dim=1), 1)[1]
            pred = pred.data.cpu().numpy().squeeze()
            b_y = b_y.data.cpu().numpy()
            corr_cnt += sum(pred == b_y)

    print(len(test_loader))
    test_acc = corr_cnt / len(dataset_test)
    avg_test_loss = running_loss / len(test_loader)
    print('Tesing Dataset Loss: {} | Accuracy: {}'.format(avg_test_loss, test_acc))

    return avg_test_loss, test_acc


if __name__ == '__main__':
    main()
