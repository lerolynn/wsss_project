import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler
import numpy as np
from PIL import Image

import os
import time

from torch.optim import lr_scheduler
from pathlib import Path
from collections import OrderedDict

"""
CS701-Team 9: Multi-class classification model
Last upddate: 2021-10-12
Reference: https://blog.csdn.net/MiaoB226/article/details/89504131
"""

# check if gpu is available
use_gpu = torch.cuda.is_available()
evaluate = False
# model_selected = "alexnet"
# model_selected = "densenet"
model_selected = "resnext"

batch_size = 64

# Define the transformers
data_transforms = {
    'train': transforms.Compose([
        # resize the image into 256*256
        transforms.Resize(256),
        # crop the images into size 227*227
        transforms.RandomResizedCrop(227),
        # flip the image
        transforms.RandomHorizontalFlip(),
        # transform the image into tensor
        transforms.ToTensor(),
        # normalize the image
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

### Load Data
# 定义数据读入
def Load_Image_Information_Train(path):
    # 图像存储路径
    image_Root_Dir = r'data/train'
    # image_Root_Dir = r'public/img_dir/train'
    # 获取图像的路径
    iamge_Dir = os.path.join(image_Root_Dir, path)
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    return Image.open(iamge_Dir).convert('RGB')

def Load_Image_Information_Test(path):
    # 图像存储路径
    image_Root_Dir = r'data/test1'

    # image_Root_Dir = r'public/img_dir/train'
    # 获取图像的路径
    iamge_Dir = os.path.join(image_Root_Dir, path)
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    return Image.open(iamge_Dir).convert('RGB')

# 定义自己数据集的数据读入类 | Define the class to load the data
class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        # 打开存储图像名与标签的txt文件
        fp = open(txt, 'r')
        images = []
        labels = []

        # 将图像名和图像标签对应存储起来 Store the image name and the labels.
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels_data = [0]*103
            for l in information[1:len(information)]:
                labels_data[int(l)-1] = 1
            labels.append(labels_data)
            # labels.append([torch.nn.functional.one_hot(torch.tensor(float(l)-1, dtype=torch.int64), num_classes=104)
            #                for l in information[1:len(information)]
            #                ])
            # for l in information[1:len(information)]:
            #     l = torch.tensor(float(l), dtype=torch.int64)
                # print(torch.nn.functional.one_hot(l, num_classes=103))
                # labels.append([torch.nn.functional.one_hot(l, num_classes=103)])
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform  # TODO: what is the target_transform?
        self.loader = loader

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        imageName = self.images[item]
        label = self.labels[item]
        # 读入图像信息
        image = self.loader(imageName)
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        # 需要将标签转换为float类型，BCELoss只接受float类型
        label = torch.FloatTensor(label)
        return image, label

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)


# 生成Pytorch所需的DataLoader数据输入格式
train_Data = my_Data_Set(r'data/train_label.txt', transform=data_transforms['train'], loader=Load_Image_Information_Train)
val_Data = my_Data_Set(r'data/val_label.txt', transform=data_transforms['val'], loader=Load_Image_Information_Test)

# create the validation data
# Creating data indices for training and validation splits:
validation_split = 0.2
shuffle_dataset = True
random_seed= 42
dataset_size = len(train_Data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_DataLoader = DataLoader(train_Data,
                              batch_size=batch_size,
                              sampler=train_sampler)
validation_Dataloader = DataLoader(train_Data,
                                   batch_size=batch_size,
                                   sampler=valid_sampler)


# train_DataLoader = DataLoader(train_Data, batch_size=64, shuffle=True)
# to test the model on the public data
val_DataLoader = DataLoader(val_Data, batch_size=1)


dataloaders = {'train': train_DataLoader,
               'validation': validation_Dataloader,
               'val': val_DataLoader}
# 读取数据集大小
dataset_sizes = {'train': train_Data.__len__(),
                 'validation': 0.2*train_Data.__len__(), #  check the actual length
                 'val': val_Data.__len__()}

# save the name of validation pictures into txt file.
def ListFilesToTxt(dir, file, wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname = os.path.join(dir, name)
        if (os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname, file, wildcard, recursion)
        else:
            for ext in exts:
                if (name.endswith(ext)):
                    (filename, extension) = os.path.splitext(name)
                    file.write(name + "\n")
                    break
def ImageNameGenerator():
    dir = "data/test1"
    outfile = "data/val_label.txt"

    val_label_path = Path(outfile)
    val_label_path.touch(exist_ok=True)

    wildcard = ".jpg"

    file = open(outfile, "w")
    if not file:
        print("cannot open the file %s for writing" % outfile)
    ListFilesToTxt(dir, file, wildcard, 1)

    file.close()


# predict the classes of the pictures
def predict(input, model, device):
  # model.to(device)
  with torch.no_grad():
    input=input.to(device)
    out = model(input)

    accuracy_th = 0.5
    pred_result = torch.sigmoid(out)
    pred_result = out > accuracy_th
    pred_result = pred_result.float()
    pred_result_lst = torch.nonzero(pred_result)
    pre = []
    if len(pred_result_lst)==0:
        _, pred_max = torch.max(out.data, 1)  # the actual label number starts from 0
        pre.append(1+pred_max.item())
    else:

        for i in pred_result_lst:
            pre.append(1+i[1].item())  # the actual label number starts from 0
    return pre


# 计算准确率——方式1
# 设定一个阈值，当预测的概率值大于这个阈值，则认为这幅图像中含有这类标签
def calculate_acuracy_mode_one(model_pred, labels):
    model_pred = torch.nn.functional.softmax(model_pred[0], dim=0)
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    # 模型预测的结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_predict_num / target_one_num

    return precision.item(), recall.item()

# 计算准确率——方式2
# 取预测概率最大的前top个标签，作为模型的预测结果
def calculate_acuracy_mode_two(model_pred, labels):
    # 取前top个预测结果作为模型的预测结果
    precision = 0
    recall = 0
    top = 5
    # 对预测结果进行按概率值进行降序排列，取概率最大的top个结果作为模型的预测结果
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]
    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0,pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        # 对每一幅图像进行预测准确率的计算
        precision += true_predict_num / top
        # 对每一幅图像进行预测查全率的计算
        recall += true_predict_num / target_one_num
    return precision, recall


## Classification Model
# Train and evaluate the network (all layers participate in the training)
def train_model(model, criterion, optimizer, scheduler, num_epochs=300):
    # Sigmoid_fun = nn.Sigmoid()
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每训练一个epoch，验证一下网络模型
        for phase in ['train', 'val', 'test']:
        # for phase in ['train']:
        # for phase in ['train']:
            running_loss = 0.0
            running_precision = 0.0
            running_recall = 0.0
            batch_num = 0

            if phase == 'train':
                # 学习率更新方式
                scheduler.step()
                #  调用模型训练
                model.train()

                for data in dataloaders[phase]:
                    inputs, labels = data
                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # 梯度清零
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    # loss = criterion(Sigmoid_fun(outputs), labels)
                    loss = criterion(outputs, labels)
                    # 这里根据自己的需求选择模型预测结果准确率的函数
                    precision, recall = calculate_acuracy_mode_one(outputs, labels)
                    # precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(outputs), labels)
                    # precision, recall = calculate_acuracy_mode_two(Sigmoid_fun(outputs), labels)
                    running_precision += precision
                    running_recall += recall
                    batch_num += 1
                    # 反传梯度
                    loss.backward()
                    # 更新权重
                    optimizer.step()
                    # 计算一个epoch的loss值和准确率
                    running_loss += loss.item() * inputs.size(0)
            if phase == 'validation':
                # 取消验证阶段的梯度
                with torch.no_grad():
                    # 调用模型测试
                    model.eval()
                    # 依次获取所有图像，参与模型训练或测试
                    for data in dataloaders[phase]:
                        # 获取输入
                        inputs, labels = data
                        # 判断是否使用gpu
                        if use_gpu:
                            inputs = inputs.cuda()
                            labels = labels.cuda()

                        # 网络前向运行
                        outputs = model(inputs)
                        # 计算Loss值
                        # BCELoss的输入（1、网络模型的输出必须经过sigmoid；2、标签必须是float类型的tensor）
                        loss = criterion(Sigmoid_fun(outputs), labels)
                        # 计算一个epoch的loss值和准确率
                        running_loss += loss.item() * inputs.size(0)

                        # 这里根据自己的需求选择模型预测结果准确率的函数
                        precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(outputs), labels)
                        # precision, recall = calculate_acuracy_mode_two(Sigmoid_fun(outputs), labels)
                        running_precision += precision
                        running_recall += recall
                        batch_num += 1
            # 计算Loss和准确率的均值
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f} '.format(phase, epoch_loss))
            epoch_precision = running_precision / batch_num
            print('{} Precision: {:.4f} '.format(phase, epoch_precision))
            epoch_recall = running_recall / batch_num
            print('{} Recall: {:.4f} '.format(phase, epoch_recall))
            if (epoch+1) % 50 == 0:
                torch.save(model.state_dict(), 'The_' + str(epoch) + '_epoch_densenet121.pkl')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

# def test_model(model_name, device):
#     model.load_state_dict(
#         torch.load(model_name,
#                    map_location=torch.device(device)
#                    )
#     )  # load the weights in the models
#     fp = open("public/val_label.txt", 'r')
#     val_predection = open("public/val_prediction.txt", "w")
#
#     for line in fp:
#         line.strip('\n')
#         line.rstrip()
#         information = line.split()
#         images = (information[0])
#         img = Image.open("public/img_dir/test1/" + images).convert('RGB')
#         img = data_transforms['val'](img)
#         img = img.unsqueeze(0)
#         ans = predict(img, model, "cpu")
#         line = information[0]
#         for label in ans:
#             line += " " + str(label)
#         print(line)
#         val_predection.write(line + "\n")
#     val_predection.close()

# fine-tune models
if __name__ == '__main__':
    # TODO: replace AlexNet with a better pretrained model
    if model_selected == "alexnet":
        model = models.alexnet(pretrained=True)
        # 获取最后一个全连接层的输入通道数
        num_input = model.classifier[6].in_features
        # 获取全连接层的网络结构
        feature_model = list(model.classifier.children())
        # 去掉原来的最后一层
        feature_model.pop()
        # 添加上适用于自己数据集的全连接层
        # 260数据集的类别数
        # feature_model.append(nn.Linear(num_input, 260))
        # 103 classes
        feature_model.append(nn.Linear(num_input, 103))
        # 仿照这里的方法，可以修改网络的结构，不仅可以修改最后一个全连接层 还可以为网络添加新的层
        # 重新生成网络的后半部分
        model.classifier = nn.Sequential(*feature_model)
    if model_selected == "densenet121":
        model = models.densenet121(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 103)),
            # ('relu', nn.ReLU()),
            # ('fc2', nn.Linear(512, 103)),
            # ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        # model.classifier = nn.Linear(1024, 103)

    if model_selected == "resnext":
        model = models.resnext101_32x8d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 103)
        for param in model.parameters():
            param.requires_grad = False

    if use_gpu:
        model = model.cuda()

    if not evaluate:
        # criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()

        # 为不同层设定不同的学习率
        # fc_params = list(map(id, model.classifier[6].parameters()))
        fc_params = list(map(id, model.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
        params = [{"params": base_params, "lr": 0.0001},
                  {"params": model.parameters(), "lr": 0.001},]
                  # {"params": model.classifier[6].parameters(), "lr": 0.001}, ]

        optimizer_ft = torch.optim.SGD(params, momentum=0.9)

        # 定义学习率的更新方式，每5个epoch修改一次学习率
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
        train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=300)

    else:  # test the model on the public data given in the contest
        # test_model("The_9_epoch_model.pklThemodel_AlexNet.pkl", "cpu")
        model.load_state_dict(
            torch.load("The_9_epoch_model.pklThemodel_AlexNet.pkl",
                       map_location=torch.device('cpu')
                       )
        )  # load the weights in the models
        fp = open("data/val_label.txt", 'r')
        val_predection = open("data/val_prediction.txt", "w")

        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images = (information[0])
            img = Image.open("data/test1/" + images).convert('RGB')
            img = data_transforms['val'](img)
            img = img.unsqueeze(0)
            ans = predict(img, model, "cpu")
            line = information[0]
            for label in ans:
                line += " " + str(label)
            print(line)
            val_predection.write(line + "\n")
        val_predection.close()

# cifar10_train = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
# cifar10_test = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform_test)
# train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)
#
#
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.resnet18 = torchvision.models.resnet18(pretrained=True)
#         self.fc = nn.Linear(in_features=1000, out_features=10, bias=True)
#
#     def forward(self, x):
#         output = self.resnet18(x)
#         output = self.fc(output)
#         return output
#
#
# # print(resnet18)
#
# def eval(data_loader, model):
#     # evaluate on test data or eval data
#     correct = 0
#     total = 0
#     loss = 0
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     # since we're not training, we don't need to calculate the gradients for our outputs
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             images = images.cuda()
#             labels = labels.cuda()
#             # calculate outputs by running images through the network
#             outputs = model(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             loss += criterion(outputs, labels).sum().item()
#     print('Accuracy on test set: %.3f' % (correct / total))
#     print('Loss on test set: %.3f' % (loss / len(test_loader)))
#     return correct / total, loss / len(test_loader)
#
#
# net = NeuralNetwork()
# net.train()
# net.cuda()
#
# writer = SummaryWriter('runs/example')
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# for epoch in range(20):  # loop over the dataset multiple times
#     running_loss = 0.0
#     acc = 0.0
#     for i, data in enumerate(train_loader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         inputs = inputs.cuda()
#         labels = labels.cuda()
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # feed the data to model
#         outputs = net(inputs)
#         # calculate the loss
#         loss = criterion(outputs, labels)
#         # back propogation
#         loss.backward()
#         # update the model parameters
#         optimizer.step()
#
#         _, predicted = torch.max(outputs.data, 1)
#         acc += (predicted == labels).sum().item() / batch_size
#
#         # print statistics
#         running_loss += loss.item()
#         if (i + 1) % 100 == 0:  # print every 100 mini-batches
#             writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + i + 1)
#             writer.add_scalar('training acc', acc / 100, epoch * len(train_loader) + i + 1)
#             print('[epoch: %d, step: %5d] loss: %.3f acc:%.3f' % (epoch + 1, i + 1, running_loss / 100, acc / 100))
#             running_loss = 0.0
#             acc = 0
#     val_loss, val_acc = eval(test_loader, net)
#     writer.add_scalar('val loss', val_acc, epoch + 1)
#     writer.add_scalar('val acc', val_loss, epoch + 1)
# print('Finished Training')
#
# # class CustomDataset(torch.utils.data.Dataset):
# #     def __init__(self, data_path):
# #         self.data_path = data_path
# #         self.labels = []
# #         self.image_names = []
# #         # To be implemented
# #     def __len__(self):
# #         return len(self.labels)
# #     def __getitem__(self, idx):
# #         image = Image.open(self.image_names[idx])
# #         label = self.labels[idx]
# #         return image,label
