import torch
import os
import pdb
import sys
from torchvision import models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import cv2
import imgaug.augmenters as iaa
from tqdm import tqdm

feature = []
def get_last_conv(m):
    """
    Get the last conv layer in an Module.
    """
    convs = filter(lambda k: isinstance(k, torch.nn.Conv2d), m.modules())
    return list(convs)[-1]

def get_feature(module,input,output):
            feature.append(output.detach())


def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('--test_file', help='test file', type=str,default='temp_test3.txt')
    parser.add_argument('--checkpoint', help='resume checkpoint .pth', type=str,default="6_0.9400516.pth")
    args = parser.parse_args()
    #define model
    model = models.resnet18(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, 2)
    #load checkpoint
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()
    f = open(args.test_file,'r').readlines()
    datas = []
    labels = []
    with torch.no_grad():
        for index,lines in tqdm(enumerate(f)):
            img = cv2.imread(lines.strip().split(' ')[0])
            label = int(lines.strip().split(' ')[1])
            img =cv2.resize(img,(224,224))
            img = img.astype(np.float32)
            #img = iaa.CenterPadToFixedSize(height=256,width=256)(image=img)
            #img = iaa.CropToFixedSize(width=224, height=224,position="center")(image=img)
            img = (img / 255-0.5)/0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
            img = img.view(1, *img.size())
            # m = get_last_conv(model)
            model.avgpool.register_forward_hook(get_feature)
            output = model(img)
            feature[-1] = feature[-1].reshape(1,-1)
            datas.append(feature[-1].numpy())
            labels.append(label)
            if index == 15000:
                break
            # print(index)
            # print(feature[-1].shape)
            # print(len(datas))
            # if index ==2:
            #     break
            # pdb.set_trace()
            res_data=np.zeros((len(datas),512)) #初始化一个np.array数组用于存数据,512为输出维度
            res_label =np.zeros((len(datas),))
            for i in range(len(datas)):
                res_data[i] = datas[i]
                res_label[i] = labels[i]
        tsne_2D = TSNE(n_components=2, init='pca', random_state=0) #调用TSNE
        result_2D = tsne_2D.fit_transform(res_data)
        fig1 = plot_embedding_2D(result_2D, res_label,'t-SNE')
        fig1.savefig("out.png")
            # pred = np.argmax(output[0]).numpy()
