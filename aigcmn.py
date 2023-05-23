import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from gendis import Generator

#生成器，输入的噪声向量x,输出的值为大小为 batch_size x 1 x 28 x 28 的张量，表示生成的图像

class AiGcMn:
    def __init__(self):
        self.generator = Generator(110,3136)
        self.generator.load_state_dict(torch.load(r'./model/Generator_cuda_32.pkl'))
        self.batchsize=0


    def input(self):
        numb1 = input('输入数字：').split()
        num_lst = list(map(int, numb1))
        print(num_lst)
        self.batchsize = 1
        self.numno=len(num_lst)
        num_lst=torch.Tensor(num_lst)
        num_lst=num_lst.reshape(self.numno,1)

        return num_lst

    def showall(self,images):#images=self.generator

        images = images.detach().numpy()
        images = 255 * (0.5 * images + 0.5)
        images = images.astype(np.uint8)
        plt.figure(figsize=(self.numno, self.numno))
        width = images.shape[2]
        gs = gridspec.GridSpec(1, self.numno, wspace=0, hspace=0)
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(img.reshape(width, width), cmap=plt.cm.gray)
            plt.axis('off')
            plt.tight_layout()
        plt.tight_layout()
        return width



    def generate(self, input)->torch.tensor:
        num_img=1
        piclist = []
        for num in input:
            num=int(num)
            numtensor = np.zeros((num_img, 10))
            numtensor[:, num] = 1
            noisetensor = torch.randn((num_img, 100))
            finaltensor = np.concatenate((noisetensor.numpy(), numtensor), 1)
            finaltensor = torch.from_numpy(finaltensor).float()
            one_img = self.generator(finaltensor)  # 将向量放入生成网络G生成一张图片
            piclist.append(one_img)
        total_img=torch.cat(piclist,0)
        self.showall(total_img)
        if not os.path.exists('./result/'):
            os.makedirs('./result/')
        output_path = './result/total.txt'
        torch.set_printoptions(threshold=np.inf)
        with open(output_path, 'w', encoding='utf-8') as file:
            print(total_img, file=file)
        plt.savefig('./result/total.png', bbox_inches='tight')
        return total_img


aigcmn = AiGcMn()
numlist=aigcmn.input()
aigcmn.generate(numlist)





