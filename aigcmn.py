"""
API to trained model
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from gendis import Generator, Discriminator
import torch.nn.functional as F


class AiGcMn:
    def __init__(self):
        self.generator = Generator(110, 3136)
        self.generator.load_state_dict(torch.load(r'./model/Generator_cuda_32.pkl'))
        self.discriminator = Discriminator()
        self.discriminator.load_state_dict(torch.load(r'./model/Discriminator_cuda_32.pkl'))
        self.batchsize = 0

    def input(self):
        numb1 = input('Input your number：').split()
        num_lst = list(map(int, numb1))
        print(num_lst)
        self.batchsize = 1
        self.numno = len(num_lst)
        num_lst = torch.Tensor(num_lst)
        num_lst = num_lst.reshape(self.numno, 1)

        return num_lst

    def show_all(self, images):
        images = images.detach().numpy()
        images = 255 * images
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

    def ad(self, img):
        """
        Adversarial Sample for ODD
        :param img: the ouput tensor of generator
        :return: adversarial sample image tensor
        """
        epsilon = 0.04
        noise = torch.zeros([1, 1, 28, 28])
        noises = torch.zeros([28 * 28, 1, 28, 28])
        for i in range(28):
            for j in range(28):
                noises[28 * i + j, 0, i, j] = epsilon
        loss_0 = torch.max(self.discriminator(img))
        # print(loss_0)
        loss_now = loss_0
        ad = img + noise
        for i in range(5):
            if float(loss_now) < 0.5:
                break
            losses = torch.max(self.discriminator(ad + noises), dim=1).values
            noise += ((losses <= loss_0) * epsilon).reshape(1, 1, 28, 28)
            noise -= ((losses > loss_0) * epsilon).reshape(1, 1, 28, 28)
            ad = img + noise
            # make pixel 0 if less than 0
            ad *= (ad >= torch.zeros(1, 1, 28, 28))
            # make pixel 1 if greater than 1
            ad = ad - ad * (ad > torch.ones(1, 1, 28, 28)) + (ad > torch.ones(1, 1, 28, 28))

            loss_now = torch.max(self.discriminator(ad))

        return ad

    def generate(self, input) -> torch.tensor:
        num_img = 1
        piclist = []
        for num in input:
            num = int(num)
            numtensor = np.zeros((num_img, 10))
            numtensor[:, num] = 1
            noisetensor = torch.randn((num_img, 100))
            finaltensor = np.concatenate((noisetensor.numpy(), numtensor), 1)
            finaltensor = torch.from_numpy(finaltensor).float()
            one_img = self.generator(finaltensor)  # put the vector into the generator
            # Normalization to [0, 1]
            one_img = (one_img - one_img.min()) / (one_img.max() - one_img.min())
            ad = self.ad(one_img)
            # use adversarial sample as output
            piclist.append(ad)
        total_img = torch.cat(piclist, 0)
        self.show_all(total_img)
        if not os.path.exists('./result/'):
            os.makedirs('./result/')
        output_path = './result/total.txt'
        torch.set_printoptions(threshold=np.inf)
        with open(output_path, 'w', encoding='utf-8') as file:
            print(total_img, file=file)
        plt.savefig('./result/total.png', bbox_inches='tight')
        return total_img


aigcmn = AiGcMn()
numlist = aigcmn.input()
aigcmn.generate(numlist)
