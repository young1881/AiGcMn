import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import matplotlib.gridspec as gridspec
import os


def save_model(model, save_dir):
    torch.save(model.state_dict(), save_dir)


def save_model_cpu(model, save_dir):
    state = model.state_dict()
    x = state.copy()
    for key in x:
        x[key] = x[key].clone().cpu()
    torch.save(x, save_dir)


def save_img(images, count):
    # create the img to figure the best model
    images = images.to('cpu')
    images = images.detach().numpy()
    images = images[[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]]
    images = 255 * (0.5 * images + 0.5)
    images = images.astype(np.uint8)
    grid_length = int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(4, 4))
    width = images.shape[2]
    gs = gridspec.GridSpec(grid_length, grid_length, wspace=0, hspace=0)

    # subplot
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(width, width), cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()

    plt.savefig(r'./CGAN_3rd/images/%d.png' % count, bbox_inches='tight')


def Data_Loader(batch_size):

    trans_img = transforms.Compose([transforms.ToTensor()])
    train = MNIST('./data', train=True, transform=trans_img, download=True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=10)
    test = MNIST('./data', train=False, transform=trans_img, download=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=10)

    return train, test, train_loader, test_loader


def build_dis_gen(dis, gen, dis_epoch, gen_epoch, Batch_size):

    # load MNIST
    train, test, train_loader, test_loader = Data_Loader(Batch_size)  # data

    # define the optimizer
    dis_optimizer = optim.Adam(dis.parameters(), lr=0.0003)
    gen_optimizer = optim.Adam(gen.parameters(), lr=0.0003)

    # BCE loss
    loss_function = nn.BCELoss()

    # train the Discriminator
    # train Generator during the Discriminator-training process
    for i in range(dis_epoch):

        for (img, label) in train_loader:
            labels = np.zeros((Batch_size, 10))
            labels[np.arange(Batch_size), label.numpy()] = 1
            img = Variable(img).cuda()

            real_label = Variable(torch.from_numpy(labels).float()).cuda()
            fake_label = Variable(torch.zeros(Batch_size, 10)).cuda()

            # loss & score of real
            real_out = dis(img)
            dis_loss_real = loss_function(real_out, real_label)
            real_score = real_out

            # loss & score of fake
            noise = Variable(torch.randn(Batch_size, z_dimension)).cuda()
            fake_img = gen(noise)
            fake_out = dis(fake_img)
            dis_loss_fake = loss_function(fake_out, fake_label)
            fake_score = fake_out

            # Back Propagation & Optimize
            dis_loss = dis_loss_real + dis_loss_fake
            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()

            # train the Generator
            for j in range(gen_epoch):
                noise = torch.randn(Batch_size, 100)
                noise = np.concatenate((noise.numpy(), labels), axis=1)
                noise = Variable(torch.from_numpy(noise).float()).cuda()
                fake_img = gen(noise)
                output = dis(fake_img)
                gen_loss = loss_function(output, real_label)

                # Back Propagation & Optimize
                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()
                final = real_label

        # save the model
        if (i % 2 == 0) and (i != 0):
            print(i)
            # cuda
            save_model(gen, r'./CGAN_3rd/Generator_cuda_epoch_%d.pkl' % i)
            save_model(dis, r'./CGAN_3rd/Discriminator_cuda_epoch_%d.pkl' % i)
            # cpu
            save_model_cpu(gen, r'./CGAN_3rd/Generator_cpu_epoch_%d.pkl' % i)
            save_model_cpu(dis, r'./CGAN_3rd/Discriminator_cpu_epoch_%d.pkl' % i)

        print('Epoch [{}/{}], Dis_loss: {:.6f}, Gen_loss: {:.6f} '
              'Dis_real: {:.6f}, Dis_fake: {:.6f}'.format(
            i, dis_epoch, dis_loss.data.item(), gen_loss.data.item(),
            real_score.data.mean(), fake_score.data.mean()))

        final = final.to('cpu')

        _, sample = torch.max(final, 1)
        sample = sample.numpy()
        print(sample[[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]])

        # save the images
        if not os.path.exists('./CGAN_3rd/images/'):
            os.makedirs('./CGAN_3rd/images/')
        save_img(fake_img, i)



#判别器，10分类，输入图像张量 x，输出类别
class Discriminator(nn.Module):

    #  卷积层和池化层
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2))
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#生成器，输入的噪声向量x,输出的值为大小为 batch_size x 1 x 28 x 28 的张量，表示生成的图像
class Generator(nn.Module):

    def __init__(self, input_size, num_feature):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # 1*56*56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

        self.gen = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True),
            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.gen(x)
        return x


if __name__ == "__main__":

    # training epoches
    Dis_epoch = 100
    # train Generator gen_epoch times in one dis_epoch
    Gen_epoch = 1

    # Batch
    batch_size = 100
    # 100 + 10 numbers
    z_dimension = 110

    # define the Discriminator and Generator
    Dis = Discriminator()
    Gen = Generator(z_dimension, 3136)  # 1*56*56
    Dis = Dis.cuda()
    Gen = Gen.cuda()

    build_dis_gen(dis=Dis, gen=Gen, dis_epoch=Dis_epoch, gen_epoch=Gen_epoch, Batch_size=batch_size)


