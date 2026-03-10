import torch
import random
from torch.utils.data import Dataset


class ImgDataset_train(Dataset):
    def __init__(self, A_ori_img, A_gt, B_ori_img, B_gt, len_num=80000):
        self.A_ori_img = torch.cat((A_ori_img, A_ori_img[1:], A_ori_img[2:], A_ori_img[3:]), dim=0)
        self.A_gt = torch.cat((A_gt, A_gt[1:], A_gt[2:], A_gt[3:]), dim=0)

        self.B_ori_img = B_ori_img
        self.B_gt = B_gt

        self.len_num = len_num
        self.H, self.W, self.C = A_ori_img.shape[1], A_ori_img.shape[2], A_ori_img.shape[3]

    def __len__(self):
        return self.len_num

    def __getitem__(self, index):
        num_A = 19674
        num_B = 19660

        idx_A = index % (num_A*4-6)


        if index % 4 == 0:
            ori_img_A = self.A_ori_img[idx_A]
            gt_A = self.A_gt[idx_A]

            idx_B = random.randint(0, num_B - 1)
            while torch.argmax(gt_A) != torch.argmax(self.B_gt[idx_B]):
                idx_B = random.randint(0, num_B - 1)

            ori_img_B = self.B_ori_img[idx_B]
            gt_B = self.B_gt[idx_B]

            wavelengths = torch.linspace(430, 860, steps=self.C)
            transmittance = torch.ones(self.C)
            transmittance -= 0.25 * torch.exp(-0.02 * (wavelengths - 430))
            transmittance -= 0.05 * torch.exp(-0.5 * ((wavelengths - 480) ** 2) / (2 * 15 ** 2))
            transmittance -= 0.1 * torch.exp(-0.5 * ((wavelengths - 760) ** 2) / (2 * 5 ** 2))
            transmittance = torch.clamp(transmittance, 0., 1.)

            x_MOT = transmittance.view(1, 1, self.C).repeat(self.H, self.W, 1)
            noise = torch.normal(mean=0.0, std=0.01, size=(self.H, self.W, self.C))
            x_MOT_A = torch.clamp(x_MOT + noise, 0.0, 1.0)
            noise = torch.normal(mean=0.0, std=0.01, size=(self.H, self.W, self.C))
            x_MOT_B = torch.clamp(x_MOT + noise, 0.0, 1.0)

            return [ori_img_A, gt_A, ori_img_B, gt_B, x_MOT_A, x_MOT_B]

        else:
            ori_img_A = self.A_ori_img[idx_A]
            gt_A = self.A_gt[idx_A]

            idx_B = random.randint(0, num_B - 1)
            while torch.argmax(gt_A) == torch.argmax(self.B_gt[idx_B]):
                idx_B = random.randint(0, num_B - 1)

            ori_img_B = self.B_ori_img[idx_B]
            gt_B = self.B_gt[idx_B]

            wavelengths = torch.linspace(430, 860, steps=self.C)
            transmittance = torch.ones(self.C)
            transmittance -= 0.25 * torch.exp(-0.02 * (wavelengths - 430))
            transmittance -= 0.05 * torch.exp(-0.5 * ((wavelengths - 480) ** 2) / (2 * 15 ** 2))
            transmittance -= 0.1 * torch.exp(-0.5 * ((wavelengths - 760) ** 2) / (2 * 5 ** 2))
            transmittance = torch.clamp(transmittance, 0., 1.)

            x_MOT = transmittance.view(1, 1, self.C).repeat(self.H, self.W, 1)
            noise = torch.normal(mean=0.0, std=0.01, size=(self.H, self.W, self.C))
            x_MOT_A = torch.clamp(x_MOT + noise, 0.0, 1.0)
            noise = torch.normal(mean=0.0, std=0.01, size=(self.H, self.W, self.C))
            x_MOT_B = torch.clamp(x_MOT + noise, 0.0, 1.0)

            return [ori_img_A, gt_A, ori_img_B, gt_B, x_MOT_A, x_MOT_B]


class ImgDataset_test(Dataset):
    def __init__(self, ori_img, gt):
        self.ori_img, self.gt = ori_img, gt

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        try:
            ori_img = self.ori_img[index]
            gt = self.gt[index]

        except:
            print(index)

        return ori_img, gt

