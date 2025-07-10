import torch
import numpy as np


def data_shuffle(ori_img_slice):
    """
    Spatial-spectral exchange
    :param ori_img_slice: tensor, input HSI img slice [N, c, slice_size, slice_size]
    :return: gen_img_slice: tensor, output HSI img slice [N, c, slice_size, slice_size]
    """
    N, c, slice_size, __ = ori_img_slice.shape
    idx_list = []
    idx_list_ori = []
    for i in range(slice_size):
        for j in range(slice_size):
            idx_list.append([i, j])
            idx_list_ori.append([i, j])  # 如果idx_list_ori = idx_list 则打乱时idx_list_ori一起跟着改变

    gen_img_slice = torch.zeros_like(ori_img_slice)
    for i in range(N):
        np.random.seed(1234)
        np.random.shuffle(idx_list)
        for j in range(len(idx_list)):
            gen_img_slice[i, :, idx_list_ori[j][0], idx_list_ori[j][1]] = ori_img_slice[i, :, idx_list[j][0], idx_list[j][1]]
        gen_img_slice[i, :, idx_list_ori[(len(idx_list)-1)//2][0], idx_list_ori[(len(idx_list)-1)//2][1]] = \
            ori_img_slice[i, :, idx_list_ori[(len(idx_list)-1)//2][0], idx_list_ori[(len(idx_list)-1)//2][1]]

    return gen_img_slice


if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = x.unsqueeze(0).unsqueeze(0)
    y = data_shuffle(x)
    print(y)
