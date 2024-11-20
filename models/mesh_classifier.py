import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch


def dice_coefficient(predictions, targets, smooth=1e-6):
    """
    计算Dice系数。

    参数:
    - predictions: 模型预测结果，维度为[B, N]，B是批次大小，N是点数。
    - targets: 真实标签，维度与predictions相同。
    - smooth: 平滑项，避免除以0的情况，默认为1e-6。
    返回:
    - dice: 计算得到的Dice系数。
    """
    # 确保predictions和targets为二值化数据（0或1）
    predictions = predictions.float()
    targets = targets.float()

    # 计算交集
    intersection = (predictions * targets).sum()

    # 计算Dice系数
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    # print(dice)
    # 返回批次内的平均Dice系数
    return dice.mean()


# # 示例使用
# # 假设有一个批次大小为2，每个样本有5个点的预测和标签
# predictions = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.float32)
# targets = torch.tensor([[1, 0, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=torch.float32)
#
# dice_score = dice_coefficient(predictions, targets)
# print(f"Dice Score: {dice_score}")


def dice_loss(output_2, target):
    """
    计算Dice Loss，忽略标签为-1的点。
    参数:
    - input: 模型输出，大小为[B, 2, N]。
    - target: 真实标签，大小为[B, N]，包含0, 1和-1，-1需要被忽略。

    返回:
    - loss: Dice Loss的值。
    """

    # print("output_2.shape, target.shape:", output_2.shape, target.shape)
    # 确保输入被softmax处理，转换成概率分布
    input_soft = F.softmax(output_2, dim=1)
    # 选择前景类别的预测概率，即类别1
    input_foreground = input_soft[:, 1, :]  # 大小为[B, N]
    # 将标签为-1的点的mask计算出来
    valid_mask = target != -1  # 大小为[B, N]
    # 仅选择有效的前景标签点
    target_foreground = (target == 1) & valid_mask  # 大小为[B, N]
    # 将input和target都转换为float类型以计算Dice系数
    input_foreground = input_foreground.type(torch.float32)
    target_foreground = target_foreground.type(torch.float32)
    # 计算分子：预测和真实前景的交集，注意要乘2，因为Dice系数的定义
    intersection = 2 * (input_foreground * target_foreground).sum(-1)
    # 计算分母：预测和真实前景的和
    union = input_foreground.sum(-1) + target_foreground.sum(-1)
    # 计算Dice Loss，添加一个小数避免除以0
    dice_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
    # 忽略无效点（-1标签的点）的影响，计算平均损失
    dice_loss = dice_loss[valid_mask.any(dim=1)].mean()
    return dice_loss

# 假设你的模型和数据都在CUDA设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建权重张量，并确保它在正确的设备上
weight = torch.tensor([1.0, 10.0, 10.0], device=device)
ce_loss = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-1)

class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        labels = torch.from_numpy(data['label']).long()
        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])


    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        # print("out.shape, self.labels.shape:", out.shape, out, self.labels.shape, self.labels)
        # self.loss = self.criterion(out, self.labels) + self.criterion(out_2, self.labels)
        self.loss = ce_loss(out, self.labels)
        # print("self.loss:", self.loss)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()


##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            # compute number of correct
            pred_class = out.data.max(1)[1]

            label_class = self.labels
            self.export_segmentation(pred_class.cpu())
            # print("pred_class.shape, label_class.shape:", pred_class.shape, label_class.shape)
            # correct = self.get_accuracy(pred_class, label_class)
            correct = 0
            # print("correct:", correct, len(label_class), len(label_class[0]))
            # pred.eq(labels).sum()

            # print("correct, len(label_class):", correct, len(label_class), len(label_class[0]))
            for file_i in range(label_class.shape[0]):
                mask_re = label_class[file_i] != -1
                # print(pred_class[mask_re].shape, label_class[mask_re].shape)
                list_pred = list(pred_class[file_i][mask_re].cpu() )
                list_label = list(label_class[file_i][mask_re].cpu() )

                # print("len(list_pred), len(list_label):", len(list_pred), len(list_label), self.mesh[0].filename)
                # print("m", self.mesh[file_i].filename)
                file_name_pred = self.mesh[file_i].filename.split(".obj")[0] + "-pred.eseg"
                file_name_label = self.mesh[file_i].filename.split(".obj")[0] + "-label.eseg"
                path_file_i_p = os.path.join("/home/zxk/code/P2ILF-Mesh/Ours-ablation-3-L/test_results", file_name_pred)
                path_file_i_l = os.path.join("/home/zxk/code/P2ILF-Mesh/Ours-ablation-3-L/test_results", file_name_label)
                np.savetxt(path_file_i_p, list_pred, fmt='%d')
                np.savetxt(path_file_i_l, list_label, fmt='%d')

                # mask_re = label_class[file_i] == 1
                # print(pred_class[mask_re].shape, label_class[mask_re].shape)
                # list_pred_1 = list(pred_class[file_i][mask_re].cpu())
                # list_label_1 = list(label_class[file_i][mask_re].cpu())
                # print(pred_class[file_i][mask_re].shape, label_class[file_i][mask_re].shape)
                # pred_class[file_i][mask_re][pred_class[file_i][mask_re]==2] = 0
                # pred_class[file_i][mask_re][label_class[file_i][mask_re]==2] = 0
                # label_class[file_i][mask_re][label_class[file_i][mask_re]==2] = 0
                
                # 假设 pred_class 和 label_class 已经根据 file_i 和 mask_re 索引
                # 使用 masked_fill_() 来原地修改满足条件的元素值
                pred_class_masked = pred_class[file_i][mask_re]
                label_class_masked = label_class[file_i][mask_re]

                # 将等于2的值设置为0
                pred_class_masked.masked_fill_(pred_class_masked == 2, 0)
                label_class_masked.masked_fill_(label_class_masked == 2, 0)

                # 重新检查最大值和最小值
                # print("After correction:")
                # print("pred_class max and min:", pred_class_masked.max(), pred_class_masked.min())
                # print("label_class max and min:", label_class_masked.max(), label_class_masked.min())



                # print("pred_class[file_i][mask_re].max(), pred_class[file_i][mask_re].min():", pred_class[file_i][mask_re].max(), pred_class[file_i][mask_re].min())
                # print("label_class[file_i][mask_re].max(), label_class[file_i][mask_re].min():", label_class[file_i][mask_re].max(), label_class[file_i][mask_re].min())
                correct_1 = dice_coefficient(pred_class_masked, label_class_masked)

                acc_1 = correct_1
                correct = correct + correct_1

                print("这个数据保存了两类，但预测镰状韧带的准确率为:", acc_1, self.mesh[file_i].filename)
            # print("两个输出的平均指标为：", result_1_acc/label_class.shape[0], result_2_acc/label_class.shape[0])

        return correct, len(label_class), acc_1

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            # correct = seg_accuracy(pred, self.soft_label, self.mesh)
            correct = pred.eq(labels).sum()
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])
