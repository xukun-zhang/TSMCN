import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.layers.mesh_conv import MeshConv, MeshConv_CBAM, MeshTrans_all_atten
import torch.nn.functional as F
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance', num_groups=1):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == 'none':
        norm_layer = NoNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, '__name__') and norm_layer.__name__ == 'NoNorm':
        norm_args = [{'fake': True} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'GroupNorm':
        norm_args = [{'num_channels': f} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'BatchNorm':
        norm_args = [{'num_features': f} for f in nfeats_list]
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_layer.func.__name__)
    return norm_args

class NoNorm(nn.Module): #todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()
    def forward(self, x):
        return x
    def __call__(self, x):
        return self.forward(x)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(input_nc, ncf, ninput_edges, nclasses, opt, gpu_ids, arch, init_type, init_gain):
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm, num_groups=opt.num_groups)

    if arch == 'mconvnet':
        net = MeshConvNet(norm_layer, input_nc, ncf, nclasses, ninput_edges, opt.pool_res, opt.fc_n,
                          opt.resblocks)
    elif arch == 'meshunet':
        down_convs = [input_nc] + ncf
        up_convs = ncf[::-1] + [nclasses]
        pool_res = [ninput_edges] + opt.pool_res
        net = MeshEncoderDecoder(pool_res, down_convs, up_convs, blocks=opt.resblocks,
                                 transfer_data=True)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_loss(opt):
    if opt.dataset_mode == 'classification':
        loss = torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'segmentation':
        weight = torch.tensor([1.0, 10.0])
        loss = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
    return loss

##############################################################################
# Classes For Classification / Segmentation Networks
##############################################################################

class MeshConvNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3):
        super(MeshConvNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))


        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, bias=False))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        x = F.relu(x)
        return x

"""
用于边、点特征进行加权融合的注意力机制
"""
class Attn(nn.Module):
    def __init__(self, in_dim):
        super(Attn, self).__init__()

        self.chanel_in = in_dim
        self.conv_a = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.conv_b = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.conv_a_local = nn.Conv1d(in_channels=in_dim // 2, out_channels=in_dim // 2, kernel_size=1)
        self.conv_b_local = nn.Conv1d(in_channels=in_dim // 2, out_channels=in_dim // 2, kernel_size=1)
        self.conv_a_tri = MeshConv(in_dim // 2, in_dim // 2)
        self.conv_b_tri = MeshConv(in_dim // 2, in_dim // 2)
        self.conv_a_fuse = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=1),
            nn.InstanceNorm1d(in_dim),
            nn.Sigmoid()
        )
        self.conv_b_fuse = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=1),
            nn.InstanceNorm1d(in_dim),
            nn.Sigmoid()
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x_0, x_1, mesh):
        batch_0, C_0, width_0 = x_0.size()
        batch_1, C_1, width_1 = x_1.size()
        x_a = self.conv_a(x_0)
        x_b = self.conv_b(x_1)
        x_all = torch.cat([x_a, x_b], dim=1)
        # print("x_0.shape, x_1.shape:", x_0.shape, x_1.shape, x_all.shape)
        """
        x_all要经过两个卷积，都为为1*1的和1*5的，然后联合起来再经过1*1的
        不仅要用1*1的卷积，还要用1*5的考虑3角形面的卷积 
        """
        a_local = self.conv_a_local(x_all)
        a_tri = self.conv_a_tri(x_all, mesh).squeeze(-1)
        x_a_two = torch.cat((a_local, a_tri), dim=1)
        b_local = self.conv_b_local(x_all)
        b_tri = self.conv_b_tri(x_all, mesh).squeeze(-1)
        x_b_two = torch.cat((b_local, b_tri), dim=1)
        # print("x_a_two.shape, x_b_two.shape:", x_a_two.shape, x_b_two.shape)
        x_a_wight = self.conv_a_fuse(x_a_two).unsqueeze(1)
        x_b_wight = self.conv_b_fuse(x_b_two).unsqueeze(1)
        weights = self.softmax(torch.cat((x_a_wight, x_b_wight), dim=1))
        a_weight, b_weight = weights[:, 0:1, :, :].squeeze(1), weights[:, 1:2, :, :].squeeze(1)
        # print("weights.shape:", weights.shape, a_weight.shape, b_weight.shape)
        # print("x_a_wight.shape, x_b_wight.shape:", x_a_wight.shape, x_b_wight.shape)
        # print("a_local.shape, a_tri.shape, b_local.shape, b_tri.shape:", a_local.shape, a_tri.shape, b_local.shape, b_tri.shape)
        aggregated_feature = x_0.mul(a_weight) + x_1.mul(b_weight)
        # print("aggregated_feature.shape:", aggregated_feature.shape)


        return aggregated_feature

class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """
    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        # print("down_convs:", down_convs)
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks)
        # self.encoder_point = MeshEncoder(pools, [3, 32, 64, 128, 256, 256], blocks=blocks)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data)

        self.atten_encoder = Attn(128)
    def forward(self, x, meshes):
        # 这里到卷积里再岔开比较好
        # print("x.shape:", x.shape)
        # print("meshes:", meshes[0])
        fe, before_pool = self.encoder((x, meshes))     # 必须进里面修改，不然池化的特征可能不同！
        # fp, before_pool = self.encoder_point((x_p, meshes))
        # print("在输入解码器时，特征 fe.shape, len(before_pool):", fe.shape, len(before_pool))
        fe_e = fe[:, :int(fe.shape[1] / 2), :]
        fe_p = fe[:, int(fe.shape[1] / 2):, :]
        fe = self.atten_encoder(fe_e, fe_p, meshes)
        # print("输入解码器前的特征 fe.shape:", fe.shape)
        fe = self.decoder((fe, meshes), before_pool, x)
        return fe

    def __call__(self, x, meshes):
        return self.forward(x, meshes)



class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0):
        super(DownConv, self).__init__()
        self.bn = []
        self.bn_e = []
        self.bn_p = []
        self.pool = None
        """
        下面4个卷积（或者主要是2个卷积）可以换为CBAM+MeshConv
        """
        # self.conv1_e = MeshConv(in_channels, out_channels)
        # self.conv1_p = MeshConv(in_channels, out_channels)
        self.conv1_e_1 = MeshConv(in_channels, out_channels)
        self.conv1_e_2 = MeshConv_CBAM(in_channels, out_channels, 3000)
        self.conv1_e_3 = MeshConv_CBAM(in_channels, out_channels, 2250)
        self.conv1_e_4 = MeshConv_CBAM(in_channels, out_channels, 1750)
        self.conv1_p_1 = MeshConv(in_channels, out_channels)
        self.conv1_p_2 = MeshConv_CBAM(in_channels, out_channels, 3000)
        self.conv1_p_3 = MeshConv_CBAM(in_channels, out_channels,  2250)
        self.conv1_p_4 = MeshConv_CBAM(in_channels, out_channels, 1750)



        self.conv1_1_e = MeshConv(5, out_channels)
        self.conv1_1_p = MeshConv(3, out_channels)
        self.conv2 = []
        self.conv2_e = []
        self.conv2_p = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
            self.conv2_e.append(MeshConv(out_channels, out_channels))
            self.conv2_e = nn.ModuleList(self.conv2_e)
            self.conv2_p.append(MeshConv(out_channels, out_channels))
            self.conv2_p = nn.ModuleList(self.conv2_p)
        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
            self.bn_e.append(nn.InstanceNorm2d(out_channels))
            self.bn_e = nn.ModuleList(self.bn_e)
            self.bn_p.append(nn.InstanceNorm2d(out_channels))
            self.bn_p = nn.ModuleList(self.bn_p)
        self.atten = Attn(out_channels)
        # self.atten_64 = Attn(64)
        # self.atten_32 = Attn(32)
        # self.atten_32 = Attn()
        if pool:
            self.pool = MeshPool(pool)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        fe, meshes = x
        """根据输入特征的通道数量要主要修改的位置---conv1"""
        # print("传入卷积层的特征 fe.shape:", fe.shape)
        if fe.shape[1] == 8:
            fe_e = fe[:, :5, :]
            fe_p = fe[:, 5:, :]
            # print("输入前第一层， fe_e.shape, fe_p.shape:", fe_e.shape, fe_p.shape)
            x1_e = self.conv1_1_e(fe_e, meshes)
            x1_p = self.conv1_1_p(fe_p, meshes)
            # print("第一层输出， x1_e.shape, x1_p.shape:", x1_e.shape, x1_p.shape)

        else:
            fe_e = fe[:, :int(fe.shape[1]/2), :]
            fe_p = fe[:, int(fe.shape[1]/2):, :]
            # print("此时卷积前边、点的特征 fe_e.shape, fe_p.shape:", fe_e.shape, fe_p.shape)
            # x1_e = self.conv1_e(fe_e, meshes)
            # x1_p = self.conv1_p(fe_p, meshes)
            if fe_e.shape[2] == 22000:
                x1_e = self.conv1_e_1(fe_e, meshes)
                x1_p = self.conv1_p_1(fe_p, meshes)
            if fe_e.shape[2] == 3000:
                x1_e = self.conv1_e_2(fe_e, meshes)
                x1_p = self.conv1_p_2(fe_p, meshes)
            if fe_e.shape[2] == 2250:
                x1_e = self.conv1_e_3(fe_e, meshes)
                x1_p = self.conv1_p_3(fe_p, meshes)
            if fe_e.shape[2] == 1750:
                x1_e = self.conv1_e_4(fe_e, meshes)
                x1_p = self.conv1_p_4(fe_p, meshes)
            # print("卷积后的边、点特征 x1_e.shape, x1_p.shape:", x1_e.shape, x1_p.shape)
        if self.bn_e:
            x1_e = self.bn_e[0](x1_e)
        x1_e = F.relu(x1_e)
        x2_e = x1_e
        if self.bn_p:
            x1_p = self.bn_p[0](x1_p)
        x1_p = F.relu(x1_p)
        x2_p = x1_p
        # print("拆分后又进行正则化之后的边、点特征 x2_e.shape, x2_p.shape:", x2_e.shape, x2_p.shape)
        """同样，根据输入特征的通道数量要主要修改的位置---conv2"""
        for idx, conv in enumerate(self.conv2_e):
            x2_e = conv(x1_e, meshes)
            if self.bn_e:
                x2_e = self.bn_e[idx + 1](x2_e)
            x2_e = x2_e + x1_e
            x2_e = F.relu(x2_e)
            x1_e = x2_e
        x2_e = x2_e.squeeze(3)

        for idx, conv in enumerate(self.conv2_p):
            x2_p = conv(x1_p, meshes)
            if self.bn_p:
                x2_p = self.bn_p[idx + 1](x2_p)
            x2_p = x2_p + x1_p
            x2_p = F.relu(x2_p)
            x1_p = x2_p
        x2_p = x2_p.squeeze(3)

        before_pool = None


        x_combine = self.atten(x2_e, x2_p, meshes)

        x2_ep = torch.cat([x2_e, x2_p], dim=1)

        if self.pool:
            before_pool = x_combine 
            x2_ep = self.pool(x_combine, x2_ep, meshes) 
        return x2_ep, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True):
        super(UpConv, self).__init__()
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = MeshConv(in_channels, out_channels)
        if transfer_data:
            # self.conv1 = MeshConv(2 * out_channels, out_channels)
            self.conv1 = MeshConv(out_channels, out_channels)
        else:
            self.conv1 = MeshConv(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)

        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)

    def __call__(self, x, from_down=None, input_f=None):
        return self.forward(x, from_down, input_f)

    def forward(self, x, from_down=None, input_f=None):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            # print("x1, from_down:", x1.shape, from_down.shape)
            # x1 = torch.cat((x1, from_down), 1)
            x1 = x1 + from_down
        x1 = self.conv1(x1, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2

        """
        原本的输出层
        """
        x2 = x2.squeeze(3)
        return x2


class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None):
        super(MeshEncoder, self).__init__()
        self.fcs = None
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool))
        self.global_pool = None
        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            if global_pool is not None:
                if global_pool == 'max':
                    self.global_pool = nn.MaxPool1d(pools[-1])
                elif global_pool == 'avg':
                    self.global_pool = nn.AvgPool1d(pools[-1])
                else:
                    assert False, 'global_pool %s is not defined' % global_pool
            else:
                last_length *= pools[-1]
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        """
        这里拆开边和点?
        """
        fe, meshes = x
        # print("传入编码器的特征 fe.shape:", fe.shape)
        encoder_outs = []
        for conv in self.convs:
            """下面输出的fe应该是Cat合并的，这样的话进行conv()时就根据fe的通道数判断，从而拆分进行卷积：除了8的，其余的都是一半一半拆开。直到最后一个Block时也需要进行加权融合？"""
            fe, before_pool = conv((fe, meshes))     # 即每一次的卷积块都会生成一个fe，这个卷积块应该是一个阶段，可以既输出点的也可以既输出边的； 但是最后一个阶段或者解码器时要进行融合编码器最末端的点-边特征；
            encoder_outs.append(before_pool)
        if self.fcs is not None:
            if self.global_pool is not None:
                fe = self.global_pool(fe)     # 池化层要根据边的特征来进行选择，或者根据边和点共有的特征来进行选择；
            fe = fe.contiguous().view(fe.size()[0], -1)
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.relu(fe)
        return fe, encoder_outs

    def __call__(self, x):
        return self.forward(x)


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True):
        super(MeshDecoder, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                        batch_norm=batch_norm, transfer_data=transfer_data))
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None, input_f=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes), None, input_f)

        # print("fe.shape, fe_2.shape:", fe.shape, fe_2.shape)
        return fe

    def __call__(self, x, encoder_outs=None, input_f=None):
        return self.forward(x, encoder_outs, input_f)

def reset_params(model): # todo replace with my init
    for i, m in enumerate(model.modules()):
        weight_init(m)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
