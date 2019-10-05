import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, stride=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride
                              )

    def forward(self, x):
        '''
        :param x: (b,1,28,28)
        :return: (b,256,20,20)
        '''
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
            for _ in range(num_capsules)])  # (b,32,6,6)

    def forward(self, x):
        '''
        :param x:(b,256,20,20)
        :return:(b,1152,8)
        '''
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=4)  # (b,32,6,6,8)
        u = u.view(x.size(0), 32 * 6 * 6, -1)  # (b,32*6*6,8)
        return self.squash(u, dim=-1)

    def squash(self, input_tensor, dim):
        squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16, iteration=3):
        super(DigitCaps, self).__init__()
        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.iterations = iteration
        self.W = nn.Parameter(0.01 * torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
        # ATTENTION! You have to include the 0.01 term, or the network won't be learning.
        # (1,1152,10,16,8)

    def forward(self, u):
        '''
        :param x: (b,32*6*6,8)
        :return: (b,10,16)
        '''
        u_sliced = u.unsqueeze(-1).unsqueeze(2)  # (b,1152,1,8,1)
        u_hat = torch.matmul(self.W, u_sliced).squeeze(
            4)  # (1,1152,10,16,8) * (b,1152,1,8,1) ->(b,1152,10,16,1)->(b,1152,10,16)
        v = self.routing(u_hat)  # (b,10,16)
        return v

    def squash(self, input_tensor, dim):
        '''
        :param input_tensor: (b,1,10,16)
        :param dim: int
        :return: (b,1,10,16)
        '''
        squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

    def routing(self, u_hat):
        '''
        :param u_hat: (b,1152,10,16)
        :return:(b,10,16)
        '''
        b = torch.zeros_like(u_hat)
        # u_hat_routing = u_hat.detach()  # should we back-propagate through each iteration?
        u_hat_routing = u_hat.detach()  # should we back-propagate through each iteration?
        for i in range(self.iterations):
            c = F.softmax(b, dim=2)  # (b,1152,10,16)
            if i == (self.iterations - 1):
                s = (c * u_hat).sum(dim=1, keepdim=True)
            else:
                # s = (c * u_hat_routing).sum(1, keepdim=True)
                s = (c * u_hat).sum(1, keepdim=True)
            v = self.squash(s, dim=-1)  # (b,1,10,16)
            if i < self.iterations - 1:
                # b = (b + (u_hat_routing * v).sum(3, keepdim=True))  # (b,1152,10,16)+(b,1152,10,1)->(b,1152,10,16)
                b = (b + (u_hat * v).sum(3, keepdim=True))  # (b,1152,10,16)+(b,1152,10,1)->(b,1152,10,16)
        return v.squeeze(dim=1)


class CapsModel(nn.Module):
    def __init__(self,
                 image_dim_size=28,
                 cl_input_channels=1,
                 cl_num_filters=256,
                 cl_filter_size=9,
                 cl_stride=1,
                 pc_input_channels=256,
                 pc_num_caps_channels=32,
                 pc_caps_dim=8,
                 pc_filter_size=9,
                 pc_stride=2,
                 dc_num_caps=10,
                 dc_caps_dim=16,
                 iterations=3):
        super(CapsModel, self).__init__()
        self.conv_layer = ConvLayer(in_channels=cl_input_channels, out_channels=cl_num_filters,
                                    kernel_size=cl_filter_size, stride=cl_stride)
        self.pc_layer = PrimaryCaps(num_capsules=pc_num_caps_channels, in_channels=pc_input_channels,
                                    out_channels=pc_caps_dim, kernel_size=pc_filter_size,
                                    stride=pc_stride)
        cl_output_dim = int((image_dim_size - cl_filter_size + 1) / cl_stride)  # 20
        pc_output_dim = int((cl_output_dim - pc_filter_size + 1) / pc_stride)  # 6
        self.pc_num_caps = pc_output_dim * pc_output_dim * pc_num_caps_channels  # 6*6*32 =  1152
        self.dc_layer = DigitCaps(num_capsules=dc_num_caps, num_routes=self.pc_num_caps, in_channels=pc_caps_dim,
                                  out_channels=dc_caps_dim, iteration=iterations)

    def forward(self, x):
        '''
        :param x:(b,1,28,28)
        :return: (b,10,16)
        '''
        c = self.conv_layer(x)  # (b,256,20,20)
        u = self.pc_layer(c)  # (b,1152,8)
        v = self.dc_layer(u)  # (b,10,16)
        return v

    def loss(self, y_gold, pred, lambda_param=0.5, m_plus=0.9, m_minus=0.1):
        '''
        :param y_gold:(b,10)
        :param pred:(b,10,16)
        :param x_true: (b,1,28,28)
        :param x_reconstructed: (b,784)
        :return:(1)
        '''
        v_norm = pred.norm(dim=2, keepdim=False)  # (b,10)
        hinge_loss = (y_gold * F.relu(m_plus - v_norm) ** 2 + lambda_param * (1 - y_gold) * F.relu(
            v_norm - m_minus) ** 2).sum(
            1).mean()
        return hinge_loss

    # %%


class Decoder(nn.Module):
    def __init__(self, dc_caps_dim, dc_num_caps, image_dim_size):
        super(Decoder, self).__init__()
        self.dc_num_caps = dc_num_caps
        self.network = nn.Sequential(
            nn.Linear(dc_caps_dim * dc_num_caps, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, image_dim_size ** 2),
            nn.Sigmoid()
        )
        self.reconst_loss = nn.MSELoss()

    def forward(self, v, y_ohe):
        '''
        :param v:(b,10,16）
        :param y_ohe:(b,10)
        :return:(b,784)
        '''
        bs = v.size(0)
        # get the predicted vector
        pred = y_ohe.unsqueeze(-1) * v
        pred = pred.view(bs, -1)  # (bs,160)
        rec = self.network(pred)
        return rec  # (b,784)

    def loss(self, x_true, x_reconstructed):
        '''
        :param x_true: (b,1,28,28)
        :param x_reconstructed: (b,28*28)
        :return:(1)
        '''
        bs = x_true.size(0)
        x_true = x_true.view(bs, -1)  # (b,784)
        rec_loss = self.reconst_loss(x_reconstructed, x_true)
        return rec_loss
