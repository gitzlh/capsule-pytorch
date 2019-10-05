import argparse
from mycapnet import *
import time
import statistics
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def train_epoch(capnet, decoder, optimizer, lr_scheduler, train_loader, opt):
    capnet.train()
    decoder.train()
    tqdm_bar = tqdm(train_loader, mininterval=2, desc='  - (Training)   ', leave=False)
    loss_train = []
    acc_train = []
    for batch in tqdm_bar:
        x, y = list(map(lambda x: x.to(opt.device), batch))
        y_ohe = torch.eye(10).cuda().index_select(dim=0, index=y)  # (10,10)->(b,10)
        v = capnet(x)
        rec = decoder(v, y_ohe)
        hinge_loss = capnet.loss(y_ohe, v)
        rec_loss = decoder.loss(x, rec)
        loss_batch = hinge_loss + rec_loss
        capnet.zero_grad()
        decoder.zero_grad()
        loss_batch.backward()
        optimizer.step()
        lr_scheduler.step()
        _, preds = v.norm(p=2, dim=2).max(dim=1)  # (b)
        acc_batch = (preds == y).float().mean()
        loss_train.append(loss_batch.item())
        acc_train.append(acc_batch.item())
        tqdm_bar.set_postfix(accuracy=statistics.mean(acc_train), loss=statistics.mean(loss_train))
    loss = statistics.mean(loss_train)
    acc = statistics.mean(acc_train)
    return loss, acc


def eval_epoch(capnet, decoder, optimizer, lr_scheduler, eval_loader, opt):
    capnet.eval()
    decoder.eval()
    tqdm_bar = tqdm(eval_loader, mininterval=2, desc='  - (Validation)   ', leave=False)
    loss_eval = []
    acc_eval = []
    with torch.no_grad():
        for batch in tqdm_bar:
            x, y = list(map(lambda x: x.to(opt.device), batch))
            y_ohe = torch.eye(10).cuda().index_select(dim=0, index=y)  # (10,10)->(b,10)
            v = capnet(x)
            rec = decoder(v, y_ohe)
            hinge_loss = capnet.loss(y_ohe, v)
            rec_loss = decoder.loss(x, rec)
            loss_batch = hinge_loss + rec_loss
            _, preds = v.norm(p=2, dim=2).max(dim=1)  # (b)
            acc_batch = (preds == y).float().mean()
            loss_eval.append(loss_batch.item())
            acc_eval.append(acc_batch.item())
            tqdm_bar.set_postfix(accuracy=statistics.mean(acc_eval), loss=statistics.mean(loss_eval))
    loss = statistics.mean(loss_eval)
    acc = statistics.mean(acc_eval)
    return loss, acc


def train(capnet, decoder, optimizer, lr_scheduler, train_loader, test_loader, opt):
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        loss, acc = train_epoch(capnet, decoder, optimizer, lr_scheduler, train_loader, opt)
        print(
            '  - (Training) loss: {loss: .4f}, accuracy :{acc: .4f},elapse: {elapse:3.3f} min'.format(
                loss=loss,
                acc=acc,
                elapse=(
                               time.time() - start) / 60))

        start = time.time()
        loss, acc = eval_epoch(capnet, decoder, optimizer, lr_scheduler, test_loader, opt)
        print(
            '  - (Validation)    loss: {loss: .4f}, accuracy:{acc:.4f}, elapse: {elapse:3.3f} min'.format(
                loss=loss,
                acc=acc,
                elapse=(
                               time.time() - start) / 60))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='./data')
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-dropout', type=float, default=0.2)
    parser.add_argument('-save_model', default='capsule.chkpt')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    # parser.add_argument('-restore_model', default='capsule.chkpt')
    parser.add_argument('-restore_model', default=None)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-cl_input_channels', default=1, action='store_true')
    parser.add_argument('-cl_num_filters', default=256, action='store_true')
    parser.add_argument('-cl_filter_size', default=9, action='store_true')
    parser.add_argument('-cl_stride', default=1, action='store_true')
    parser.add_argument('-pc_input_channels', default=256, action='store_true')
    parser.add_argument('-pc_num_caps_channels', default=32, action='store_true')
    parser.add_argument('-pc_caps_dim', default=8, action='store_true')
    parser.add_argument('-pc_filter_size', default=9, action='store_true')
    parser.add_argument('-pc_stride', default=2, action='store_true')
    parser.add_argument('-image_dim_size', default=28, action='store_true')
    parser.add_argument('-dc_num_caps', default=10, action='store_true')
    parser.add_argument('-dc_caps_dim', default=16, action='store_true')
    parser.add_argument('-iterations', default=3, action='store_true')
    parser.add_argument('-reconst_loss_scale', default=0.392, action='store_true')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    # ========= Loading Dataset =========#
    train_loader, test_loader = prepare_dataloaders(path=opt.data)
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    opt.reconst_loss_scale = 0.0005 * opt.image_dim_size ** 2
    capnet = CapsModel(
        opt.image_dim_size,
        opt.cl_input_channels,
        opt.cl_num_filters,
        opt.cl_filter_size,
        opt.cl_stride,
        opt.pc_input_channels,
        opt.pc_num_caps_channels,
        opt.pc_caps_dim,
        opt.pc_filter_size,
        opt.pc_stride,
        opt.dc_num_caps,
        opt.dc_caps_dim,
        opt.iterations).cuda()

    decoder = Decoder(opt.dc_caps_dim, opt.dc_num_caps, opt.image_dim_size, ).cuda()
    optimizer = torch.optim.Adam(list(capnet.parameters()) + list(decoder.parameters()), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96 ** (1 / 2000.))
    if opt.restore_model:
        checkpoint = torch.load(opt.restore_model)
        capnet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        opt.epoch_hist = checkpoint['epoch']
        print('[Info] Old Trained model state loaded.')
    num_parameters = sum(p.numel() for p in capnet.parameters() if p.requires_grad) + sum(
        p.numel() for p in decoder.parameters() if p.requires_grad)
    print("[INFO] Total parameter number: ", num_parameters)
    train(capnet=capnet, decoder=decoder, optimizer=optimizer, lr_scheduler=lr_scheduler, train_loader=train_loader,
          test_loader=test_loader, opt=opt)


def prepare_dataloaders(path='./data', download=True, batch_size=100, shift_pixels=2):
    """
    Construct dataloaders for training and test data. Data augmentation is also done here.
    :param path: file path of the dataset
    :param download: whether to download the original data
    :param batch_size: batch size
    :param shift_pixels: maximum number of pixels to shift in each direction
    :return: train_loader, test_loader
    """
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=download,
                       transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
                                                     transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=download,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


if __name__ == '__main__':
    main()
