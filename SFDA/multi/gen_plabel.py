import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from utils.dataset import Camelyon17, Prostate
from utils.loss import DiceLoss, JointLoss
from nets.models import DenseNet, UNet
from utils.weight_perturbation import WPOptim

import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import math

if __name__ == '__main__':
    available_datasets = ['camelyon17','prostate']
    parser = argparse.ArgumentParser()
    # parser.add_argument('--log', action='store_true', help='whether to log')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch', type = int, default= 8, help ='batch size')
    # parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    # parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local client between communication')
    # parser.add_argument('--alpha', type=float, default=0.05, help='The hyper parameter of perturbation')
    parser.add_argument('--data', type = str, choices=available_datasets, default='camelyon17', help='Different dataset')
    parser.add_argument('--save_path', type = str, default='../checkpoint/', help='path to save the checkpoint')
    parser.add_argument('--test_path', type=str, default='../checkpoint/', help='path to saved model, for testing')
    # parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--gpu', type = int, default=0, help = 'gpu device number')
    parser.add_argument('--seed', type = int, default=0, help = 'random seed')
    # parser.add_argument('--test', action='store_true', help='test model')
    parser.add_argument('--src_num', type = int, default=5, help = 'number of source models')
    # parser.add_argument('--imbalance', action='store_true', help='do not truncate train data to same length')

    args = parser.parse_args()
    args.log = True

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    pseudo_label_dic = {}
    threshold = 0.75

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    target_model = UNet(input_shape=[3, 384, 384])
    target_model.to(device)
    target_model.eval()

    path = '../models/prostate/Site_ISBI_1.5_latest'
    checkpoint = torch.load(path, map_location=device)
    target_model.load_state_dict(checkpoint['server_model'], strict=False)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    sites = ['UCL','BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5']
    trainset = Prostate(site='UCL', split='train', transform=transform)
    valset = Prostate(site='UCL', split='val', transform=transform)
    testset = Prostate(site='UCL', split='test', transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=True)

    with torch.no_grad():
        for step, (data, _ , names) in enumerate(train_loader):

            data = data.to(device)
            output, feature = target_model(data)

            pseudo_label = output.clone()
            pseudo_label[pseudo_label > threshold] = 1.0; pseudo_label[pseudo_label <= threshold] = 0.0

            pseudo_label = pseudo_label.detach().cpu().numpy()

            for i in range(output.shape[0]):
                pseudo_label_dic[names[i]] = pseudo_label[i]

    np.savez("plabel", pseudo_label_dic)
