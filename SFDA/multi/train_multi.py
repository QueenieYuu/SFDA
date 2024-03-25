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
from utils.loss import DiceLoss, JointLoss, EntKLProp
from nets.models import DenseNet, UNet, Scalar
from utils.weight_perturbation import WPOptim

import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import math

from segmentation_mask_overlay import overlay_masks
from PIL import Image

bceloss = nn.BCELoss(reduction='none')

def RandomRotate(sample):
    img = sample
    seed = random.random()

    if seed > 0.5:
        rotater = transforms.RandomRotation(degrees=(0, 180))
        img = rotater(img)
    return img

def RandomFlip(sample):
    img = sample
    if random.random() < 0.5:
        Hflip = transforms.RandomHorizontalFlip(p=0.5)
        img = Hflip(img)
    if random.random() < 0.5:
        Vflip = transforms.RandomVerticalFlip(p=0.5)
        img = Vflip(img)
    return img

def entloss(p):
    # y1 = -1.0*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C))
    n, c, h, w = p.size()
    y1 = -torch.mul(p, torch.log2(p + 1e-30)) / np.log2(c)
    ent = torch.mean(y1)
    return ent


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=0)
    target_softmax = F.softmax(target_logits, dim=0)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def test(args, model, data_loader, loss_fun, device):
    model.to(device)
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    test_acc = 0.
    segmentation = model.__class__.__name__ == 'UNet'

    with torch.no_grad():
        for step, (data, target, name) in enumerate(data_loader):

            data = data.to(device)
            target = target.to(device)
            output, feature = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()


            for index in range(data.shape[0]):

                image = data[index,0,:,:].cpu().detach().numpy()
                image = Image.fromarray(image.astype(np.uint8)).convert('P')
                image = image.convert('RGBA')

                # mask = output.argmin(dim=1)
                # mask = torch.stack([mask == c for c in range(1)], dim=1).type(torch.int32)
                # # print(mask.shape)
                mask = torch.argmax(output, dim =1)
                mask = mask[index, :,:].cpu().detach().numpy()
                mask_ = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                mask_[:, :, 0] = mask * 255
                mask_[:, :, 1] = mask * 255
                mask_[:, :, 2] = mask * 255

                palette = [255, 255, 255]
                zero_pad = 256 * 3 - len(palette)

                for i in range(zero_pad):
                    palette.append(0)

                mask_ = Image.fromarray(mask_.astype(np.uint8)).convert('P')
                mask_.putpalette(palette)
                mask_ = mask_.convert('RGBA')

                gt = target[index,0,:,:].cpu().detach().numpy()
                gt_ = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
                gt_[:, :, 0] = gt * 255
                gt_[:, :, 1] = gt * 255
                gt_[:, :, 2] = gt * 255

                palette = [255, 255, 255]
                zero_pad = 256 * 3 - len(palette)

                for i in range(zero_pad):
                    palette.append(0)

                gt_ = Image.fromarray(gt_.astype(np.uint8)).convert('P')
                gt_.putpalette(palette)
                gt_ = gt_.convert('RGBA')

                # mask.putpalette(palette)
                # mask = mask.convert('RGBA')
                # masks.append(img)

                mask_ = Image.blend(image, mask_, 0.5)
                gt_ = Image.blend(image, gt_, 0.5)
                image.save('results/%s.png' % (str(name[index])))
                mask_.save('results/pd_%s.png' % (str(name[index])))
                gt_.save('results/gt_%s.png' % (str(name[index])))

            if segmentation:
                test_acc += DiceLoss().dice_coef(output, target).item()
            else:
                total += target.size(0)
                pred = output.data.max(1)[1]
                batch_correct = pred.eq(target.view(-1)).sum().item()
                correct += batch_correct
                if step % math.ceil(len(data_loader)*0.2) == 0:
                    print(' [Step-{}|{}]| Test Acc: {:.4f}'.format(step, len(data_loader), batch_correct/target.size(0)), end='\r')

    loss = loss_all / len(data_loader)
    acc = test_acc/ len(data_loader) if segmentation else correct/total
    # model.to('cpu')
    return loss, acc

def train(args, target_model, models, data_loader, optimizer, loss_fun, device):

    # total = 0
    # correct = 0
    # train_acc = 0.
    segmentation = target_model.__class__.__name__ == 'UNet'
    npfilename = 'plabel.npz'

    npdata = np.load(npfilename, allow_pickle=True)
    pseudo_label_dic = npdata['arr_0'].item()

    w = 2*torch.rand((len(models)))-1
    netG_list = [Scalar(w[i]).cuda() for i in range(len(models))]

    for step, (data, target, name) in enumerate(data_loader):

        # loss_all = 0
        loss_consis = 0
        loss_avg = 0
        entropy_loss = 0
        optimizer.zero_grad()

        outputs_all = torch.zeros(len(models), data.shape[0], 2,  data.shape[2],  data.shape[3])
        weights_all = torch.ones(data.shape[0],  len(models))
        outputs_all_w = torch.zeros(data.shape[0], 2, data.shape[2], data.shape[3])
        # init_ent = torch.zeros(1, len(models))

        data = data.to(device)
        target = target.to(device)
        output, feature = target_model(data)

        pseudo_label = [pseudo_label_dic.get(key) for key in name]
        pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()

        pseudo_label = torch.argmax(pseudo_label, dim=1).unsqueeze(axis=1)
        loss_seg = loss_fun(output, pseudo_label)

        #ADAMI
        # loss_1, loss_cons_prior,est_prop =  loss_fn(output, pseudo_label)
        # loss = loss_1  + loss_cons_prior + loss_seg
        #ADAMI

        noise = torch.clamp(torch.randn_like(data) * 0.1, -0.2, 0.2)
        data_noise = data + noise
        # data_aug = RandomFlip(RandomRotate(data))

        for client_idx, model in enumerate(models):

            output_noise, feature_noise = model(data_noise)
            con_loss = softmax_mse_loss(output_noise, output).mean()

            w_diff = torch.tensor(0.).cuda()
            for w, w_t in zip(target_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            com_loss = 1e-2 / 2. * w_diff

            # output_aug, _ = model(data_aug)
            # aug_loss = softmax_kl_loss(output_aug, output).mean()

            # DECISION
            # softmax_ = F.softmax(output_noise, dim=0)
            # ent_loss = torch.mean(entloss(softmax_))
            # init_ent[:,client_idx] = ent_loss
            # weights_test = netG_list[client_idx](feature_noise)
            # DECISION

            outputs_all[client_idx] = output_noise
            weights_all[:,client_idx] = con_loss

            loss_consis += con_loss
            loss_avg += com_loss

            # SHOT
            # softmax_out = F.softmax(output_noise, dim=0)
            # entropy_loss_1= torch.mean(entloss(softmax_out))
            #
            # msoftmax = softmax_out.mean(dim=1)
            # entropy_loss_2 = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5)) * 0.00001
            #
            # entropy_loss =  entropy_loss_1 - entropy_loss_2
            # SHOT

        # z = torch.sum(weights_all, dim=1)
        # z = z + 1e-16
        # weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
        #
        outputs_all = torch.transpose(outputs_all, 0, 1)

        for i in range(data.shape[0]):
            # print(torch.transpose(torch.transpose(outputs_all[i],0,3),0,1).shape)
            # print(torch.mean(torch.mean(torch.transpose(torch.transpose(weights_all[i],0,2),1,2), 1), 1).shape)
            outputs_all_w[i] = torch.matmul(torch.transpose(torch.transpose(outputs_all[i],0,3),0,1),weights_all[i])
        outputs_all_w = outputs_all_w.cuda()

        softmax_out = F.softmax(outputs_all_w, dim=0)
        entropy_loss_1= torch.mean(entloss(softmax_out))

        msoftmax = softmax_out.mean(dim=1)
        entropy_loss_2 = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5)) * 0.00001

        entropy_loss =  entropy_loss_1 - entropy_loss_2

        loss = loss_seg.mean() + torch.mean(loss_consis) + torch.mean(entropy_loss) + torch.mean(loss_avg)
        print('Seg Loss: {:.4f} | Consis Loss: {:.4f} | Ent Loss: {:.4f} | Avg Loss: {:.4f}'.format(loss_seg.mean(), torch.mean(loss_consis), torch.mean(entropy_loss), torch.mean(loss_avg)))

        #
        # loss = loss_seg.mean() + torch.mean(loss_consis) + torch.mean(entropy_loss)
        # print('Seg Loss: {:.4f} | Consis Loss: {:.4f} | Ent Loss: {:.4f}'.format(loss_seg.mean(), torch.mean(loss_consis), torch.mean(entropy_loss)))

        # loss = loss_seg.mean() + torch.mean(entropy_loss)
        # print('Seg Loss: {:.4f} | Ent Loss: {:.4f}'.format(loss_seg.mean(), torch.mean(entropy_loss)))


        loss.backward()
        optimizer.generate_delta(zero_grad=True)
        loss_fun(target_model(data)[0], pseudo_label).backward()
        optimizer.step(zero_grad=True)

    # model.to('cpu')
    return loss.item()

def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params

        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key]).to(device)
            for client_idx in range(len(client_weights)):
                print(temp)
                print(client_weights[client_idx])
                print(models[client_idx].state_dict()[key])
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key].to(device)
            server_model.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(client_weights)):
                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            if 'running_amp' in key:
                # aggregate at first round only to save communication cost
                server_model.amp_norm.fix_amp = True
                for model in models:
                    model.amp_norm.fix_amp = True
    return server_model, models

if __name__ == '__main__':
    available_datasets = ['camelyon17','prostate']
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch', type = int, default=8, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    # parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local client between communication')
    parser.add_argument('--alpha', type=float, default=0.05, help='The hyper parameter of perturbation')
    parser.add_argument('--data', type = str, choices=available_datasets, default='prostate', help='Different dataset')
    parser.add_argument('--save_path', type = str, default='../checkpoint/', help='path to save the checkpoint')
    parser.add_argument('--test_path', type=str, default='../checkpoint/', help='path to saved model, for testing')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--gpu', type = int, default=0, help = 'gpu device number')
    parser.add_argument('--seed', type = int, default=0, help = 'random seed')
    parser.add_argument('--test', action='store_true', help='test model')
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

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.save_path = '../checkpoints/{}/'.format(args.data)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.test_path = '../models/prostate/'
    args.test_path = os.path.join(args.test_path, 'Site_')

    target_model = UNet(input_shape=[3, 384, 384])
    target_model.to(device)
    target_model.train()

    path = '../models/prostate/Site_ISBI_1.5_latest'
    checkpoint = torch.load(path, map_location=device)
    target_model.load_state_dict(checkpoint['server_model'], strict=False)

    loss_fun = JointLoss()
    loss_fn = EntKLProp()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    sites = ['HK', 'BIDMC', 'UCL', 'I2CVB', 'ISBI_1.5']
    trainset = Prostate(site='ISBI', split='train', transform=transform)
    valset = Prostate(site='ISBI', split='val', transform=transform)
    testset = Prostate(site='ISBI_1.5', split='test', transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=True)

    source_models = [copy.deepcopy(target_model) for idx in range(args.src_num)]

    # federated client number
    # client_weights = [1./args.src_num for i in range(args.src_num)]
    # client_weights = torch.Tensor(client_weights).to(device)

    if args.test:
        # evaluate performance on testset using model already trained.
        print('Loading snapshots...')

        # for idx, model in enumerate(source_models):
        #
        #     test_path = args.test_path + sites[idx] + '_latest'
        #
        #     checkpoint = torch.load(test_path, map_location=device)
        #     model.load_state_dict(checkpoint['server_model'], strict=False)
        #
        #     test_loss, test_acc = test(args, model, test_loader, loss_fun, device)
        #     print('[Site-{}]  Test Loss: {:.4f}, Test Acc: {:.4f}'.format(sites[idx],test_loss,test_acc))

        # test_path = '../models/prostate/target_model_latest'
        test_path = '../checkpoints/prostate/best_isbi2'
        checkpoint = torch.load(test_path, map_location=device)
        target_model.load_state_dict(checkpoint['server_model'], strict=False)
        test_loss, test_acc = test(args, target_model, test_loader, loss_fun, device)
        print('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(test_loss,test_acc))

        exit(0)

    best_changed = False

    optimizer = WPOptim(params=target_model.parameters(), base_optimizer=optim.Adam, lr=args.lr, alpha=args.alpha, weight_decay=1e-4)

    best_epoch = 0
    best_acc = [0. for j in range(args.src_num)]
    start_iter = 0

    for client_idx, model in enumerate(source_models):

        test_path = args.test_path + sites[client_idx] + '_latest'

        checkpoint = torch.load(test_path, map_location=device)
        model.load_state_dict(checkpoint['server_model'], strict=False)

        model.to(device)
        model.train()

    # Start training
    for a_iter in range(start_iter, args.iters):

        print("============ Train epoch {} ============".format(a_iter))
        # if args.log:
        #     logfile.write("============ Train epoch {} ============\n".format(a_iter))

        loss = train(args, target_model, source_models, train_loader, optimizer, loss_fun, device)
        # if args.log:
        #     logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx], train_loss, train_acc))

        with torch.no_grad():

            # server_model, models = communication(args, target_model, source_models, client_weights)

            print('============== {} =============='.format('Validation'))
            # if args.log:
            #         logfile.write('============== {} ==============\n'.format('Global Validation'))
            # for client_idx, model in enumerate(models):
            val_loss, val_acc = test(args, target_model, val_loader, loss_fun, device)
            print('Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(val_loss, val_acc))
            # if args.log:
            #     logfile.write(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss, val_acc))
            #     logfile.flush()
            if np.mean(val_acc) > np.mean(best_acc):
                best_acc = val_acc
                best_epoch = a_iter
                best_changed=True
                print('Best Epoch:{} | Best  Acc: {:.4f}'.format(best_epoch, best_acc))
            if best_changed:
                print(' Saving the local and server checkpoint to {}...'.format(args.save_path))
                model_dicts = {'server_model': target_model.state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'a_iter': a_iter}

                # for o_idx in range(client_num):
                #     model_dicts['optim_{}'.format(o_idx)] = optimizers[o_idx].state_dict()

                model_dicts['optim'] = optimizer.state_dict()

                torch.save(model_dicts, args.save_path + 'target_model_best')
                best_changed = False
            else:
                print(' Saving the local and server checkpoint to {}...'.format(args.save_path))
                model_dicts = {'server_model': target_model.state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'a_iter': a_iter}
                model_dicts['optim'] = optimizer.state_dict()

                torch.save(model_dicts, args.save_path + 'target_model_latest')
