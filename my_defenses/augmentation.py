from art.defences.preprocessor import PreprocessorPyTorch
import numpy as np
import torch
import torch.nn.functional as F

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2



def cutmix(inputs, targets, cutmix_prob = 1.0, beta = 1.0):
    r = np.random.rand(1)
    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(inputs.size()[0]).cuda()
        target_a = targets
        target_b = targets[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

        mixed_target = lam * F.one_hot(targets, num_classes=43) + (1 - lam) * F.one_hot(targets[rand_index], num_classes=43)

    return inputs, mixed_target


def cutmix_fixed(inputs, targets, cutmix_prob = 1.0, beta = 1.0):
    r = np.random.rand(1)
    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(inputs.size()[0]).cuda()
        target_a = targets
        target_b = targets[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    return inputs, F.one_hot(target_a, num_classes=43)




def mixup(inputs, targets, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_input = lam * inputs + (1 - lam) * inputs[index, :]
    mixed_target = lam * F.one_hot(targets, num_classes=43) + (1 - lam) * F.one_hot(targets[index], num_classes=43)

    return mixed_input, mixed_target


class Augmentation(PreprocessorPyTorch):

    def __init__(self, method):
        self.method = method

    def forward(self, x, y):
        if self.method == 'mixup' :
            x, y = mixup(x,y)
        elif self.method == 'cutmix' :
            x, y = cutmix(x,y)
        elif self.method == 'cutmix_fixed' :
            x, y = cutmix_fixed(x,y)
        else :
            raise NotImplementedError

        return x, y