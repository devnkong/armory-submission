"""Separate file showing only robust(er) training and data augmentations.

This is not runnable code, but a template to implement these defenses into your own code!

Several helper files from forest/ are imported below which have to be bundled when copying this snippet.
"""
import torch
import torch.nn as nn
from PIL import ImageOps, Image
import numpy as np
import pickle
import pdb

nclasses = 43  # GTSRB has 43 classes
from forest.victims.batched_attacks import construct_attack
from forest.data.mixing_data_augmentations import Cutmix


def preprocessing_fn(batch):
    img_size = 48
    img_out = []
    quantization = 255.0
    for im in batch:
        img_eq = ImageOps.equalize(Image.fromarray(im))
        width, height = img_eq.size
        min_side = min(img_eq.size)
        center = width // 2, height // 2

        left = center[0] - min_side // 2
        top = center[1] - min_side // 2
        right = center[0] + min_side // 2
        bottom = center[1] + min_side // 2

        img_eq = img_eq.crop((left, top, right, bottom))
        img_eq = np.array(img_eq.resize([img_size, img_size])) / quantization

        img_out.append(img_eq)

    return np.array(img_out, dtype=np.float32)


class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, input):
        return input.permute(0, 3, 1, 2)


def Net():
    conv1 = nn.Conv2d(3, 1, kernel_size=1)
    conv2 = nn.Conv2d(1, 29, kernel_size=5)
    maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    conv3 = nn.Conv2d(29, 59, kernel_size=3)
    maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    conv4 = nn.Conv2d(59, 74, kernel_size=3)
    maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    fc1 = nn.Linear(1184, 300)
    fc2 = nn.Linear(300, nclasses)
    conv1_bn = nn.BatchNorm2d(1)
    conv2_bn = nn.BatchNorm2d(29)
    conv3_bn = nn.BatchNorm2d(59)
    conv4_bn = nn.BatchNorm2d(74)
    dense1_bn = nn.BatchNorm1d(300)
    ReLU = nn.ReLU()

    return nn.Sequential(
        Permute(),
        conv1,
        conv1_bn,
        ReLU,
        conv2,
        conv2_bn,
        ReLU,
        maxpool2,
        conv3,
        conv3_bn,
        ReLU,
        maxpool3,
        conv4,
        conv4_bn,
        ReLU,
        maxpool4,
        nn.Flatten(),
        fc1,
        ReLU,
        dense1_bn,
        fc2,
        # nn.LogSoftmax(),
    )

def _split_data(inputs, labels, p=0.75):
    """Split data for meta update steps and other defenses."""
    batch_size = inputs.shape[0]
    p_actual = int(p * batch_size)

    inputs, temp_targets, = inputs[0:p_actual], inputs[p_actual:]
    labels, temp_true_labels = labels[0:p_actual], labels[p_actual:]
    temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size - p_actual)
    return temp_targets, inputs, temp_true_labels, labels, temp_fake_label


def robust_train(model, x_train_final, y_train_final) :

    # hyperparameters:
    epochs = 40
    defense = dict(type='adversarial-wb', target_selection='sep-p96', steps=5, strength=16)
    mixing_method = dict(type='CutMix', correction=True, strength=1.0)
    num_classes = 43

    setup = dict(device=torch.device('cuda'), dtype=torch.float)


    # Define optimizer, dataloader and loss_fn
    # ...
    dataloader = torch.utils.data.DataLoader(range(len(x_train_final)), batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()


    # Prepare data_mean and data_std
    data_mean, data_std = [0,0,0], [1,1,1]
    dm = torch.tensor(data_mean)[None, :, None, None].to(**setup)
    ds = torch.tensor(data_std)[None, :, None, None].to(**setup)

    # Prepare defense:
    attacker = construct_attack(defense, model, loss_fn, dm, ds, tau=0.1, init='randn', optim='signAdam',
                                num_classes=num_classes, setup=setup)
    mixer = Cutmix(alpha=mixing_method['strength'])


    # Training loop:
    for epoch in range(epochs):
        for batch, idx in enumerate(dataloader):
            # Prep Mini-Batch
            # ...

            # Transfer to GPU
            # ...

            # Add basic data augmentation
            # ...

            inputs, labels = torch.tensor(x_train_final[idx]).cuda(), torch.tensor(y_train_final[idx]).cuda()
            inputs = inputs.permute(0,3,1,2)

            # ###  Mixing defense ###
            if mixing_method['type'] != '':
                inputs, extra_labels, mixing_lmb = mixer(inputs, labels, epoch=epoch)

            # ### AT defense: ###
            # Split Data
            [temp_targets, inputs, temp_true_labels, labels, temp_fake_label] = _split_data(inputs, labels, p=0.75)
            # Apply poison attack
            model.eval()
            delta, additional_info = attacker.attack(inputs, labels, temp_targets, temp_true_labels, temp_fake_label,
                                                     steps=defense['steps'])
            # temp targets are modified for trigger attacks:
            if 'patch' in defense['type']:
                temp_targets = temp_targets + additional_info
            inputs = inputs + delta


            # Switch into training mode
            model.train()

            # Change loss function to include corrective terms if mixing with correction
            if (mixing_method['type'] != '' and mixing_method['correction']):
                def criterion(outputs, labels):
                    return mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
            else:
                def criterion(outputs, labels):
                    loss = loss_fn(outputs, labels)
                    predictions = torch.argmax(outputs.data, dim=1)
                    correct_preds = (predictions == labels).sum().item()
                    return loss, correct_preds


            # Recombine poisoned inputs and targets into a single batch
            inputs = torch.cat((inputs, temp_targets))
            labels = torch.cat((labels, temp_true_labels))

            # Normal training from here on: ....
            optimizer.zero_grad()

            outputs = model(inputs.permute(0,2,3,1))
            loss, preds = criterion(outputs, labels)
            loss.backward()

            # Optimizer step
            optimizer.step()

    return model


model = Net().cuda()
with open('/cmlscratch/kong/projects/armory-submission/poisoned.pkl', 'rb') as f :
    data = pickle.load(f)
x_train_final, y_train_final = data[0], data[1]
robust_train(model, x_train_final, y_train_final)