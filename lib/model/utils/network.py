import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision.models as models

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im

def slice_vgg16(model):

    slices = []
    # we fix conv1_1, conv1_2, conv2_1, conv2_2
    slices.append(nn.Sequential(*list(model.features.children())[:10]))
    # we finetune conv3_1, conv3_2, conv3_3
    slices.append(nn.Sequential(*list(model.features.children())[10:17]))
    # we retrain conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3
    slices.append(nn.Sequential(*list(model.features.children())[17:-1]))

    # we copy fc6
    slices.append(model.classifier[0])

    # we copy fc7
    slices.append(model.classifier[3])

    return slices

def load_baseModel(model_name):
    if model_name == "vgg16":
        pretrained_model = models.vgg16(pretrained=True)
        return slice_vgg16(pretrained_model)
    elif model_name == "resnet50":
        return None

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_checkpoint(state, filename):
    torch.save(state, filename)