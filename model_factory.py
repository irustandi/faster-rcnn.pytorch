from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from model.faster_rcnn.faster_rcnn import FasterRCNN
from model.faster_rcnn.vgg import VGGBackBone
from model.faster_rcnn.resnet import ResNetBackBone


def get_model(args, imdb):
    # initialize the network here.
    if args.net == 'vgg16':
        backbone = VGGBackBone(pretrained=True)
    elif args.net == 'res101':
        backbone = ResNetBackBone(101, pretrained=True)
    elif args.net == 'res50':
        backbone = ResNetBackBone(50, pretrained=True)
    elif args.net == 'res152':
        backbone = ResNetBackBone(152, pretrained=True)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN = FasterRCNN(classes=imdb.classes, class_agnostic=args.class_agnostic, backbone=backbone)

    fasterRCNN.create_architecture()

    return fasterRCNN
