import torch
from modelling import *

_fusion_type = {"mcb": MCB_baseline, "mfb": MFB_baseline, "mutan": MUTAN_baseline}

class VQA_Classifier(torch.nn.Module):

    def __init__(self, cfg, model, out_size):
        super(VQA_Classifier, self).__init__()
        img_feature_size = cfg.PRETRAINED.OUT_DIM
        text_feature_size = cfg.PRETRAINED.TEXT_DIM
        fs = _fusion_type.get(cfg.FUSION.TYPE, MCB_baseline)
        self.fusion_layer = fs(img_feature_size, text_feature_size, cfg.FUSION.OUT_DIM, cfg.FUSION.FACTOR)
        self.linear = torch.nn.Linear(cfg.FUSION.OUT_DIM, out_size)
        self.m = torch.nn.LogSoftmax(dim=1)

        self.model = model
        #self.act = torch.nn.ReLU()

    def forward(self, im, text):

        image_features = self.model.encode_image(im).float()

        text_features = self.model.encode_text(text).float()

        y = self.fusion_layer(text_features, image_features)
        X = self.m(self.linear(y))
        return X