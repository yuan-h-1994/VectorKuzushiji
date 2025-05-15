import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, backbone, hidden_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.class_embed = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, samples):
        features = self.backbone(samples)
        outputs_class = self.class_embed(features)

        out = {"pred_logits": outputs_class}
        return out


class SetCriterion(nn.Module):
    def __init__(self, losses, weight_dict):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
    
    def loss_labels(self, outputs, targets, log=True):
        src_logits = outputs['pred_logits']
        loss_ce = F.cross_entropy(src_logits, targets)
        losses = {'loss_ce': loss_ce}
        return losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'labels': self.loss_labels
        }
        assert loss in loss_map
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))
        
        return losses


def build(args):
    device = torch.device(args.device)

    backbone = models.resnet50(pretrained=True)
    model = Classifier(backbone, 1000, 2)

    losses = ['labels']
    weight_dict = {'loss_ce': 1}
    criterion = SetCriterion(losses, weight_dict)
    criterion.to(device)

    return model, criterion
