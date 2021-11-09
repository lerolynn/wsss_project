import torchvision
from torchvision import models
import torch

class DeepLabV3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(DeepLabV3Wrapper, self).__init__()
        self.model = model

    def forward(self, input):
        output = self.model(input)['out']
        return output

def initialize_model(num_classes, keep_feature_extract=False, use_pretrained=True):
    model_deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model_deeplabv3.aux_classifier = None

    input_size = 224
    model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

    return model_deeplabv3, input_size

