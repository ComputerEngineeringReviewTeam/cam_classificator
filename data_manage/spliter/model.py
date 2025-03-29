import torch
import timm


class FragmentClassifier(torch.nn.Module):
    """
    A simple classifier model that uses a pretrained ResNet18 model from timm
    with only 2 output classes
    """

    def __init__(self):
        super(FragmentClassifier, self).__init__()
        self.resnet = timm.create_model("resnet18", pretrained=True)
        self.fc = torch.nn.Linear(in_features=self.resnet.fc.out_features, out_features=2)

    def forward(self, x):
        y = self.resnet(x)
        return self.fc(y)
