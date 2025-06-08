import torch
import timm

from ai.config import Modes, FEATURES, DROPOUT


class CamNet(torch.nn.Module):
    def __init__(self,
                 model_name,
                 mode: Modes = Modes.BOTH,
                 pretrained=True,
                 num_aux_inputs=1):
        """
        Initialize CamNet model

        Args:
            model_name (str): Name of the base model to use
            mode (str): Operating mode - 'classifier', 'regressor', or 'both'
            pretrained (bool): Whether to use pretrained weights
            num_aux_inputs (int): Number of auxiliary inputs
        """
        super(CamNet, self).__init__()

        if mode not in Modes:
            raise ValueError("Mode must be one of: Modes.CLASSIFIER, Modes. REGRESSOR, Modes.BOTH")

        import segmentation_models_pytorch as smp

        self.mode = mode
        self.encoder = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).encoder
        feature_dim = 512

        if mode in [Modes.CLASSIFIER, Modes.BOTH]:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, FEATURES),
                torch.nn.ReLU(),
                torch.nn.Dropout(DROPOUT),
                torch.nn.Linear(FEATURES, 1),
            )

        if mode in [Modes.REGRESSOR, Modes.BOTH]:
            self.regressor = torch.nn.Sequential(
                torch.nn.Linear(feature_dim + num_aux_inputs, FEATURES),
                torch.nn.ReLU(),
                torch.nn.Dropout(DROPOUT),
                torch.nn.Linear(FEATURES, 1),
            )

    def forward(self, inputs):
        image, scale = inputs
        features = self.encoder(image)[-1]  # [B, 512, H', W']
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)  # [B, 512, 1, 1]
        flattened = pooled.view(pooled.size(0), -1)  # [B, 512]

        if self.mode == Modes.CLASSIFIER:
            return self.classifier(flattened), None
        elif self.mode == Modes.REGRESSOR:
            return None, self.regressor(flattened)
        else:
            return self.classifier(flattened), self.regressor(flattened)
