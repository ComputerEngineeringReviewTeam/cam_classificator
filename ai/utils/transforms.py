import torchvision.transforms

import ai.config as conf


# TODO: make it more modular ?
# TODO: decouple from config
class CamTransforms:
    class Train:
        std = torchvision.transforms.Compose([
            torchvision.transforms.Resize(conf.TARGET_SIZE),
            torchvision.transforms.RandomRotation(30, fill=255),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
        ])
        grayscale = torchvision.transforms.Compose([
            torchvision.transforms.Resize(conf.TARGET_SIZE),
            torchvision.transforms.RandomRotation(30, fill=255),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
            torchvision.transforms.Grayscale(num_output_channels=3)
        ])
    class Test:
        std = torchvision.transforms.Compose([
            torchvision.transforms.Resize(conf.TARGET_SIZE),
        ])
        grayscale = torchvision.transforms.Compose([
            torchvision.transforms.Resize(conf.TARGET_SIZE),
            torchvision.transforms.Grayscale(num_output_channels=3)
        ])