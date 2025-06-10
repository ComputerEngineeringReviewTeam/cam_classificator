import torch


class CamNetConv(torch.nn.Module):
    def __init__(self):
        super(CamNetConv, self).__init__()

        self.extractor = torch.nn.Sequential(
            # 224 x 224 x 3
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(),
            
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 110 x 110 x 64
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 54 x 54 x 128
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 25 x 25 x 32
            torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(),
            # 21 x 21 x 3
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(in_features=1323, out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=256, out_features=1),
        )

        # self.regressor = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=1323, out_features=256),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(in_features=256, out_features=1),
        #     # torch.nn.LeakyReLU(),
        #     # torch.nn.Linear(in_features=64, out_features=1),
        #     # torch.nn.Sigmoid()
        # )



    def forward(self, inputs):
        image, scale = inputs
        extracted_features = self.extractor(image)
        # # print(extracted_features)
        # flattened_features = extracted_features.flatten(start_dim=1)
        # print(flattened_features)
        # regression_output = self.regressor(flattened_features)
        # print(regression_output)
        # return torch.zeros(regression_output.size()), regression_output
        return torch.zeros(extracted_features.size()), extracted_features
