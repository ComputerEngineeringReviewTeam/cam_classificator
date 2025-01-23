from ai.nn.config import *
from ai.dataset.cam_dataset import CamDataset
from ai.nn.camnet import CamNet
from ai.nn.custom_loss import CustomLoss


def train(model, device, loss_fn, dataset, optimizer, epochs):
    for epoch in range(epochs):
        for i, ((image, scale), label) in enumerate(dataset):
            image, scale, label = image.to(device), scale.to(device), label.to(device)
            result = model((image, scale))
            loss = loss_fn(result, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    model = CamNet(model_name=MODEL_NAME, pretrained=True, num_aux_inputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CustomLoss()
    if TRAIN:
        train(model, DEVICE, loss_fn, CamDataset, optimizer, EPOCHS)




