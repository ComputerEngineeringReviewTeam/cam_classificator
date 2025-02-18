import torchmetrics
import torchvision.transforms as transforms

from ai.dataset.cam_label import JsonLabelLoader
from ai.config import *
from ai.paths import LABELS_PATH, IMG_DIR, MODEL_PATH
from ai.nn.camnet import CamNet
from ai.nn.custom_loss import CustomLoss
from ai.dataset.dataset_helpers import train_test_data, describe_dataset
from ai.tools.cam_snapshot import CamSnapshot


def prepare_tensors(image: torch.Tensor,
                    scale: torch.Tensor,
                    binary_target: torch.Tensor,
                    regression_target: torch.Tensor,
                    device: str):
    """
    Moves tensors to device and reshapes binary_target to have 2 dimensions

    ! Reshaping must be done after moving to device !

    Args:
        image: torch.Tensor
        scale: torch.Tensor
        binary_target: torch.Tensor
        regression_target: torch.Tensor
        device: str

    Returns:
        image: torch.Tensor
        scale: torch.Tensor
        binary_target: torch.Tensor
        regression_target: torch.Tensor
    """

    image, scale = image.to(device), scale.to(device)
    binary_target, regression_target = binary_target.to(device), regression_target.to(device)
    binary_target = binary_target.unsqueeze(1)
    return image, scale, binary_target, regression_target


def train(model: torch.nn.Module,
          device: str,
          loss_fn: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          acc: torchmetrics.Metric):
    print('Training...')
    for epoch in range(epochs):

        for i, ((image, scale), (binary_target, regression_target)) in enumerate(dataloader):
            image, scale, binary_target, regression_target = prepare_tensors(image, scale, binary_target, regression_target, device)

            binary_output, regression_output = model((image, scale))  # model.random_trans(image)
            loss = loss_fn(binary_output, regression_output, binary_target, regression_target)
            acc(binary_output, binary_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item()} Accuracy: {acc.compute()}")
        print(binary_output)
        acc.reset()
    print('Finished Training')


def test(model: torch.nn.Module,
         device: str,
         dataloader: torch.utils.data.DataLoader,
         metric: torchmetrics.Metric):
    with torch.no_grad():
        for i, ((image, scale), (binary_target, regression_target)) in enumerate(dataloader):
            image, scale, binary_target, regression_target = prepare_tensors(image, scale, binary_target, regression_target, device)

            binary_output, regression_output = model((image, scale))
            metric(binary_output, binary_target)


if __name__ == '__main__':
    tsfms = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
    ])

    label_loader = JsonLabelLoader()  # Loads data from JSON file
    (train_dataset, test_dataset), (train_loader, test_loader) = train_test_data(label_loader.load(LABELS_PATH),
                                                                                 img_dir=IMG_DIR,
                                                                                 train_fraction=TRAIN_FRACTION,
                                                                                 batch_size=BATCH_SIZE,
                                                                                 transform=tsfms,
                                                                                 random_state=RANDOM_SEED,
                                                                                 balanced=True
                                                                                 )

    model = (CamNet(model_name=MODEL_NAME,
                    pretrained=True,
                    num_aux_inputs=0)
             .to(device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CustomLoss()

    snapshot = CamSnapshot(model, optimizer, loss_fn, train_loader, DEVICE)

    train_acc = torchmetrics.classification.BinaryAccuracy(threshold=BINARY_ACCURACY_THRESHOLD).to(device=DEVICE)
    test_acc = torchmetrics.classification.BinaryAccuracy(threshold=BINARY_ACCURACY_THRESHOLD).to(device=DEVICE)

    print("Device:", DEVICE)

    if TRAIN:
        describe_dataset(train_dataset)
        model.train()
        train(model, DEVICE, loss_fn, train_loader, optimizer, EPOCHS, train_acc)
        if SAVE_MODEL:
            torch.save(model.state_dict(), MODEL_PATH)

    if TEST:
        describe_dataset(test_dataset)
        if LOAD_MODEL:
            model = CamNet(model_name=MODEL_NAME, pretrained=True, num_aux_inputs=1).to(device=DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        test(model, DEVICE, test_loader, test_acc)
        print(f"Test accuracy: {test_acc.compute()}")

    if SNAPSHOT:
        snapshot.save(SNAPSHOT_PATH)
