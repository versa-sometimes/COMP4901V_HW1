from .models import CNNClassifier
from .utils import ConfusionMatrix, load_data
import torch
import torchvision
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    model = CNNClassifier()
    model.eval()
    batch_size = 64
    model.load_state_dict(torch.load('./model_20230401_063051_10'))
    train_data = load_data('drive-download-20230329T090612Z-001/train_subset', 2, batch_size)
    valid_data = load_data('drive-download-20230329T090612Z-001/validation_subset', 2, batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")    # select GPU device
        model = model.to(device)   

    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for the vehicle classification task
    Hint: use the ConfusionMatrix for you to calculate accuracy
    """
    train_cm = ConfusionMatrix(6)
    valid_cm = ConfusionMatrix(6)

    for i, data in enumerate(train_data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        train_cm.add(outputs, labels)

    for i, data in enumerate(valid_data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        valid_cm.add(outputs, labels)

    return train_cm.average_accuracy(), valid_cm.average_accuracy()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
