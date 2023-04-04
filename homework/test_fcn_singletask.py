import torch
import numpy as np

from .models import FCN_ST 
from .utils import load_dense_data, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your single-task model, and perform evaluation for the segmentation task
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task
     
    """
    model = FCN_ST()
    model.eval()
    batch_size = 64
    model.load_state_dict(torch.load(args.log_dir, torch.device('cpu')))
    train_data = load_dense_data('drive-download-20230401T115945Z-001/train', 2, batch_size)
    valid_data = load_dense_data('drive-download-20230401T115945Z-001/val', 2, batch_size)
    test_data = load_dense_data('drive-download-20230401T115945Z-001/test', 2, batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")    # select GPU device
        model = model.to(device)   

    train_cm = ConfusionMatrix(19)
    valid_cm = ConfusionMatrix(19)
    test_cm = ConfusionMatrix(19)

    with torch.no_grad():
        for i, data in enumerate(train_data):
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)
            outputs = model(inputs)
            outputs = outputs.max(1)[1]
            # print(outputs.shape)

            train_cm.add(outputs, labels)
        print("Clear 1")

        for i, data in enumerate(valid_data):
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)
            outputs = model(inputs)
            outputs = outputs.max(1)[1]

            # print(outputs.shape)

            valid_cm.add(outputs, labels)
        
        print("Clear 2")

        for i, data in enumerate(test_data):
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)
            outputs = model(inputs)
            outputs = outputs.max(1)[1]

            # print(outputs.shape)

            test_cm.add(outputs, labels)

        print("Clear all")

    print("Training: ")
    print(train_cm.global_accuracy)
    print(train_cm.class_accuracy)
    print(train_cm.average_accuracy)
    print(train_cm.iou)
    print(train_cm.class_iou)

    print("Validation: ")
    print(valid_cm.global_accuracy)
    print(valid_cm.class_accuracy)
    print(valid_cm.average_accuracy)
    print(valid_cm.iou)
    print(valid_cm.class_iou)

    print("Testing: ")
    print(test_cm.global_accuracy)
    print(test_cm.class_accuracy)
    print(test_cm.average_accuracy)
    print(test_cm.iou)
    print(test_cm.class_iou)

    print("End")
        

    return train_cm.average_accuracy, valid_cm.average_accuracy, test_cm.average_accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
