import torch
import numpy as np

from .models import FCN_MT 
from .utils import load_dense_data, ConfusionMatrix, DepthError
from . import dense_transforms
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for both segmentation and depth estimation tasks
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task
    Hint: use DepthError for you to calculate rel, a1, a2, and a3 for the depth estimation task. 
    """
    model = FCN_MT()
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
        abs_rel, a1, a2, a3 = 0,0,0,0
        for i, data in enumerate(train_data):
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)
            output_ss, output_dp = model(inputs)
            output_ss = output_ss.max(1)[1]
            # print(outputs.shape)
            train_cm.add(output_ss, labels)
            de = DepthError(output_dp, depth)
            nabs_rel, na1, na2, na3 = de.compute_errors()
            abs_rel += nabs_rel
            a1 += na1
            a2 += na2
            a3 += na3

        abs_rel /= i
        a1 /= i
        a2 /= i
        a3 /= i

        print("Clear 1")

        abs_rel, a1, a2, a3 = 0,0,0,0
        for i, data in enumerate(valid_data):
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)
            output_ss, output_dp = model(inputs)
            output_ss = output_ss.max(1)[1]
            # print(outputs.shape)
            valid_cm.add(output_ss, labels)
        
        print("Clear 2")

        for i, data in enumerate(test_data):
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)
            output_ss, output_dp = model(inputs)
            output_ss = output_ss.max(1)[1]
            # print(outputs.shape)
            test_cm.add(output_ss, labels)

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

    # return accuracy, mIoU, rel, a1, a2, a3


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
