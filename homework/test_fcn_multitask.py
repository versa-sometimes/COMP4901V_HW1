import torch
import numpy as np

from .models import FCN_MT 
from .utils import load_dense_data, ConfusionMatrix, DepthError, DenseVisualization
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
    model.load_state_dict(torch.load(args.model_dir, torch.device('cpu')))
    train_data = load_dense_data('drive-download-20230401T115945Z-001/train', 2, batch_size)
    valid_data = load_dense_data('drive-download-20230401T115945Z-001/val', 2, batch_size)
    test_data = load_dense_data('drive-download-20230401T115945Z-001/test', 2, batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")    # select GPU device
        model = model.to(device)   

    train_cm = ConfusionMatrix(19)
    valid_cm = ConfusionMatrix(19)
    test_cm = ConfusionMatrix(19)

    trN, vaN, teN = 0,0,0
    trabs_rel, tra1, tra2, tra3 = 0,0,0,0
    vaabs_rel, vaa1, vaa2, vaa3 = 0,0,0,0
    teabs_rel, tea1, tea2, tea3 = 0,0,0,0

    with torch.no_grad():
        for i, data in enumerate(train_data):
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)
            output_ss, output_dp = model(inputs)
            output_ss = output_ss.max(1)[1]
            # print(outputs.shape)
            train_cm.add(output_ss, labels)

            for i, d in enumerate(output_dp):
                trN += 1
                de = DepthError(d.cpu(), depth[i].cpu())
                nabs_rel, na1, na2, na3 = de.compute_errors
                trabs_rel += nabs_rel
                tra1 += na1
                tra2 += na2
                tra3 += na3

        print("Clear 1")
        for i, data in enumerate(valid_data):
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)
            output_ss, output_dp = model(inputs)
            output_ss = output_ss.max(1)[1]
            # print(outputs.shape)
            valid_cm.add(output_ss, labels)

            for i, d in enumerate(output_dp):
                vaN += 1
                de = DepthError(d.cpu(), depth[i].cpu())
                nabs_rel, na1, na2, na3 = de.compute_errors
                vaabs_rel += nabs_rel
                vaa1 += na1
                vaa2 += na2
                vaa3 += na3

        print("Clear 2")

        for i, data in enumerate(test_data):
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)
            output_ss, output_dp = model(inputs)
            output_ss = output_ss.max(1)[1]
            # print(outputs.shape)
            test_cm.add(output_ss, labels)

            for i, d in enumerate(output_dp):
                teN += 1
                de = DepthError(d.cpu(), depth[i].cpu())
                nabs_rel, na1, na2, na3 = de.compute_errors
                teabs_rel += nabs_rel
                tea1 += na1
                tea2 += na2
                tea3 += na3

        print("Clear all")

    trabs_rel /= trN
    tra1 /= trN
    tra2 /= trN
    tra3 /= trN

    vaabs_rel /= vaN
    vaa1 /= vaN
    vaa2 /= vaN
    vaa3 /= vaN

    teabs_rel /= teN
    tea1 /= teN
    tea2 /= teN
    tea3 /= teN

    print("Training: ")
    print(train_cm.global_accuracy)
    print(train_cm.class_accuracy)
    print(train_cm.average_accuracy)
    print(train_cm.iou)
    print(train_cm.class_iou)
    print(trabs_rel)
    print(tra1)
    print(tra2)
    print(tra3)

    print("Validation: ")
    print(valid_cm.global_accuracy)
    print(valid_cm.class_accuracy)
    print(valid_cm.average_accuracy)
    print(valid_cm.iou)
    print(valid_cm.class_iou)
    print(vaabs_rel)
    print(vaa1)
    print(vaa2)
    print(vaa3)

    print("Testing: ")
    print(test_cm.global_accuracy)
    print(test_cm.class_accuracy)
    print(test_cm.average_accuracy)
    print(test_cm.iou)
    print(test_cm.class_iou)
    print(teabs_rel)
    print(tea1)
    print(tea2)
    print(tea3)

    print("End")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)