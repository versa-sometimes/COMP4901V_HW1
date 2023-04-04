import torch
import numpy as np

from .models import FCN_MT, save_model, MultiTaskLoss
from .utils import load_dense_data, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
from datetime import datetime


def train(args):
    from os import path
    model = FCN_MT()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)


    train_tf = dense_transforms.Compose3([
            # dense_transforms.RandomHorizontalFlip(),
            # dense_transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            dense_transforms.ToTensor3(),
            dense_transforms.Normalize3(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    valid_tf = dense_transforms.Compose3([
            dense_transforms.ToTensor3(),
            dense_transforms.Normalize3(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    weights = torch.tensor([3.29, 21.9, 4.68, 121.32, 266.84, 117.6, 1022.23, 205.68, 6.13, 118.81, 35.17, 168.36, 460.62, 15.53, 272.62, 501.94, 3536.12, 2287.91, 140.32])
    

    if torch.cuda.is_available():
        device = torch.device("cuda")    # select GPU device
        weights = weights.to(device)

    loss = MultiTaskLoss(weights)

    train_data = load_dense_data('drive-download-20230401T115945Z-001/train', num_workers=2, batch_size=32, transform=train_tf)
    valid_data = load_dense_data('drive-download-20230401T115945Z-001/val', num_workers=2, batch_size=32, transform=valid_tf)

    """
    Your code here
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: Use dense_transforms for data augmentation. If you found a good data augmentation parameters for the CNN, use them here too.
    Hint: Use the log function below to debug and visualize your model
    """
    if torch.cuda.is_available():
        model = model.to(device)         # move model to GPU memory

    print("All data loaded.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    N = len(train_data)
    best_vloss = 100000

    for epoch in range(100):
        print("Epoch {}".format(epoch))
        model.train()
        # print("Here")

        val = 0
        for i, data in enumerate(train_data):
            # load data and labels 
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)

            # zero the grads
            optimizer.zero_grad()

            # produce one set of outputs
            outputs_ss, outputs_dp = model(inputs)
            # if torch.cuda.is_available():
            #     outputs = outputs.to(device)

            # calculate loss and grads
            # print(weights.is_cuda)
            t_loss_ss = loss(outputs_ss, labels, outputs_dp, depth)
            t_loss_ss.backward()

            # Adjust weights
            optimizer.step()
            # Retrieve loss
            val += t_loss_ss.item()
            # print(t_loss.item())
            train_logger.add_scalar('train', t_loss_ss.item(), i + N * epoch)
            train_logger.flush()

        val /= len(train_data)
        print('Epoch {}, training loss: {}'.format(epoch, val))

        model.eval()

        valid_loss = 0
        for i, data in enumerate(valid_data):
            # load data and labels 
            inputs, labels, depth = data
            if torch.cuda.is_available():
                inputs, labels, depth = inputs.to(device), labels.to(device), depth.to(device)
            # produce one set of outputs
            outputs_ss, outputs_dp = model(inputs)
            # if torch.cuda.is_available():
            #     outputs = outputs.to(device)

            # calculate loss and grads
            # print(weights.is_cuda)
            t_loss_ss = loss(outputs_ss, labels, outputs_dp, depth)
            t_loss_ss.backward()

            # Adjust weights
            # Retrieve loss
            val += t_loss_ss.item()

        valid_loss /= len(valid_data)
        valid_logger.add_scalar('valid', valid_loss, i + len(train_data) * epoch)
        valid_logger.flush()
        print('Epoch {}, validation loss: {}'.format(epoch, valid_loss))

        if valid_loss < best_vloss:
            best_vloss = valid_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
