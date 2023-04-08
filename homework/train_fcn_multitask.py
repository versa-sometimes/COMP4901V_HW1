import torch
import numpy as np

from .models import FCN_MT, save_model#, MultiTaskLoss
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
            dense_transforms.ToTensor3()
        ])
    
    valid_tf = dense_transforms.Compose3([
            dense_transforms.ToTensor3()
        ])

    weights = torch.tensor([3.29, 21.9, 4.68, 121.32, 266.84, 117.6, 1022.23, 205.68, 6.13, 118.81, 35.17, 168.36, 460.62, 15.53, 272.62, 501.94, 3536.12, 2287.91, 140.32])
    

    if torch.cuda.is_available():
        device = torch.device("cuda")    # select GPU device
        weights = weights.to(device)

    loss_ss = torch.nn.CrossEntropyLoss(weights, ignore_index=255)
    loss_dp = torch.nn.L1Loss()

    train_data = load_dense_data('drive-download-20230401T115945Z-001/train', num_workers=2, batch_size=64, transform=train_tf)
    valid_data = load_dense_data('drive-download-20230401T115945Z-001/val', num_workers=2, batch_size=64, transform=valid_tf)

    """
    Your code here
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: Use dense_transforms for data augmentation. If you found a good data augmentation parameters for the CNN, use them here too.
    Hint: Use the log function below to debug and visualize your model
    """
    if args.model_dir is not None:
        model.fcn_st.load_state_dict(torch.load(args.model_dir))

    if torch.cuda.is_available():
        model = model.to(device)         # move model to GPU memory

    print("All data loaded.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    N = len(train_data)
    best_vloss = 100000

    max_loss_1 = 0
    max_loss_2 = 0

    for epoch in range(100):

        log_flag = False
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


            if (epoch+1) % 5 == 0:
                if not log_flag:
                    log(train_logger, inputs, labels, outputs_ss, epoch)
                    log_flag=True

            # calculate loss and grads
            # print(weights.is_cuda)
            outputs_ss = outputs_ss.softmax(1)
            t_loss_ss = loss_ss(outputs_ss, labels)
            t_loss_dp = loss_dp(outputs_dp, depth)

            if t_loss_ss.item() > max_loss_1:
                max_loss_1 = t_loss_ss.item()

            if t_loss_dp.item() > max_loss_2:
                max_loss_2 = t_loss_dp.item()

            total_loss = (1/((max_loss_1/t_loss_ss) + (max_loss_2/t_loss_dp)))
            total_loss.backward()

            # Adjust weights
            optimizer.step()
            # Retrieve loss
            val += total_loss.item()

        val /= len(train_data)
        train_logger.add_scalar('train', val, epoch)
        train_logger.flush()
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

            # calculate loss and grads
            outputs_ss = outputs_ss.softmax(1)
            t_loss_ss = loss_ss(outputs_ss, labels)
            t_loss_dp = loss_dp(outputs_dp, depth)
            
            total_loss = (1/((max_loss_1/t_loss_ss) + (max_loss_2/t_loss_dp)))
            # Retrieve loss
            valid_loss += total_loss.item()

        valid_loss /= len(valid_data)
        valid_logger.add_scalar('valid', valid_loss, epoch)
        valid_logger.flush()
        print('Epoch {}, validation loss: {}'.format(epoch, valid_loss))

        if valid_loss < best_vloss:
            best_vloss = valid_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

    # save_model(model)


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
    parser.add_argument('--model_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
