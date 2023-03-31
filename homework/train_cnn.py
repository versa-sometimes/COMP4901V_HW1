from .models import CNNClassifier, save_model, SoftmaxCrossEntropyLoss
from .utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
import torchvision
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    batch_size = 64

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    """
    Your code here
    """
    # train_logger = tb.SummaryWriter('cnn')
    loss = SoftmaxCrossEntropyLoss()
    train_data = load_data('drive-download-20230329T090612Z-001/train_subset', 2, batch_size)
    valid_data = load_data('drive-download-20230329T090612Z-001/validation_subset', 2, batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    N = len(train_data)
    for iter in range(5):
        model.train()

        val = 0
        for i, data in enumerate(train_data):
            # load data and labels 
            inputs, labels = data

            # zero the grads
            optimizer.zero_grad()

            # produce one set of outputs
            outputs = model(inputs)

            # calculate loss and grads
            t_loss = loss(outputs, labels)
            t_loss.backward()

            # Adjust weights
            optimizer.step()

            print("Progress!")

            # Retrieve loss
            val += t_loss.item()
            train_logger.add_scalar('train', val, i + N * iter)

        model.eval()

        valid_loss = 0
        for i, data in enumerate(valid_data):
            # load data and labels 
            inputs, labels = data
            # produce one set of outputs
            outputs = model(inputs)

            # calculate loss and grads
            t_loss = loss(outputs, labels)
            # Retrieve loss
            valid_loss += t_loss.item()

        valid_loss /= len(valid_data)
        valid_logger.add_scalar('valid', valid_loss, i + len(train_data) * iter)


        print("Progress222!")
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
