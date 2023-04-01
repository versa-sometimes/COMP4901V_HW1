from .models import CNNClassifier, save_model, SoftmaxCrossEntropyLoss
from .utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
import torchvision
import torch.utils.tensorboard as tb
from datetime import datetime


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


    # if torch.cuda.is_available():
    #     device = torch.device("cuda")    # select GPU device
    #     model = model.to(device)         # move model to GPU memory

    print("All data loaded.")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    N = len(train_data)
    best_vloss = 100000

    for epoch in range(50):
        print("Epoch {}".format(epoch))
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
            # Retrieve loss
            val += t_loss.item()
            train_logger.add_scalar('train', val, i + N * epoch)

        print('Epoch {}, training loss: {}'.format(epoch, val))

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
        valid_logger.add_scalar('valid', valid_loss, i + len(train_data) * epoch)
        print('Epoch {}, validation loss: {}'.format(epoch, valid_loss))

        if valid_loss < best_vloss:
            best_vloss = valid_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
