from .models import CNNClassifier, save_model, SoftmaxCrossEntropyLoss
from .utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
import torchvision
import torchvision.transforms as transforms
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

    transform1 = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomOrder([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.RandomResizedCrop((224,224))]),

        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_data = load_data(dataset_path='drive-download-20230329T090612Z-001/train_subset', num_workers=0, batch_size=32, transform = transform1)
    valid_data = load_data(dataset_path='drive-download-20230329T090612Z-001/validation_subset', num_workers=0, batch_size=32, transform = transform2)


    if torch.cuda.is_available():
        device = torch.device("cuda")    # select GPU device
        model = model.to(device)         # move model to GPU memory

    print("All data loaded.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.01)
    N = len(train_data)
    best_vloss = 100000

    for epoch in range(50):
        print("Epoch {}".format(epoch))
        model.train()

        val = 0
        for i, data in enumerate(train_data):
            # load data and labels 
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)

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
            # print(t_loss.item())
            train_logger.add_scalar('train', t_loss.item(), i + N * epoch)
            train_logger.flush()

        val /= len(train_data)
        print('Epoch {}, training loss: {}'.format(epoch, val))

        model.eval()

        valid_loss = 0
        for i, data in enumerate(valid_data):
            # load data and labels 
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            # produce one set of outputs
            outputs = model(inputs)

            # calculate loss and grads
            t_loss = loss(outputs, labels)
            # Retrieve loss
            valid_loss += t_loss.item()
            # print(t_loss.item())

        valid_loss /= len(valid_data)
        valid_logger.add_scalar('valid', valid_loss, i + len(train_data) * epoch)
        valid_logger.flush()
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

    # from os import path
    args = parser.parse_args()
    # train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
    # i = 0
    # while True:
    #     i += 1 
    #     train_logger.add_scalar('y=2x', i * 2, i)
    train(args)
