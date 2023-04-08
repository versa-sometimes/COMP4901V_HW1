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
    # batch_size = 64

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
    train_data = load_data(dataset_path='drive-download-20230329T090612Z-001/train_subset', num_workers=2, batch_size=64, transform = transform1)
    valid_data = load_data(dataset_path='drive-download-20230329T090612Z-001/validation_subset', num_workers=2, batch_size=64, transform = transform2)


    if torch.cuda.is_available():
        device = torch.device("cuda")    # select GPU device
        model = model.to(device)         # move model to GPU memory

    print("All data loaded.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    N = len(train_data)
    best_vloss = 100000

    for epoch in range(50):
        print("Epoch {}".format(epoch))
        model.train()

        val = 0
        for i, data in enumerate(train_data):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            t_loss = loss(outputs, labels)
            t_loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            val += t_loss.item()

        val /= len(train_data)
        train_logger.add_scalar('train', val, epoch)
        train_logger.flush()
        print('Epoch {}, training loss: {}'.format(epoch, val))

        model.eval()

        valid_loss = 0
        for i, data in enumerate(valid_data):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            t_loss = loss(outputs, labels)
            valid_loss += t_loss.item()

        valid_loss /= len(valid_data)
        valid_logger.add_scalar('valid', valid_loss, epoch)
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

    args = parser.parse_args()
    train(args)
