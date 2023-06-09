import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pickle as pkl
import json

import argparse
import datetime
import time
import os
import matplotlib.pyplot as plt

from utils import create_model, save_model
from dataloader import BirdsDataset
from model import UNet


@torch.no_grad()
def test_network(model, loader):
    model.eval()
    accuracy = []
    for model_input, target in loader.dataset:
        model_input = model_input.unsqueeze(0)
        prediction = model(model_input)

        predictions = prediction.argmax(dim=1)[0].cpu().numpy()
        targets = target.argmax(dim=0).cpu().numpy()

        # calculate the accuracy
        count = np.count_nonzero(target == predictions)
        accuracy.append(count)

    n_samples = len(loader.dataset)
    random_index = int(np.random.random()*n_samples)
    model_input, target = loader.dataset[random_index]
    model_input = model_input.unsqueeze(0)
    prediction = model(model_input)
    targets = target.argmax(dim=0)

    # plot the predictions
    prediction = prediction[0].cpu().numpy()
    targets = targets.cpu().numpy()
    prediction = np.abs(prediction)
    prediction = prediction / np.sum(prediction, axis=0)

    x = np.arange(targets.shape[0])
    plt.figure(figsize=(10, 3))
    bottom = np.zeros(targets.shape[0])
    for i, row in enumerate(prediction):
        color = ['gray' if val != i else 'green' for val in targets]
        plt.bar(x, row, bottom=bottom, color=color, width=1)
        bottom += row
        plt.step(x, bottom, color='black', linewidth=1, where='post')
    plt.show()

    # Calculate overall accuracy
    return sum(accuracy) / len(accuracy)


def train_network(model, config, optimizer):
    # create the dataset
    dataset = BirdsDataset(config, config.device)

    # split the dataset into train and test
    train_size = int(len(dataset) * config.train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Train the model
    model.train()
    total_step = len(train_loader)
    mse = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(config.num_epochs):
        # print the epoch and current time
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Start Epoch: {epoch + 1}/{config.num_epochs}   {time_now}   (lr: {lr})")

        # loop through the training loader
        for i, (model_input, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(model_input)
            loss = mse(outputs, targets)

            # calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{config.num_epochs}]',
                      f'Step [{i + 1}/{total_step}]',
                      f'Loss: {loss.item():.4f}')

        # add the number of epochs
        config.current_epoch += 1

        # save the model if enough time has passed
        if abs(time.time() - start_time) >= config.save_time or epoch == config.num_epochs - 1:
            save_model(model, optimizer, config)
            test_error = test_network(model, test_loader)
            print(f'The test accuracy is: {test_error}')
            start_time = time.time()

    return model, config, optimizer


def main():
    # Load JSON config file
    with open(args.config_path) as f:
        config_json = json.load(f)

    # create the config file
    class Config:
        pass
    config = Config()
    for key, data in config_json.items():
        setattr(config, key, data)
    config.current_epoch = 0

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    # create the model
    model, config, optimizer = create_model(config, load=args.load, lr=args.lr)
    model = model.to(device)

    # print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f'Number of trainable parameters: {num_params:,}',
        f'with epoch {config.current_epoch}',
    )

    # train the network
    if args.train:
        train_network(model, config, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Model')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--config_path', type=str,
                        help='Path to the configuration file')
    parser.add_argument('--load', action='store_true',
                        help='load a model')
    parser.add_argument('--lr', type=float, default=False,
                        help='change the learning rate')
    args = parser.parse_args()

    main()
