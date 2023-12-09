#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tetris import TetrisNN

def main():
    epochs = 5000
    model = TetrisNN(44).to('mps')

    # load weights...
    model.load_state_dict(torch.load("supervised.pt"))
    model.eval()

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, amsgrad=True)
    dataset = np.load('dataset.npy')
    print(dataset.shape)
    X = torch.tensor(np.reshape(dataset[:, :20], newshape=(len(dataset), 1, 20)), device='mps', dtype=torch.float32)
    y = torch.tensor(dataset[:, 20:], device='mps', dtype=torch.float32)

    for epoch in range(epochs):
        print('ITERATION = %d/%d' %(epoch + 1, epochs))
        predictions = model(X)
        criterion = nn.SmoothL1Loss()
        loss = criterion(predictions, y)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(model.parameters(), 100)
        optimizer.step()

    torch.save(model.state_dict(), "supervised.pt")


if __name__ == '__main__':
    main()
