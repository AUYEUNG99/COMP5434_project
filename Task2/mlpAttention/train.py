import torch.nn as nn
import torch
import torch.optim as optim


def train(dataloader, model):
    model.train()
    total_step = len(dataloader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    epochs = 100
    for epoch in range(epochs):
        for i, (x, y, z) in enumerate(dataloader):
            predict = model(x, y)  # output shape is [Batch, num, classes]
            predict = torch.permute(predict, [0, 2, 1])
            optimizer.zero_grad()
            loss = criterion(predict, z)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, total_step,
                                                                         loss.item()))
        scheduler.step()

def eval():
    pass