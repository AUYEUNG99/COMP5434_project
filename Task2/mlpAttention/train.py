import torch.nn as nn
import torch
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


def train(dataloader, model, testloader):
    model.train()
    model.to(device)
    total_step = len(dataloader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    epochs = 50
    best_acc = 0
    for epoch in range(epochs):
        for i, (x, y, z) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            # print(y.shape)
            z = z.to(device)
            predict = model(x, y)  # output shape is [Batch, num, classes]
            predict = torch.permute(predict, [0, 2, 1])
            optimizer.zero_grad()
            loss = criterion(predict, z)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, total_step,
                                                                         loss.item()))

                accuracy = eval(model, testloader)
                if accuracy > best_acc:
                    best_acc = accuracy
                    torch.save(model, "./5.pth")

        # scheduler.step()


def eval(model, testloader):
    """
    since test data doesn't have labels at all, directly use train data to measure accuracy
    """
    model.eval()
    totalNum = 0
    correctNum = 0
    with torch.no_grad():
        for i, (x, y, z) in enumerate(testloader):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            predict = model(x, y)  # [Batch, sequence_len, classes]
            totalNum += x.shape[0] * x.shape[1]
            predict = torch.argmax(predict, dim=-1)
            correctNum += torch.sum(predict == z)
    acc = correctNum / totalNum
    print("Accuracy: {}".format(acc))
    return acc
