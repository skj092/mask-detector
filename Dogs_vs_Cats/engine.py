import torch 
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import config


def train(model, train_dl, valid_dl, optimizer, loss_fn):
    model.train()
    for epoch in range(config.EPOCHS):
        train_losses = []
        train_acc = []
        val_losses = []
        val_accuracies = []
        loop = tqdm(train_dl)
        for xb, yb in loop:
            xb = xb.to(config.device)
            xb = xb.reshape(xb.shape[0], -1)
            yb = yb.to(config.device)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            prediction = torch.argmax(out, dim=1)
            acc = accuracy_score(yb, prediction)
            train_acc.append(acc)           
        model.eval()
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb = xb.to(config.device)
                xb = xb.reshape(xb.shape[0], -1)
                yb = yb.to(config.device)
                output = model(xb)
                loss = loss_fn(output, yb)
                prediction = torch.argmax(output, dim=1)
                accuracy = accuracy_score(yb, prediction)
                val_losses.append(loss)
                val_accuracies.append(accuracy)
        print(f'epoch={epoch}, train_loss = {sum(train_losses)/len(train_losses)},\
            val_loss={sum(val_losses)/len(val_losses)}, train_acc = {sum(train_acc)/len(train_acc)}, \
                 val_acc={sum(val_accuracies)/len(val_accuracies)}')