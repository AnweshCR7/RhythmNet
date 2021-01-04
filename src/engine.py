from tqdm import tqdm
import torch
import config
from utils.model_utils import save_model_checkpoint


def train_fn(model, data_loader, optimizer, criterion, save_model=False):
    model.train()
    fin_loss = 0
    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        # for (key, value) in data:
        #     data[key] = value.to(config.DEVICE)
        x = data[0].to(config.DEVICE)
        targets = data[1].to(config.DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()

    if save_model:
        save_model_checkpoint(model, optimizer, loss, config.CHECKPOINT_PATH)

    return fin_loss/len(data_loader)


def eval_fn(model, data_loader, criterion):
    model.eval()
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for data in tk:
            # for key, value in data.items():
            #     data[key] = value.to(config.DEVICE)
            x = data[0].to(config.DEVICE)
            targets = data[1].to(config.DEVICE)
            out = model(x)
            loss = criterion(out, targets)
            _, batch_preds = torch.max(out.data, 1)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
        return fin_preds, fin_loss / len(data_loader)