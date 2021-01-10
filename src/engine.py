from tqdm import tqdm
import torch
import config
from utils.model_utils import save_model_checkpoint


def train_fn(model, data_loader, optimizer, loss_fn, save_model=False):
    model.train()
    fin_loss = 0
    loss = 0.0
    tk_iterator = tqdm(data_loader, total=len(data_loader))
    for data in tk_iterator:
        # an item of the data is available as a dictionary
        for (key, value) in data.items():
            data[key] = value.to(config.DEVICE)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out = model(**data)
            # preds = out.squeeze(1).squeeze(1)
            loss = loss_fn(out, data["target"])
            loss.backward()
            optimizer.step()

        fin_loss += loss.item()

    # if save_model:
    #     save_model_checkpoint(model, optimizer, loss, config.CHECKPOINT_PATH)

    return fin_loss / len(data_loader)


def eval_fn(model, data_loader, loss_fn):
    model.eval()
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for data in tk:
            for key, value in data.items():
                data[key] = value.to(config.DEVICE)

            with torch.set_grad_enabled(False):
                out = model(**data)
                loss = loss_fn(out, data["target"])
                # _, batch_preds = torch.max(out.data, 1)
            fin_loss += loss.item()
            # fin_preds.append(batch_preds)
        return fin_preds, fin_loss / len(data_loader)
