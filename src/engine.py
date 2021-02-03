from tqdm import tqdm
import torch
import config
from utils.model_utils import save_model_checkpoint


def train_fn(model, data_loader, optimizer, loss_fn):
    model.train()
    fin_loss = 0
    loss = 0.0

    target_hr_list = []
    predicted_list = []
    tk_iterator = tqdm(data_loader, total=len(data_loader))
    for data in tk_iterator:
        # an item of the data is available as a dictionary
        for (key, value) in data.items():
            data[key] = value.to(config.DEVICE)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(**data)
            # loss = loss_fn(out.squeeze(1), data["target"])
            loss = loss_fn(outputs, data["target"])
            loss.backward()
            optimizer.step()
        target_hr_list.append(data["target"].item())
        predicted_list.append(outputs[2].mean().item())
        fin_loss += loss.item()

    return target_hr_list, predicted_list, fin_loss / len(data_loader)


def eval_fn(model, data_loader, loss_fn):
    model.eval()
    fin_loss = 0
    target_hr_list = []
    predicted_list = []
    with torch.no_grad():
        tk_iterator = tqdm(data_loader, total=len(data_loader))
        for data in tk_iterator:
            for (key, value) in data.items():
                data[key] = value.to(config.DEVICE)

            # with torch.set_grad_enabled(False):
            _, _, out = model(**data)
            # loss = loss_fn(out, data["target"])
            # _, batch_preds = torch.max(out.data, 1)
            # fin_loss += loss.item()
            predicted_list.append(out.mean().item())
            target_hr_list.append(data["target"].item())

        return target_hr_list, predicted_list
