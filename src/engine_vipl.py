import config
import torch
from tqdm import tqdm


def train_fn(model, data_loader, optimizer, loss_fn):
    model.train()
    fin_loss = 0
    loss = 0.0

    target_hr_list = []
    predicted_hr_list = []
    tk_iterator = tqdm(data_loader, total=len(data_loader))
    batched_data = []
    for batch in tk_iterator:
        for data in batch:
            # an item of the data is available as a dictionary
            for (key, value) in data.items():
                data[key] = value.to(config.DEVICE)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(**data)
                # w/o GRU
                loss = loss_fn(outputs.squeeze(0), data["target"])
                # loss = loss_fn(outputs, data["target"])
                loss.backward()
                optimizer.step()
            # "For each face video, the avg of all HR (bpm) of individual clips are computed as the final HR result
            # target_hr_batch = list(data["target"].mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
            # For per map metrics
            # target_hr_list.extend(list(data["target"].cpu().detach().numpy()))
            target_hr_list.append(data["target"].mean().item())

            # predicted_hr_batch = list(outputs.squeeze(2).mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
            # For per map metrics
            # predicted_hr_list.extend(list(outputs.squeeze(0).cpu().detach().numpy()))
            predicted_hr_list.append(outputs.squeeze(0).mean().item())
            fin_loss += loss.item()

    return target_hr_list, predicted_hr_list, fin_loss / (len(data_loader)*config.BATCH_SIZE)


def eval_fn(model, data_loader, loss_fn):
    # model.eval()
    fin_loss = 0
    target_hr_list = []
    predicted_hr_list = []
    with torch.no_grad():
        tk_iterator = tqdm(data_loader, total=len(data_loader))
        for batch in tk_iterator:
            for data in batch:
                for (key, value) in data.items():
                    data[key] = value.to(config.DEVICE)

                # with torch.set_grad_enabled(False):
                outputs = model(**data)
                # _, _, out = model(**data)
                loss = loss_fn(outputs.squeeze(0), data["target"])
                # _, batch_preds = torch.max(out.data, 1)
                fin_loss += loss.item()
                # target_hr_batch = list(data["target"].mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
                # For per map metrics
                # target_hr_list.extend(list(data["target"].cpu().detach().numpy()))
                target_hr_list.append(data["target"].mean().item())

                # predicted_hr_batch = list(outputs.squeeze(2).mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
                # For per map metrics
                # predicted_hr_list.extend(list(outputs.squeeze(0).cpu().detach().numpy()))
                predicted_hr_list.append(outputs.squeeze(0).mean().item())

        return target_hr_list, predicted_hr_list, fin_loss / (len(data_loader)*config.BATCH_SIZE)
