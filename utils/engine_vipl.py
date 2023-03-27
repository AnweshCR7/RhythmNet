from tqdm import tqdm
import torch
from vital_signs.utils import config


def train_fn(model, data_loader, optimizer, loss_fn):
    model.train()
    fin_loss = 0

    target_hr_list = []
    predicted_hr_list = []

    tk_iterator = tqdm(data_loader, total=len(data_loader))

    for batch_data in tk_iterator:
        for data in batch_data:

            for (key, value) in data.items():
                data[key] = value.to(config.DEVICE)

            map_shape = data["st_maps"].shape
            data["st_maps"] = data["st_maps"].reshape((-1, map_shape[3], map_shape[1], map_shape[2]))
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, gru_outputs = model(**data)
                loss = loss_fn(outputs.squeeze(0), gru_outputs, data["target"])
                loss.backward()
                optimizer.step()

            # For each face video, the avg of all HR (bpm) of individual clips (st_map)
            # are computed as the final HR result
            target_hr_list.append(data["target"].mean().item())

            predicted_hr_list.append(outputs.squeeze(0).mean().item())
            fin_loss += loss.item()

    return target_hr_list, predicted_hr_list, fin_loss / (len(data_loader) * config.BATCH_SIZE)


def eval_fn(model, data_loader, loss_fn):
    model.eval()
    fin_loss = 0
    target_hr_list = []
    predicted_hr_list = []
    with torch.no_grad():
        tk_iterator = tqdm(data_loader, total=len(data_loader))
        for batch in tk_iterator:
            for data in batch:
                for (key, value) in data.items():
                    data[key] = value.to(config.DEVICE)

                map_shape = data["st_maps"].shape
                data["st_maps"] = data["st_maps"].reshape((-1, map_shape[3], map_shape[1], map_shape[2]))

                outputs, gru_outputs = model(**data)

                loss = loss_fn(outputs.squeeze(0), gru_outputs, data["target"])
                fin_loss += loss.item()

                target_hr_list.append(data["target"].mean().item())
                predicted_hr_list.append(outputs.squeeze(0).mean().item())

        return target_hr_list, predicted_hr_list, fin_loss / (len(data_loader) * config.BATCH_SIZE)
