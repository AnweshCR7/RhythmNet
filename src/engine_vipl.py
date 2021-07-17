from tqdm import tqdm
import torch
import config
import h5py
import os
import numpy as np
from utils.model_utils import save_model_checkpoint
from loss_func.TorchLossComputer import TorchLossComputer


def store_as_hdfs(data, save_dir, filename, frame_rate=None):
    output_file_path = os.path.join(save_dir, f'{filename}.h5')
    h5f = h5py.File(output_file_path, 'w', libver="latest")
    h5f.create_dataset("extractor", data=data, compression="gzip", compression_opts=9)
    if frame_rate is not None:
        h5f.attrs.create("frame_rate", frame_rate)
    # h5f.create_dataset("signal", data=signal)
    # print(f"Images Saved using hdfs at: {output_file_path}")
    h5f.close()
    return


def extractor_fn(model, data_loader, optimizer, loss_fn):
    # model.train()
    model.eval()
    fin_loss = 0
    loss = 0.0

    target_hr_list = []
    predicted_hr_list = []
    tk_iterator = tqdm(data_loader, total=len(data_loader))
    estimator_outs = []

    with torch.no_grad():
        for batch in tk_iterator:
            # for data in batch:
            batch = batch.to(config.DEVICE)
                # an item of the data is available as a dictionary
                # for (key, value) in data.items():
                #     if key == "video_file_name":
                #         data[key] = value
                #     else:
                #         data[key] = value.to(config.DEVICE)

                # optimizer.zero_grad()
                # with torch.set_grad_enabled(True):
                #     # outputs, gru_outputs = model(**data)
                #     # w/o GRU
                #     outputs = model(data["st_maps"])
                #     # outputs = model(**data)
                #     loss = loss_fn(outputs.squeeze(0), data["target"])
                #     # with GRU
                #     # loss = loss_fn(outputs.squeeze(0), gru_outputs, data["target"])
                #     loss.backward()
                #     optimizer.step()
            outputs = model(batch)
            print('asdasd')
            # store_as_hdfs(outputs.detach().cpu().numpy(), config.EXRACTOR_SAVE_DIR, data["video_file_name"], frame_rate=None)

                # estimator_outs.append(outputs)
                # # "For each face video, the avg of all HR (bpm) of individual clips are computed as the final HR result
                # # target_hr_batch = list(data["target"].mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
                # target_hr_list.append(data["target"].mean().item())
                #
                # # predicted_hr_batch = list(outputs.squeeze(2).mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
                # predicted_hr_list.append(outputs.squeeze(0).mean().item())
                # fin_loss += loss.item()

    # return outputs


def train_extractor_fn(model, data_loader, optimizer, loss_fn, train_ds):
    model.train()
    # model.eval()
    fin_loss = 0
    loss = 0.0

    target_hr_list = []
    predicted_hr_list = []
    tk_iterator = tqdm(data_loader, total=len(data_loader))
    estimator_outs = []

    # with torch.no_grad():
    for batch_idx, batch in enumerate(tk_iterator):
        # for data in batch:
        # batch = batch.to(config.DEVICE)
        # for data in batch:
        train_rmses = []
        train_losses = []
        train_aes = []
        data, target = batch[0], batch[1]
        if config.DEVICE == 'cuda':
            data = data.to('cuda')
            target = target.to('cuda')
        # an item of the data is available as a dictionary
        # for (key, value) in data.items():
        #     if key == "video_file_name":
        #         data[key] = value
        #     else:
        #         data[key] = value.to(config.DEVICE)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
        #     # outputs, gru_outputs = model(**data)
        #     # w/o GRU
            outputs = model(data)
            Fs, regularization_factor = train_ds.get_fps_and_regularization_factor(batch_idx * int(config.BATCH_SIZE))
            train_loss, train_rmse, train_ae = TorchLossComputer.cross_entropy_power_spectrum_loss(outputs.squeeze(1), target, Fs, regularization_factor)
            print('check')
            #     # outputs = model(**data)
            #     loss = loss_fn(outputs.squeeze(0), data["target"])
            #     # with GRU
            #     # loss = loss_fn(outputs.squeeze(0), gru_outputs, data["target"])
            #     loss.backward()
            #     optimizer.step()
            if len(train_rmses) == 0:
                train_rmses = train_rmse
                train_losses = train_loss
                train_aes = train_ae
            else:
                train_rmses = torch.cat((train_rmses, train_rmse), dim=0)
                train_losses = torch.cat((train_losses, train_loss), dim=0)
                train_aes = torch.cat((train_aes, train_ae), dim=0)

            train_loss.backward()
            optimizer.step()

        return train_losses.data.cpu().numpy(), train_rmses.data.cpu().numpy(), train_aes.data.cpu().numpy()

        print('asdasd')
        # store_as_hdfs(outputs.detach().cpu().numpy(), config.EXRACTOR_SAVE_DIR, data["video_file_name"], frame_rate=None)

            # estimator_outs.append(outputs)
            # # "For each face video, the avg of all HR (bpm) of individual clips are computed as the final HR result
            # # target_hr_batch = list(data["target"].mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
            # target_hr_list.append(data["target"].mean().item())
            #
            # # predicted_hr_batch = list(outputs.squeeze(2).mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
            # predicted_hr_list.append(outputs.squeeze(0).mean().item())
            # fin_loss += loss.item()

    # return outputs

def estimator_fn(model, data_loader, optimizer, loss_fn):
    # model.train()
    model.eval()
    fin_loss = 0
    loss = 0.0

    target_hr_list = []
    predicted_hr_list = []
    tk_iterator = tqdm(data_loader, total=len(data_loader))
    estimator_outs = []
    target_hr_list_save = []
    predicted_hr_list_save = []

    # with torch.no_grad():
    for batch in tk_iterator:
        for data in batch:
            # an item of the data is available as a dictionary
            for (key, value) in data.items():
                if key == "video_file_name":
                    data[key] = value
                else:
                    data[key] = value.to(config.DEVICE)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # Estimator model
                outputs, to_save = model(data["input"].view(1, 1, -1))
                # Possibly MAE/RMSE?
                loss = loss_fn(outputs.flatten(), data["target"].squeeze(0))
                loss.backward()
                optimizer.step()

            # estimator_outs.append(outputs)
            # # "For each face video, the avg of all HR (bpm) of individual clips are computed as the final HR result
            # # target_hr_batch = list(data["target"].mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
            # REMOVE MEAN!!!!! ESTIMATOR O/Ps a single HR value
            target_hr_list.append(data["target"].mean().item())
            # target_hr_list_save.append(to_save)
            predicted_hr_list_save.append(list(to_save.cpu().detach().numpy()))

            #
            # # predicted_hr_batch = list(outputs.squeeze(2).mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
            predicted_hr_list.append(outputs.squeeze(0).mean().item())
            fin_loss += loss.item()

    return target_hr_list, predicted_hr_list, fin_loss / (len(data_loader)*config.BATCH_SIZE)


def train_fn(model, data_loader, optimizer, loss_fn):
    print("defunct")


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
                    if key == "video_file_name":
                        data[key] = value
                    else:
                        data[key] = value.to(config.DEVICE)

                outputs = model(data["input"].view(1, 1, -1))
                loss = loss_fn(outputs.flatten(), data["target"])

                fin_loss += loss.item()
                # target_hr_batch = list(data["target"].mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
                target_hr_list.append(data["target"].mean().item())

                # predicted_hr_batch = list(outputs.squeeze(2).mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
                predicted_hr_list.append(outputs.squeeze(0).mean().item())

        return target_hr_list, predicted_hr_list, fin_loss / (len(data_loader)*config.BATCH_SIZE)
