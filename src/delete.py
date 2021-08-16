from datetime import datetime
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

from datasets import CFD, VGGFace2
from facenet_mtl import FacenetMTL
from utils import pass_epoch, percentile_splitter_CFDVGG, print_epoch, sacc, evaluate_epoch

Any = object()


# from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN
class TypeAlias:
    # Class for defining generic aliases for library types.
    def __init__(self, target_type: type) -> None: ...

    def __getitem__(self, typeargs: Any) -> Any: ...


Dict = TypeAlias(object)


def train(
        net: torch.nn.Module,
        train_loader: DataLoader,
        parameters: Dict[str, float],
        dtype: torch.dtype,
        device: torch.device,
) -> nn.Module:
    """
    Train CNN on provided data set.

    Args:
        net: initialized neural network
        train_loader: DataLoader containing training set
        parameters: dictionary containing parameters to be passed to the optimizer.
            - lr: default (0.001)
            - momentum: default (0.0)
            - weight_decay: default (0.0)
            - num_epochs: default (1)
        dtype: torch dtype
        device: torch device
    Returns:
        nn.Module: trained CNN.
    """
    ########### INITIALIZE from utils

    (main_loss,
     dom_loss,
     trs_loss,
     att_loss,
     cmp_loss,
     lkl_loss,
     chr_loss,
     msc_loss,
     sad_loss,
     ang_loss) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    (dom_cnt,
     trs_cnt,
     att_cnt,
     cmp_cnt,
     lkl_cnt,
     chr_cnt,
     msc_cnt,
     sad_cnt,
     ang_cnt) = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    (mtr_dom,
     mtr_trs,
     mtr_att,
     mtr_cmp,
     mtr_lkl,
     mtr_chr,
     mtr_msc,
     mtr_sad,
     mtr_ang) = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    ###########

    net.train()
    # Define loss and optimizer
    criterion = nn.NLLLoss(reduction="sum")
    loss_fn = torch.nn.SmoothL1Loss()

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=parameters.get("lr", 0.001),
        momentum=parameters.get("momentum", 0.0),
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=50, min_lr=1e-8, verbose=True,
    )
    num_epochs = parameters.get("num_epochs", 1)

    # Train Network
    # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
    for _ in range(num_epochs):

        try:
            for i_batch, sample in enumerate(train_loader):
                x = sample["image"]
                x = x.float()
                x = x.to(device)
                y_dom = sample["dom"]
                y_dom = y_dom.to(device).float()
                y_trs = sample["trs"]
                y_trs = y_trs.to(device).float()
                y_att = sample["att"]
                y_att = y_att.to(device).float()
                y_cmp = sample["cmp"]
                y_cmp = y_cmp.to(device).float()
                y_lkl = sample["lkl"]
                y_lkl = y_lkl.to(device).float()
                y_chr = sample["chr"]
                y_chr = y_chr.to(device).float()
                y_msc = sample["msc"]
                y_msc = y_msc.to(device).float()
                y_sad = sample["sad"]
                y_sad = y_sad.to(device).float()
                y_ang = sample["ang"]
                y_ang = y_ang.to(device).float()

                (y_out_dom,
                 y_out_trs,
                 y_out_att,
                 y_out_cmp,
                 y_out_lkl,
                 y_out_chr,
                 y_out_msc,
                 y_out_sad,
                 y_out_ang) = net(x)
                loss_batch = 0
                loss_cnt = 0

                # for each target, find individual loss by filling in the zero values with -1?
                if torch.count_nonzero(y_dom > 0) > 0:
                    loss1 = loss_fn(y_out_dom[y_dom > 0], y_dom.view(-1, 1)[y_dom > 0])
                    mtr_dom += sacc(y_out_dom[y_dom > 0], y_dom[y_dom > 0])
                    loss_batch += loss1
                    loss_cnt += 1
                    loss1_np = loss1.detach().cpu().numpy()
                    dom_loss += loss1_np
                    dom_cnt += 1
                if torch.count_nonzero(y_trs > 0) > 0:
                    loss2 = loss_fn(y_out_trs[y_trs > 0], y_trs.view(-1, 1)[y_trs > 0])
                    mtr_trs += sacc(y_out_trs[y_trs > 0], y_trs[y_trs > 0])
                    loss_batch += loss2
                    loss_cnt += 1
                    loss2_np = loss2.detach().cpu().numpy()
                    trs_loss += loss2_np
                    trs_cnt += 1
                if torch.count_nonzero(y_att > 0) > 0:
                    loss3 = loss_fn(y_out_att[y_att > 0], y_att.view(-1, 1)[y_att > 0])
                    mtr_att += sacc(y_out_att[y_att > 0], y_att[y_att > 0])
                    loss_batch += loss3
                    loss_cnt += 1
                    loss3_np = loss3.detach().cpu().numpy()
                    att_loss += loss3_np
                    att_cnt += 1
                if torch.count_nonzero(y_cmp > 0) > 0:
                    loss4 = loss_fn(y_out_cmp[y_cmp > 0], y_cmp.view(-1, 1)[y_cmp > 0])
                    mtr_cmp += sacc(y_out_cmp[y_cmp > 0], y_cmp[y_cmp > 0])
                    loss_batch += loss4
                    loss_cnt += 1
                    loss4_np = loss4.detach().cpu().numpy()
                    cmp_loss += loss4_np
                    cmp_cnt += 1
                if torch.count_nonzero(y_lkl > 0) > 0:
                    loss5 = loss_fn(y_out_lkl[y_lkl > 0], y_lkl.view(-1, 1)[y_lkl > 0])
                    mtr_lkl += sacc(y_out_lkl[y_lkl > 0], y_lkl[y_lkl > 0])
                    loss_batch += loss5
                    loss_cnt += 1
                    loss5_np = loss5.detach().cpu().numpy()
                    lkl_loss += loss5_np
                    lkl_cnt += 1
                if torch.count_nonzero(y_chr > 0) > 0:
                    loss6 = loss_fn(y_out_chr[y_chr > 0], y_chr.view(-1, 1)[y_chr > 0])
                    mtr_chr += sacc(y_out_chr[y_chr > 0], y_chr[y_chr > 0])
                    loss_batch += loss6
                    loss_cnt += 1
                    loss6_np = loss6.detach().cpu().numpy()
                    chr_loss += loss6_np
                    chr_cnt += 1
                if torch.count_nonzero(y_msc > 0) > 0:
                    loss7 = loss_fn(y_out_msc[y_msc > 0], y_msc.view(-1, 1)[y_msc > 0])
                    mtr_msc += sacc(y_out_msc[y_msc > 0], y_msc[y_msc > 0])
                    loss_batch += loss7
                    loss_cnt += 1
                    loss7_np = loss7.detach().cpu().numpy()
                    msc_loss += loss7_np
                    msc_cnt += 1
                if torch.count_nonzero(y_sad > 0) > 0:
                    loss8 = loss_fn(y_out_sad[y_sad > 0], y_sad.view(-1, 1)[y_sad > 0])
                    mtr_sad += sacc(y_out_sad[y_sad > 0], y_sad[y_sad > 0])
                    loss_batch += loss8
                    loss_cnt += 1
                    loss8_np = loss8.detach().cpu().numpy()
                    sad_loss += loss8_np
                    sad_cnt += 1
                if torch.count_nonzero(y_ang > 0) > 0:
                    loss9 = loss_fn(y_out_ang[y_ang > 0], y_ang.view(-1, 1)[y_ang > 0])
                    mtr_ang += sacc(y_out_ang[y_ang > 0], y_ang[y_ang > 0])
                    loss_batch += loss9
                    loss_cnt += 1
                    loss9_np = loss9.detach().cpu().numpy()
                    ang_loss += loss9_np
                    ang_cnt += 1

                loss_batch = loss_batch / loss_cnt

                loss_batch.float().backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_batch = loss_batch.detach().cpu().numpy()
                main_loss += loss_batch

            main_loss = main_loss / (i_batch + 1)
            #     you divide over individual counters
            dom_loss = dom_loss / dom_cnt
            trs_loss = trs_loss / trs_cnt
            att_loss = att_loss / att_cnt
            cmp_loss = cmp_loss / cmp_cnt
            lkl_loss = lkl_loss / lkl_cnt
            chr_loss = chr_loss / chr_cnt
            msc_loss = msc_loss / msc_cnt
            sad_loss = sad_loss / sad_cnt
            ang_loss = ang_loss / ang_cnt

            mtr_dom /= dom_cnt
            mtr_trs /= trs_cnt
            mtr_att /= att_cnt
            mtr_cmp /= cmp_cnt
            mtr_lkl /= lkl_cnt
            mtr_chr /= chr_cnt
            mtr_msc /= msc_cnt
            mtr_sad /= sad_cnt
            mtr_ang /= ang_cnt

            metrics = {}
            metrics["sacc"] = np.asarray(
                [
                    mtr_dom,
                    mtr_trs,
                    mtr_att,
                    mtr_cmp,
                    mtr_lkl,
                    mtr_chr,
                    mtr_msc,
                    mtr_sad,
                    mtr_ang,
                ]
            )

        except Exception as e:
            print(e)
            continue

    return net


def evaluate(
        net: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device
) -> float:
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """

    (main_loss,
     dom_loss,
     trs_loss,
     att_loss,
     cmp_loss,
     lkl_loss,
     chr_loss,
     msc_loss,
     sad_loss,
     ang_loss) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    (dom_cnt,
     trs_cnt,
     att_cnt,
     cmp_cnt,
     lkl_cnt,
     chr_cnt,
     msc_cnt,
     sad_cnt,
     ang_cnt) = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    (mtr_dom,
     mtr_trs,
     mtr_att,
     mtr_cmp,
     mtr_lkl,
     mtr_chr,
     mtr_msc,
     mtr_sad,
     mtr_ang) = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    loss_fn = torch.nn.SmoothL1Loss()
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i_batch, sample in enumerate(data_loader):
            try:
                x = sample["image"]
                x = x.float()
                x = x.to(device)
                y_dom = sample["dom"]
                y_dom = y_dom.to(device).float()
                y_trs = sample["trs"]
                y_trs = y_trs.to(device).float()
                y_att = sample["att"]
                y_att = y_att.to(device).float()
                y_cmp = sample["cmp"]
                y_cmp = y_cmp.to(device).float()
                y_lkl = sample["lkl"]
                y_lkl = y_lkl.to(device).float()
                y_chr = sample["chr"]
                y_chr = y_chr.to(device).float()
                y_msc = sample["msc"]
                y_msc = y_msc.to(device).float()
                y_sad = sample["sad"]
                y_sad = y_sad.to(device).float()
                y_ang = sample["ang"]
                y_ang = y_ang.to(device).float()

                (y_out_dom,
                 y_out_trs,
                 y_out_att,
                 y_out_cmp,
                 y_out_lkl,
                 y_out_chr,
                 y_out_msc,
                 y_out_sad,
                 y_out_ang) = net(x)
                loss_batch = 0
                loss_cnt = 0

                # for each target, find individual loss by filling in the zero values with -1?
                if torch.count_nonzero(y_dom > 0) > 0:
                    loss1 = loss_fn(y_out_dom[y_dom > 0], y_dom.view(-1, 1)[y_dom > 0])
                    mtr_dom += sacc(y_out_dom[y_dom > 0], y_dom[y_dom > 0])
                    loss_batch += loss1
                    loss_cnt += 1
                    loss1_np = loss1.detach().cpu().numpy()
                    dom_loss += loss1_np
                    dom_cnt += 1
                if torch.count_nonzero(y_trs > 0) > 0:
                    loss2 = loss_fn(y_out_trs[y_trs > 0], y_trs.view(-1, 1)[y_trs > 0])
                    mtr_trs += sacc(y_out_trs[y_trs > 0], y_trs[y_trs > 0])
                    loss_batch += loss2
                    loss_cnt += 1
                    loss2_np = loss2.detach().cpu().numpy()
                    trs_loss += loss2_np
                    trs_cnt += 1
                if torch.count_nonzero(y_att > 0) > 0:
                    loss3 = loss_fn(y_out_att[y_att > 0], y_att.view(-1, 1)[y_att > 0])
                    mtr_att += sacc(y_out_att[y_att > 0], y_att[y_att > 0])
                    loss_batch += loss3
                    loss_cnt += 1
                    loss3_np = loss3.detach().cpu().numpy()
                    att_loss += loss3_np
                    att_cnt += 1
                if torch.count_nonzero(y_cmp > 0) > 0:
                    loss4 = loss_fn(y_out_cmp[y_cmp > 0], y_cmp.view(-1, 1)[y_cmp > 0])
                    mtr_cmp += sacc(y_out_cmp[y_cmp > 0], y_cmp[y_cmp > 0])
                    loss_batch += loss4
                    loss_cnt += 1
                    loss4_np = loss4.detach().cpu().numpy()
                    cmp_loss += loss4_np
                    cmp_cnt += 1
                if torch.count_nonzero(y_lkl > 0) > 0:
                    loss5 = loss_fn(y_out_lkl[y_lkl > 0], y_lkl.view(-1, 1)[y_lkl > 0])
                    mtr_lkl += sacc(y_out_lkl[y_lkl > 0], y_lkl[y_lkl > 0])
                    loss_batch += loss5
                    loss_cnt += 1
                    loss5_np = loss5.detach().cpu().numpy()
                    lkl_loss += loss5_np
                    lkl_cnt += 1
                if torch.count_nonzero(y_chr > 0) > 0:
                    loss6 = loss_fn(y_out_chr[y_chr > 0], y_chr.view(-1, 1)[y_chr > 0])
                    mtr_chr += sacc(y_out_chr[y_chr > 0], y_chr[y_chr > 0])
                    loss_batch += loss6
                    loss_cnt += 1
                    loss6_np = loss6.detach().cpu().numpy()
                    chr_loss += loss6_np
                    chr_cnt += 1
                if torch.count_nonzero(y_msc > 0) > 0:
                    loss7 = loss_fn(y_out_msc[y_msc > 0], y_msc.view(-1, 1)[y_msc > 0])
                    mtr_msc += sacc(y_out_msc[y_msc > 0], y_msc[y_msc > 0])
                    loss_batch += loss7
                    loss_cnt += 1
                    loss7_np = loss7.detach().cpu().numpy()
                    msc_loss += loss7_np
                    msc_cnt += 1
                if torch.count_nonzero(y_sad > 0) > 0:
                    loss8 = loss_fn(y_out_sad[y_sad > 0], y_sad.view(-1, 1)[y_sad > 0])
                    mtr_sad += sacc(y_out_sad[y_sad > 0], y_sad[y_sad > 0])
                    loss_batch += loss8
                    loss_cnt += 1
                    loss8_np = loss8.detach().cpu().numpy()
                    sad_loss += loss8_np
                    sad_cnt += 1
                if torch.count_nonzero(y_ang > 0) > 0:
                    loss9 = loss_fn(y_out_ang[y_ang > 0], y_ang.view(-1, 1)[y_ang > 0])
                    mtr_ang += sacc(y_out_ang[y_ang > 0], y_ang[y_ang > 0])
                    loss_batch += loss9
                    loss_cnt += 1
                    loss9_np = loss9.detach().cpu().numpy()
                    ang_loss += loss9_np
                    ang_cnt += 1

                loss_batch = loss_batch / loss_cnt

                loss_batch = loss_batch.detach().cpu().numpy()
                main_loss += loss_batch

            except Exception as e:
                print(e)
                continue

    #             # move data to proper dtype and device
    #             inputs = inputs.to(dtype=dtype, device=device)
    #             labels = labels.to(device=device)
    #             outputs = net(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

        main_loss = main_loss / (i_batch + 1)
        #     you divide over individual counters
        dom_loss = dom_loss / dom_cnt
        trs_loss = trs_loss / trs_cnt
        att_loss = att_loss / att_cnt
        cmp_loss = cmp_loss / cmp_cnt
        lkl_loss = lkl_loss / lkl_cnt
        chr_loss = chr_loss / chr_cnt
        msc_loss = msc_loss / msc_cnt
        sad_loss = sad_loss / sad_cnt
        ang_loss = ang_loss / ang_cnt

        mtr_dom /= dom_cnt
        mtr_trs /= trs_cnt
        mtr_att /= att_cnt
        mtr_cmp /= cmp_cnt
        mtr_lkl /= lkl_cnt
        mtr_chr /= chr_cnt
        mtr_msc /= msc_cnt
        mtr_sad /= sad_cnt
        mtr_ang /= ang_cnt

        metrics = {}
        metrics["sacc"] = np.asarray(
            [
                mtr_dom,
                mtr_trs,
                mtr_att,
                mtr_cmp,
                mtr_lkl,
                mtr_chr,
                mtr_msc,
                mtr_sad,
                mtr_ang,
            ]
        )


    return metrics["sacc"].mean()


def train_evaluate(parameterization):
    #     print(parameterization)
    img_root_cfd = "CFD"
    img_root_vgg = "VGG"
    gender = "male"
    suffix = "4to10"
    df_cfd = pd.read_csv(f"cfd_{gender}_{suffix}.csv")
    df_vgg = pd.read_csv(f"vgg_{gender}_{suffix}.csv")

    trait_classes = percentile_splitter_CFDVGG(df_cfd, df_vgg)

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, train_size=None, test_size=0.2, random_state=42
    )
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, train_size=None, test_size=0.5, random_state=42
    )

    tr_indices = []
    val_indices = []
    tst_indices = []

    for train_index, val_test_index in msss.split(
            np.zeros(len(trait_classes)), trait_classes
    ):
        tr_indices.append(train_index)
        for val_index, test_index in msss2.split(
                np.zeros(len(val_test_index)), trait_classes[val_test_index]
        ):
            val_indices.append(val_index)
            tst_indices.append(test_index)

    tr_ind = tr_indices[0]
    val_ind = val_indices[0]
    tst_ind = tst_indices[0]

    tr_ind_cfd = tr_ind[tr_ind < len(df_cfd)]
    tr_ind_vgg = tr_ind[tr_ind >= len(df_cfd)] - len(df_cfd)
    val_ind_cfd = val_ind[val_ind < len(df_cfd)]
    val_ind_vgg = val_ind[val_ind >= len(df_cfd)] - len(df_cfd)
    tst_ind_cfd = tst_ind[tst_ind < len(df_cfd)]
    tst_ind_vgg = tst_ind[tst_ind >= len(df_cfd)] - len(df_cfd)

    df_tr_cfd = df_cfd.iloc[tr_ind_cfd].reset_index(drop=True)
    df_tr_vgg = df_vgg.iloc[tr_ind_vgg].reset_index(drop=True)
    df_val_cfd = df_cfd.iloc[val_ind_cfd].reset_index(drop=True)
    df_val_vgg = df_vgg.iloc[val_ind_vgg].reset_index(drop=True)
    df_tst_cfd = df_cfd.iloc[tst_ind_cfd].reset_index(drop=True)
    df_tst_vgg = df_vgg.iloc[tst_ind_vgg].reset_index(drop=True)

    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    facenet_mtl = FacenetMTL(freeze_until=parameterization.get("freeze_until", 13)).to(device)

    # These are all dataloaders for cfd
    cfd_tr = CFD(df_tr_cfd, img_root_cfd)
    cfd_val = CFD(df_val_cfd, img_root_cfd, augment=False)
    cfd_tst = CFD(df_tst_cfd, img_root_cfd, augment=False)

    # Dataloaders for VGG
    vgg_tr = VGGFace2(df_tr_vgg, img_root_vgg)
    vgg_val = VGGFace2(df_val_vgg, img_root_vgg, augment=False)
    vgg_tst = VGGFace2(df_tst_vgg, img_root_vgg, augment=False)

    # Concatenate the two for a single dataloader now: ConcatDataset is a torch function
    dt_tr = ConcatDataset([cfd_tr, vgg_tr])
    dt_val = ConcatDataset([cfd_val, vgg_val])
    dt_tst = ConcatDataset([cfd_tst, vgg_tst])

    batch_size = 16
    n_workers = 4

    # Then effectievly form the main dataloaders
    train_loader = DataLoader(
        dt_tr, batch_size=batch_size, num_workers=n_workers, drop_last=True, shuffle=True,
    )
    val_loader = DataLoader(
        dt_val, batch_size=batch_size, num_workers=n_workers, shuffle=False
    )
    tst_loader = DataLoader(
        dt_tst, batch_size=batch_size, num_workers=n_workers, shuffle=False
    )

    #     lr = 0.1
    #     momentum = 0.9
    #     epochs = 1
    #     loss_fn = torch.nn.SmoothL1Loss()

    #     optimizer = optim.SGD(
    #         filter(lambda p: p.requires_grad, facenet_mtl.parameters()),
    #         lr=lr,
    #         momentum=momentum,
    #     )

    #     scheduler = ReduceLROnPlateau(
    #         optimizer, mode="max", factor=0.1, patience=50, min_lr=1e-8, verbose=True,
    #     )

    metrics = {"sacc": sacc}

    tr_loss_hist = []
    tr_sacc_hist = []
    val_loss_hist = []
    val_sacc_hist = []
    val_std_hist = []

    best_sacc = 0

    dtype = torch.float
    cur_date = datetime.now().strftime("%d%m%y")

    #     facenet_mtl.train()
    facenet_mtl = train(net=facenet_mtl, train_loader=train_loader, parameters=parameterization, dtype=dtype,
                        device=device)

    return evaluate(
        net=facenet_mtl,
        data_loader=tst_loader,
        dtype=dtype,
        device=device,
    )