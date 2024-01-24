import scanpy as sc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_Combosciplex_data(args=None):
    adata = sc.read("/mnt/data/Combosciplex.h5ad")

    unique_treatments = pd.concat([adata.obs["Drug1"], adata.obs["Drug2"]]).unique()
    unique_treatments = np.append(["DMSO"],unique_treatments[unique_treatments!="DMSO"])
    num_treatments = unique_treatments[unique_treatments!="DMSO"].shape[0] #NOTE: treatment 0 is control


    ood_data = adata[adata.obs["ood"]]
    train_data = adata[~adata.obs["ood"]]

    idx_train, idx_test = train_test_split(train_data.obs_names, test_size=0.25, random_state=42)

    test_data = train_data[idx_test]
    train_data = train_data[idx_train]

    control_train = train_data[train_data.obs["control"]]
    train_data = train_data[~train_data.obs["control"]]
    control_test = test_data[test_data.obs["control"]]
    test_data = test_data[~test_data.obs["control"]]
    control_test = torch.Tensor(control_test.X.todense()).to(device)

    if args is not None:
        control_train_tds = TensorDataset(torch.Tensor(control_train.X.todense()).to(device), torch.Tensor(control_train.obs[["Drug1_numeric","Drug2_numeric"]].to_numpy()).to(device))
        control_train_dl = DataLoader(control_train_tds, batch_size=args["batch_size"], shuffle=True, drop_last=True)

        train_data_tds = TensorDataset(torch.Tensor(train_data.X.todense()).to(device), torch.Tensor(train_data.obs[["Drug1_numeric","Drug2_numeric"]].to_numpy()).to(device))
        train_dl = torch.utils.data.DataLoader(train_data_tds, batch_size=args["batch_size"], shuffle=True, drop_last=True)
        return train_dl, control_train_dl, control_test, test_data, ood_data, num_treatments

    else:
        return control_test, test_data, ood_data, num_treatments