import scanpy as sc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gears import PertData
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


def load_data(args, data_name="norman"):
    data_path = args["data_path"]

    pert_data = PertData(data_path)
    if args["download_data"]:
        pert_data.load(data_name=data_name)
    pert_data.load(data_path=data_path + data_name)
    pert_data.prepare_split(split=args["experiment"], seed=args["seed"])
    # NOTE: pert_data.set2conditions has the holds the train_val_test_splits
    # NOTE: pert_data.subgroup splits the test data

    adata = pert_data.adata

    # Change uns to top 20 non-zero de used in GEARS
    adata.uns["rank_genes_groups_cov"] = adata.uns['top_non_zero_de_20']
    adata.obs["pert_categories"] = adata.obs["condition_name"]
    adata.obs["Drug1"] = 'ctrl'
    adata.obs["Drug2"] = 'ctrl'
    adata.obs["split"] = 'train'
    adata.obs["subgroup"] = 'train'
    drug_list = []
    for i, name in enumerate(adata.obs["condition"]):
        if name != "ctrl":
            name_split = name.split('+')
            drug_list.append(name_split[0])
            drug_list.append(name_split[1])
            adata.obs["Drug1"][i] = name_split[0]
            adata.obs["Drug2"][i] = name_split[1]
        if name in pert_data.set2conditions["test"]:
            adata.obs["split"][i] = "test"

        if name in pert_data.subgroup["test_subgroup"]["combo_seen0"]:
            adata.obs["subgroup"][i] = "combo_seen0"
        elif name in pert_data.subgroup["test_subgroup"]["combo_seen1"]:
            adata.obs["subgroup"][i] = "combo_seen1"
        elif name in pert_data.subgroup["test_subgroup"]["combo_seen2"]:
            adata.obs["subgroup"][i] = "combo_seen2"
        elif name in pert_data.subgroup["test_subgroup"]["unseen_single"]:
            adata.obs["subgroup"][i] = "unseen_single"

    train_data = adata[adata.obs["split"] != 'test']
    # NOTE: We use a random train/al split and not the one provided by GEARS

    # Unique treatments only for train data: test data might contain additional treatments
    unique_treatments = pd.concat([train_data.obs["Drug1"], train_data.obs["Drug2"]]).unique()
    unique_treatments = np.append(["ctrl"], unique_treatments[unique_treatments != "ctrl"])
    num_treatments = unique_treatments[unique_treatments != "ctrl"].shape[0]  # NOTE: treatment 0 is control

    enc = LabelEncoder()
    enc.classes_ = unique_treatments
    train_data.obs["Drug1_numeric"] = enc.transform(train_data.obs["Drug1"])
    train_data.obs["Drug2_numeric"] = enc.transform(train_data.obs["Drug2"])

    idx_train, idx_test = train_test_split(train_data.obs_names, test_size=0.25, random_state=42)

    test_data = train_data[idx_test]
    train_data = train_data[idx_train]

    control_train = train_data[train_data.obs["control"] == 1]
    train_data = train_data[train_data.obs["control"] == 0]
    control_test = test_data[test_data.obs["control"] == 1]
    test_data = test_data[test_data.obs["control"] == 0]
    control_test = torch.Tensor(control_test.X.todense()).to(device)

    control_train_tds = TensorDataset(torch.Tensor(control_train.X.todense()).to(device),
                                      torch.Tensor(
                                          control_train.obs[["Drug1_numeric", "Drug2_numeric"]].to_numpy()).to(
                                          device))
    control_train_dl = DataLoader(control_train_tds, batch_size=args["batch_size"], shuffle=True, drop_last=True)

    train_data_tds = TensorDataset(torch.Tensor(train_data.X.todense()).to(device),
                                   torch.Tensor(train_data.obs[["Drug1_numeric", "Drug2_numeric"]].to_numpy()).to(
                                       device))
    train_dl = torch.utils.data.DataLoader(train_data_tds, batch_size=args["batch_size"], shuffle=True,
                                           drop_last=True)
    return train_dl, control_train_dl, control_test, test_data, adata, num_treatments, pert_data, enc.classes_