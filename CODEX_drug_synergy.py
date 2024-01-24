import numpy as np
import torch
import pandas as pd
from codex.CODEX_Synergy import fit_CODEX_synergy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

"""
docker run -it --rm --gpus \"device=0\" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_drug_synergy.py -l 4096 2048 1024 512 -do 0.2 -wd 0.1 -bs 4096 --fold 1 -p 50 --synergy Synergy_Zip --setting lpo
docker run -it --rm --gpus \"device=1\" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_drug_synergy.py -l 4096 2048 1024 512 -do 0.2 -wd 0.1 -bs 4096 --fold 1 -p 50 --synergy Synergy_Zip --setting lto
docker run -it --rm --gpus \"device=2\" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_drug_synergy.py -l 4096 2048 1024 512 -do 0.2 -wd 0.1 -bs 4096 --fold 1 -p 50 --synergy Synergy_Smean --setting lto
docker run -it --rm --gpus \"device=3\" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_drug_synergy.py -l 4096 2048 1024 512 -do 0.2 -wd 0.1 -bs 4096 --fold 1 -p 50 --synergy Synergy_Smean --setting lpo
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Drug-synergy experiment.")
    parser.add_argument("--num_features", type=int, default=4639)
    parser.add_argument("--num_treatments", type=int, default=670)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("-ft", "--fine_tuning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-l", "--layers", nargs='+', type=int, required=True)  # [512, 128, 64]
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-bs", "--batch_size", type=int, default=4096)
    parser.add_argument("-do", "--dropout", type=float, default=0.2)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.001)
    parser.add_argument("-bn", "--batch_norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-p", "--patience", type=int, default=50)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--synergy", type=str, default="Synergy_Zip")  # or "Synergy_Zip" "Synergy_Smean"
    parser.add_argument("--setting", type=str, default="lto")  # or "lto" "lpo"
    # Experient folder
    parser.add_argument("--save_folder", type=str, default="/mnt/models/MARSY")
    return dict(vars(parser.parse_args()))


if __name__ == '__main__':
    args = parse_arguments()
    args["save_folder"] = args["save_folder"] + "_" + args["synergy"] + "_CV" + str(args["fold"])
    args["experiment_description"] = "CODEX,l={},ft={},lr={},bs={},do={},wd={},bn={},e={},p={},s={}".format(
        args["layers"], args["fine_tuning"],
        args["learning_rate"], args["batch_size"], args["dropout"], args["weight_decay"],
        args["batch_norm"], args["epochs"], args["patience"], args["seed"])

    ### Load data
    data_path = 'data/MARSY'
    cell_line_expression = pd.read_csv(f'{data_path}/75_cell_lines_gene_expression.csv', delimiter=',')
    fold = args["fold"]

    if args["setting"] == "lpo":
        train_idx = pd.read_csv(f'{data_path}/lpo_folds/Pair_Tr{fold}.txt', delimiter=',',
                                header=None).to_numpy().flatten()
        test_idx = pd.read_csv(f'{data_path}/lpo_folds/Pair_Tst{fold}.txt', delimiter=',',
                               header=None).to_numpy().flatten()
        train_idx = train_idx[
            train_idx % 2 == 0]  # we do not need to change the order of treatments since CODEX is invariant
        test_idx = test_idx[test_idx % 2 == 0]

    elif args["setting"] == "lto":
        idx = pd.read_csv(f'data/MARSY/lto_folds/Tpl_Folds_Sig.txt', delimiter=',', header=None).to_numpy().flatten()
        kf = KFold(n_splits=5)
        train_index, test_index = list(kf.split(idx))[args["fold"] - 1]
        train_idx = idx[train_index]
        test_idx = idx[test_index]

    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)
    data = pd.read_csv('data/MARSY/data.csv', delimiter=',')
    unique_treatments = np.unique(data[["Drug1_PC3", "Drug2_PC3"]].to_numpy())

    enc = LabelEncoder()
    enc.classes_ = unique_treatments
    data["Drug1"] = enc.transform(data["Drug1_PC3"])
    data["Drug2"] = enc.transform(data["Drug2_PC3"])

    X_data = cell_line_expression[data["Cell_line"]].T.to_numpy()
    X_train = torch.Tensor(X_data[train_idx]).to(device)
    X_val = torch.Tensor(X_data[val_idx]).to(device)
    X_test = torch.Tensor(X_data[test_idx]).to(device)

    y_train = torch.Tensor(data.iloc[train_idx][args["synergy"]].values).to(device)
    y_val = torch.Tensor(data.iloc[val_idx][args["synergy"]].values).to(device)
    y_test = torch.Tensor(data.iloc[test_idx][args["synergy"]].values).to(device)

    t_train = torch.Tensor(data.iloc[train_idx][["Drug1", "Drug2"]].values).to(device)
    t_val = torch.Tensor(data.iloc[val_idx][["Drug1", "Drug2"]].values).to(device)
    t_test = torch.Tensor(data.iloc[test_idx][["Drug1", "Drug2"]].values).to(device)

    data_tds = TensorDataset(X_train, y_train, t_train)
    train_dl = DataLoader(data_tds, batch_size=args["batch_size"], shuffle=True, drop_last=True)

    ### Fit model
    net = fit_CODEX_synergy(args, train_dl, X_val, y_val, t_val, X_test, y_test, t_test)
