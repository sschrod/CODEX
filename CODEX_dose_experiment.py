import torch
from torch.utils.data import DataLoader
import argparse
from codex.CODEX_Dose import *
from codex.dose_utils import load_dataset_splits

"""
docker run -it --rm --gpus \"device=0\" -v /sybig/home/ssc/CODEX:/mnt codex python3 CODEX_dose_experiment.py -l 512 128 64 
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sciplex2 experiment.")
    parser.add_argument("--num_features", type=int, default=4999)
    parser.add_argument("--num_treatments", type=int, default=None)
    parser.add_argument("-ft", "--fine_tuning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-l", "--layers", nargs='+', type=int, required=True)  # [512, 128, 64]
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-bs", "--batch_size", type=int, default=1024)
    parser.add_argument("-do", "--dropout", type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.000001)
    parser.add_argument("-bn", "--batch_norm", type=bool, default=False)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-p", "--patience", type=int, default=50)
    parser.add_argument("-s", "--seed", type=int, default=42)

    # Experient folder
    parser.add_argument("--save_folder", type=str, default="/mnt/models/GSM")
    return dict(vars(parser.parse_args()))


if __name__ == '__main__':
    args = parse_arguments()
    args["experiment_description"] = "DNMLVAE,l={},ft={},lr={},bs={},do={},wd={},bn={},e={},p={},s={}".format(
        args["layers"], args["fine_tuning"],
        args["learning_rate"], args["batch_size"], args["dropout"], args["weight_decay"],
        args["batch_norm"], args["epochs"], args["patience"], args["seed"])

    datasets = load_dataset_splits("data/CPA_datasets/GSM_new.h5ad", "condition", "dose_val", ["cell_type"], "split",
                                   None)
    args["num_treatments"] = datasets["train_treated"][:][1].numpy().shape[1]

    train_dl = DataLoader(datasets["train_treated"], batch_size=args["batch_size"], shuffle=True, drop_last=True)
    control_dl = DataLoader(datasets["train_vehicle"], batch_size=args["batch_size"], shuffle=True)
    control_X = datasets["test_vehicle"][:][0]

    net = fit_DNMLVAE_ES(args, train_dl, control_dl, control_X, datasets["test_treated"], datasets["ood"])
