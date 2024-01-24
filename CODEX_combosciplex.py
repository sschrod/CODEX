import argparse
from codex.CODEX_reconstruction import *
from codex.combosciplex_utils import load_Combosciplex_data

device = "cuda:0" if torch.cuda.is_available() else "cpu"

"""
docker run -it --rm --gpus \"device=0\" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_combosciplex.py -l 1024 512 256
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Combosciplex experiment.")
    parser.add_argument("--num_features", type=int, default=5000)
    parser.add_argument("--num_treatments", type=int, default=None)
    parser.add_argument("-ft", "--fine_tuning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-l", "--layers", nargs='+', type=int, required=True)  # [1024, 512, 256]
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-bs", "--batch_size", type=int, default=1024)
    parser.add_argument("-do", "--dropout", type=float, default=0.2)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0000001)
    parser.add_argument("-bn", "--batch_norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-p", "--patience", type=int, default=50)
    parser.add_argument("-s", "--seed", type=int, default=42)
    # Experient folder
    parser.add_argument("--save_folder", type=str, default="/mnt/models/Combosciplex")
    return dict(vars(parser.parse_args()))


if __name__ == '__main__':
    args = parse_arguments()
    args["experiment_description"] = "DNMLVAE,l={},ft={},lr={},bs={},do={},wd={},bn={},e={},p={},s={}".format(
        args["layers"], args["fine_tuning"],
        args["learning_rate"], args["batch_size"], args["dropout"], args["weight_decay"],
        args["batch_norm"], args["epochs"], args["patience"], args["seed"])
    print(args["experiment_description"])

    train_dl, control_train_dl, control_test, test_data, ood_data, num_treatments = load_Combosciplex_data(args)
    args["num_treatments"] = num_treatments

    net = fit_CODEX_reconstruction_r2(args, train_dl, control_train_dl, control_test, test_data, ood_data)
