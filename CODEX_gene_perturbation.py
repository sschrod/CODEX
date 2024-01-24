import argparse
from codex.CODEX_reconstruction import *
from codex.reconstruction_utils import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

"""
docker run -it --rm --gpus \"device=0\" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_gene_perturbation.py -l 512 128 64 -s 1 -dn norman
docker run -it --rm --gpus \"device=0\" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_gene_perturbation.py -l 512 128 64 -s 1 -dn replogle_rpe1_essential --download_data
docker run -it --rm --gpus \"device=0\" -v /sybig/home/ssc/CODEX:/mnt codex python3 -i CODEX_gene_perturbation.py -l 512 128 64 -s 1 -dn replogle_k562_essential --download_data
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Gene perturbation experiments.")
    # training arguments
    parser.add_argument("--num_features", type=int, default=5045)
    parser.add_argument("--num_treatments", type=int, default=None)
    parser.add_argument("-ft", "--fine_tuning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-l", "--layers", nargs='+', type=int, required=True)  # [512, 128, 64]
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-bs", "--batch_size", type=int, default=256)
    parser.add_argument("-do", "--dropout", type=float, default=0.1)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.000001)
    parser.add_argument("-bn", "--batch_norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-p", "--patience", type=int, default=50)
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument("-exp", "--experiment", type=str, default="simulation")
    parser.add_argument("-dp", "--data_path", type=str, default="/mnt/data/")
    parser.add_argument("-dn", "--data_name", type=str,
                        default="norman")  # alternative: "replogle_rpe1_essential", "replogle_k562_essential"
    parser.add_argument("--download_data", action=argparse.BooleanOptionalAction, default=False)
    # Experient folder
    parser.add_argument("--save_folder", type=str, default="/mnt/models/")
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    args = parse_arguments()
    args["save_folder"] = args["save_folder"] + args["data_name"] + "_" + args["experiment"] + "_seed" + str(
        args["seed"])
    args["experiment_description"] = "CODEX,l={},ft={},lr={},bs={},do={},wd={},bn={},e={},p={},s={}".format(
        args["layers"], args["fine_tuning"],
        args["learning_rate"], args["batch_size"], args["dropout"], args["weight_decay"],
        args["batch_norm"], args["epochs"], args["patience"], args["seed"])
    print(args["experiment_description"])

    train_dl, control_train_dl, control_test, test_data, adata, num_treatments, _, _ = load_data(args, data_name=args[
        "data_name"])
    # NOTE: adata.obs.subgroup defines wether the samples are used for training or one of the testing scenarios
    # (["combo_seen2", "combo_seen1", "combo_seen0", "unseen_single"])
    args["num_treatments"] = num_treatments
    args["num_features"] = test_data.shape[1]
    print(num_treatments)

    net = fit_CODEX_reconstruction_mse(args, train_dl, control_train_dl, control_test, test_data)
