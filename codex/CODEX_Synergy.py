import numpy as np
import torch
from torch import Tensor, nn
from codex.Network_base import network_block, single_layer
import copy
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr


class CODEXSynergyModel(nn.Module):
    def __init__(self, in_features, num_nodes, num_treatments=None, batch_norm=False, dropout=None, random_seed=42):
        super().__init__()

        self.num_treatments = num_treatments
        torch.manual_seed(random_seed)
        self.encoder = network_block(
            in_features, num_nodes[0], num_nodes[1],
            batch_norm=batch_norm, dropout=dropout, activation=nn.ReLU, output_activation=nn.ReLU,
            output_batch_n=batch_norm, output_dropout=True)

        self.latent_rep_dim = num_nodes[2]
        self.T_rep = torch.nn.ModuleList()
        for i in range(num_treatments):
            net = single_layer(
                num_nodes[1], num_nodes[2],
                batch_norm=batch_norm, dropout=dropout, activation=nn.ReLU)
            self.T_rep.append(net)

        self.decoder = network_block(
            num_nodes[2], num_nodes[3], 1,
            batch_norm=batch_norm, dropout=dropout, activation=nn.ReLU, output_activation=None,
            output_batch_n=False, output_dropout=False)

    def forward(self, input: Tensor, treatment: Tensor):
        latent_rep = torch.zeros((input.shape[0], self.latent_rep_dim), device=input.device)
        embedding = self.encoder(input)
        for t in range(self.num_treatments):
            mask = torch.any(treatment == t, dim=1, keepdim=False)
            if torch.sum(mask) > 1:
                latent_rep[mask] += self.T_rep[t](embedding[mask])

        out = self.decoder(latent_rep)
        return out

    def predict(self, X, treatment):
        self.eval()
        pred0 = self(X, treatment)
        self.train()
        return pred0

    def predict_numpy(self, X, treatment):
        X = torch.Tensor(X)
        treatment = torch.Tensor(treatment)
        out = self.predict(X, treatment)
        return out.detach().numpy()


def fit_CODEX_synergy(args, train_dl, x_val, y_val, t_val, x_test, y_test, t_test):
    log = []
    log.append(["epoch", "train_loss", "MSE_val", "MSE_test", "Pearson_test", "Spearman_test"])
    print(log[-1])
    net = CODEXSynergyModel(in_features=args["num_features"], num_nodes=args["layers"],
                            num_treatments=args["num_treatments"], batch_norm=args["batch_norm"],
                            dropout=args["dropout"], random_seed=args["seed"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    net.to(device)

    optimizer_all = torch.optim.Adam(net.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    optimizer_latent = torch.optim.Adam(net.T_rep.parameters(), lr=args["learning_rate"],
                                        weight_decay=args["weight_decay"])

    best_net = None
    best_val_accuracy = np.inf
    loss_fkt_val = nn.MSELoss()
    loss_fkt = nn.MSELoss()
    early_stopping_count = 0
    for epoch in range(args["epochs"]):  # loop over the dataset multiple times
        train_loss = 0.0
        for i, data in enumerate(train_dl, 0):
            x, y, treatment = data[0], data[1], data[2]

            # Step A
            lossA = 0.0
            optimizer_all.zero_grad()
            pred = net(x, treatment).flatten()
            lossA += loss_fkt(pred, y)
            lossA.backward()
            optimizer_all.step()
            train_loss = lossA.item()

            if args["fine_tuning"]:
                optimizer_latent.zero_grad()
                pred = net(x, treatment).flatten()
                lossB = loss_fkt(pred, y)
                lossB.backward()
                optimizer_latent.step()
                train_loss += lossB.item()

        with torch.no_grad():
            pred_val = net.predict(x_val, t_val).flatten()
            MSE_val = loss_fkt_val(pred_val, y_val)

            pred_test = net.predict(x_test, t_test).flatten()
            MSE_test = loss_fkt_val(pred_test, y_test)
            P_test = pearsonr(pred_test.detach().cpu().numpy(), y_test.detach().cpu().numpy())[0]
            S_test = spearmanr(pred_test.detach().cpu().numpy(), y_test.detach().cpu().numpy())[0]

        log.append([epoch, train_loss, MSE_val.item(), MSE_test.item(), P_test, S_test])
        print(log[-1])

        if MSE_val < best_val_accuracy:
            early_stopping_count = 0
            best_val_accuracy = MSE_val
            best_net = copy.deepcopy(net)
        if early_stopping_count > args["patience"]:
            break
        early_stopping_count = early_stopping_count + 1

    if not os.path.exists("{}/{}/".format(args["save_folder"], args["experiment_description"])):
        os.makedirs("{}/{}/".format(args["save_folder"], args["experiment_description"]))
    torch.save(best_net, "{}/{}/model.pt".format(args["save_folder"], args["experiment_description"]))
    pd.DataFrame(log).to_csv("{}/{}/log.csv".format(args["save_folder"], args["experiment_description"]),
                             index=False, header=False)
    return net
