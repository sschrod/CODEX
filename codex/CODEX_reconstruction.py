import numpy as np
import torch
from torch import Tensor, nn
from codex.Network_base import network_block, single_layer
from sklearn.metrics import r2_score, mean_squared_error
import copy
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr


class CODEXReconstruction(nn.Module):
    def __init__(self, in_features, num_nodes, num_treatments=None, batch_norm=False, dropout=None, random_seed=42):
        super().__init__()

        self.num_treatments = num_treatments
        torch.manual_seed(random_seed)
        self.encoder = network_block(
            in_features, num_nodes[0:1], num_nodes[2], batch_norm=batch_norm, dropout=dropout, activation=nn.ReLU,
            output_activation=nn.ReLU, output_batch_n=batch_norm, output_dropout=True)

        torch.manual_seed((random_seed) * 2)
        self.latent_rep_dim = num_nodes[2]
        self.T_rep = torch.nn.ModuleList()
        for i in range(num_treatments):
            net = single_layer(num_nodes[2], num_nodes[2], batch_norm=batch_norm, dropout=dropout, activation=nn.ReLU)
            self.T_rep.append(net)

        num_nodes_flip = np.flip(num_nodes)
        torch.manual_seed(random_seed * 2)
        self.decoder = network_block(
            num_nodes_flip[0], num_nodes_flip[1:], in_features * 2, batch_norm=batch_norm, dropout=dropout,
            activation=nn.ReLU, output_activation=None, output_batch_n=False, output_dropout=False)

    def forward(self, input: Tensor, treatment: Tensor):
        latent_rep = torch.zeros((input.shape[0], self.latent_rep_dim), device=input.device)
        dim = input.size()[1]

        embedding = self.encoder(input)
        for t in range(self.num_treatments):  # 0 is for control
            mask = torch.any(treatment == t + 1, dim=1, keepdim=False)
            if torch.sum(mask) > 1:
                latent_rep[mask] += self.T_rep[t](embedding[mask])

        gene_reconstruction = self.decoder(latent_rep)
        # convert variance estimates to a positive value in [1e-3, \infty)
        gene_means = gene_reconstruction[:, :dim]
        gene_vars = nn.functional.softplus(gene_reconstruction[:, dim:]).add(1e-3)
        gene_reconstructions2 = torch.cat([gene_means, gene_vars], dim=1)

        return gene_reconstructions2

    def predict_with_weighted_perturbations(self, input: Tensor, treatment: Tensor, weight: Tensor):
        self.eval()
        latent_rep = torch.zeros((input.shape[0], self.latent_rep_dim), device=input.device)
        dim = input.size()[1]

        embedding = self.encoder(input)
        for t in range(self.num_treatments):  # 0 is for control
            mask = torch.any(treatment == t + 1, dim=1, keepdim=False)
            if torch.sum(mask) > 1:
                latent_rep[mask] += weight[t + 1] * self.T_rep[t](embedding[mask])

        gene_reconstruction = self.decoder(latent_rep)
        gene_means = gene_reconstruction[:, :dim]
        gene_vars = nn.functional.softplus(gene_reconstruction[:, dim:]).add(1e-3)
        gene_reconstructions2 = torch.cat([gene_means, gene_vars], dim=1)
        self.train()
        return gene_reconstructions2

    def predict(self, X, treatment):
        self.eval()
        pred0 = self(X, treatment)
        self.train()
        return pred0

    def predict_numpy(self, X, treatment):
        X = torch.Tensor(X)
        out = self.predict(X, treatment)
        return out[0].detach().numpy()


def evaluate_r2_v2(autoencoder, dataset, genes_control, for_plot=False):
    """
    Measures different quality metrics about an CPA `autoencoder`, when
    tasked to translate some `genes_control` into each of the drug/covariates
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
    num, dim = genes_control.size(0), genes_control.size(1)
    total_cells = len(dataset)

    full_results = []
    pert_categories = []
    for i, pert_category in enumerate(np.unique(dataset.obs.pert_categories)):
        pert_categories.append(pert_category)
        idx = dataset.obs.pert_categories == pert_category

        de_idx = np.where(dataset.var_names.isin(np.array(dataset.uns["rank_genes_groups_cov"][str(pert_category)])))[0]

        if np.sum(idx) > 30:
            emb_drugs = torch.Tensor(
                dataset[idx].obs[["Drug1_numeric", "Drug2_numeric"]].to_numpy()[0:1].repeat(num, 0)).to(device)

            genes_predict = (autoencoder.predict(genes_control, emb_drugs).detach().cpu().numpy())

            mean_predict = genes_predict[:, :dim]
            var_predict = genes_predict[:, dim:]

            yp_m = mean_predict.mean(0)
            yp_v = var_predict.mean(0)

            y_true = np.array(dataset[idx].X.todense())

            # true means and variances
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)

            mean_score.append(r2_score(yt_m, yp_m))
            var_score.append(r2_score(yt_v, yp_v))

            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
            var_score_de.append(r2_score(yt_v[de_idx], yp_v[de_idx]))

            if for_plot:
                print([pert_categories[-1], mean_score[-1], mean_score_de[-1], var_score[-1], var_score_de[-1]])
                full_results.append(
                    [pert_categories[-1], mean_score[-1], mean_score_de[-1], var_score[-1],
                     var_score_de[-1]])

    if for_plot:
        return full_results

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de, var_score, var_score_de]
    ]


def evaluate_mse(autoencoder, dataset, genes_control, for_plot=False):
    """
    Measures different quality metrics about an CPA `autoencoder`, when
    tasked to translate some `genes_control` into each of the drug/covariates
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
    num, dim = genes_control.size(0), genes_control.size(1)
    total_cells = len(dataset)

    full_results = []
    pert_categories = []
    for i, pert_category in enumerate(np.unique(dataset.obs.pert_categories)):
        pert_categories.append(pert_category)
        idx = dataset.obs.pert_categories == pert_category

        de_idx = np.where(dataset.var_names.isin(np.array(dataset.uns["rank_genes_groups_cov"][str(pert_category)])))[0]

        if np.sum(idx) > 30:
            emb_drugs = torch.Tensor(
                dataset[idx].obs[["Drug1_numeric", "Drug2_numeric"]].to_numpy()[0:1].repeat(num, 0)).to(device)

            genes_predict = (autoencoder.predict(genes_control, emb_drugs).detach().cpu().numpy())

            mean_predict = genes_predict[:, :dim]
            var_predict = genes_predict[:, dim:]

            yp_m = mean_predict.mean(0)
            yp_v = var_predict.mean(0)

            y_true = np.array(dataset[idx].X.todense())

            # true means and variances
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)

            mean_score.append(mean_squared_error(yt_m, yp_m))
            var_score.append(pearsonr(yt_m, yp_m)[0])

            mean_score_de.append(mean_squared_error(yt_m[de_idx], yp_m[de_idx]))
            var_score_de.append(pearsonr(yt_m[de_idx], yp_m[de_idx])[0])

            if for_plot:
                print([pert_categories[-1], mean_score[-1], mean_score_de[-1], var_score[-1], var_score_de[-1]])
                full_results.append(
                    [pert_categories[-1], mean_score[-1], mean_score_de[-1], var_score[-1],
                     var_score_de[-1]])

    if for_plot:
        return full_results

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de, var_score, var_score_de]
    ]


def fit_CODEX_reconstruction_r2(args, dl_train_treated, dl_train_vehicle, vehicle_test, test_data, ood_data):
    log = []
    log.append(["epoch", "train_loss", "R2_mean_val", "R2_mean_DEG_val", "R2_var_val", "R2_var_DEG_val",
                "R2_mean_ood", "R2_mean_DEG_ood", "R2_var_ood", "R2_var_DEG_ood"])
    print(log[-1])

    net = CODEXReconstruction(in_features=args["num_features"], num_nodes=args["layers"],
                              num_treatments=args["num_treatments"],
                              batch_norm=args["batch_norm"], dropout=args["dropout"], random_seed=args["seed"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net.to(device)

    optimizer_all = torch.optim.Adam(net.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    optimizer_latent = torch.optim.Adam(net.T_rep.parameters(), lr=args["learning_rate"],
                                        weight_decay=args["weight_decay"])

    best_net = None
    best_val_accuracy = -np.inf
    loss_fkt = nn.GaussianNLLLoss()
    early_stopping_count = 0

    for epoch in range(args["epochs"]):  # loop over the dataset multiple times
        train_loss = 0.0
        for i, data in enumerate(dl_train_treated, 0):
            x = next(dl_train_vehicle.__iter__())[0]  # Get input data randomly sampled from vehicle cells
            y, treatment = data[0], data[1]

            # Step A
            lossA = 0.0
            optimizer_all.zero_grad()
            gene_reconstructions = net(x, treatment)
            dim = gene_reconstructions.size(1) // 2
            gene_means = gene_reconstructions[:, :dim]
            gene_vars = gene_reconstructions[:, dim:]
            lossA += loss_fkt(gene_means, y, gene_vars)
            lossA.backward()
            optimizer_all.step()
            train_loss += lossA.item()

            if args["fine_tuning"]:
                ### Fine-tune step
                optimizer_latent.zero_grad()
                gene_reconstructions = net(x, treatment)
                dim = gene_reconstructions.size(1) // 2
                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                lossB = loss_fkt(gene_means, y, gene_vars)
                lossB.backward()
                optimizer_latent.step()
                train_loss += lossB.item()

        # Validation loss
        val_loss = 0.0
        with torch.no_grad():
            r2_test = evaluate_r2_v2(net, test_data, vehicle_test)
            r2 = evaluate_r2_v2(net, ood_data, vehicle_test)

        log.append([epoch, train_loss, r2_test[0], r2_test[1], r2_test[2], r2_test[3], r2[0], r2[1], r2[2], r2[3]])
        print(log[-1])
        if r2_test[1] > best_val_accuracy:
            early_stopping_count = 0
            best_val_accuracy = r2_test[1]
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


def fit_CODEX_reconstruction_mse(args, dl_train_treated, dl_train_vehicle, vehicle_test, test_data, ood_data=None):
    log = []
    log.append(
        ["epoch", "train_loss", "MSE_test", "MSE_DEG_test", "Pearson_test", "Pearson_DEG_test", "MSE_ood",
         "MSE_DEG_ood", "Pearson_ood", "Pearson_DEG_ood"])
    print(log[-1])

    net = CODEXReconstruction(in_features=args["num_features"], num_nodes=args["layers"],
                              num_treatments=args["num_treatments"], batch_norm=args["batch_norm"],
                              dropout=args["dropout"], random_seed=args["seed"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net.to(device)

    optimizer_all = torch.optim.Adam(net.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    optimizer_latent = torch.optim.Adam(net.T_rep.parameters(), lr=args["learning_rate"],
                                        weight_decay=args["weight_decay"])

    best_net = None
    best_val_accuracy = np.inf
    loss_fkt = nn.GaussianNLLLoss()
    early_stopping_count = 0

    for epoch in range(args["epochs"]):  # loop over the dataset multiple times
        train_loss = 0.0
        for i, data in enumerate(dl_train_treated, 0):
            x = next(dl_train_vehicle.__iter__())[0]  # Get input data randomly sampled from vehicle cells
            y, treatment = data[0], data[1]

            # Step A
            lossA = 0.0
            optimizer_all.zero_grad()
            gene_reconstructions = net(x, treatment)
            dim = gene_reconstructions.size(1) // 2
            gene_means = gene_reconstructions[:, :dim]
            gene_vars = gene_reconstructions[:, dim:]
            lossA += loss_fkt(gene_means, y, gene_vars)
            lossA.backward()
            optimizer_all.step()
            train_loss += lossA.item()

            if args["fine_tuning"]:
                ### Fine-tune step
                optimizer_latent.zero_grad()
                gene_reconstructions = net(x, treatment)
                dim = gene_reconstructions.size(1) // 2
                gene_means = gene_reconstructions[:, :dim]
                gene_vars = gene_reconstructions[:, dim:]
                lossB = loss_fkt(gene_means, y, gene_vars)
                lossB.backward()
                optimizer_latent.step()
                train_loss += lossB.item()

        # Validation loss
        val_loss = 0.0
        with torch.no_grad():
            r2_test = evaluate_mse(net, test_data, vehicle_test)
            if ood_data is not None:
                r2 = evaluate_mse(net, ood_data, vehicle_test)
            else:
                r2 = [-1, -1, -1, -1]

        log.append([epoch, train_loss, r2_test[0], r2_test[1], r2_test[2], r2_test[3], r2[0], r2[1], r2[2], r2[3]])
        print(log[-1])
        if r2_test[1] < best_val_accuracy:
            early_stopping_count = 0
            best_val_accuracy = r2_test[1]
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
