import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from codex.Network_base import network_block, single_layer
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import copy
import os


class CODEXDose(nn.Module):
    def __init__(self, in_features, num_nodes, num_treatments=None, batch_norm=False, dropout=None, random_seed=42):
        super().__init__()

        self.num_treatments = num_treatments
        torch.manual_seed(random_seed)
        self.encoder = network_block(
            in_features, num_nodes[0:1], num_nodes[2], batch_norm=batch_norm, dropout=dropout, activation=nn.ReLU,
            output_activation=nn.ReLU, output_batch_n=batch_norm, output_dropout=True)

        self.latent_rep_dim = num_nodes[2]
        self.T_rep = torch.nn.ModuleList()
        for i in range(num_treatments):
            net = single_layer(
                num_nodes[2] + 1, num_nodes[2], batch_norm=batch_norm, dropout=dropout, activation=nn.ReLU)
            self.T_rep.append(net)

        num_nodes_flip = np.flip(num_nodes)
        self.decoder = network_block(
            num_nodes_flip[0], num_nodes_flip[1:], in_features * 2, batch_norm=batch_norm, dropout=dropout,
            activation=nn.ReLU, output_activation=None, output_batch_n=False, output_dropout=False)

    def forward(self, input: Tensor, treatment_and_dosages: Tensor):

        dosages = treatment_and_dosages
        treatment = torch.argmax(treatment_and_dosages, dim=1)

        latent_rep = torch.zeros((input.shape[0], self.latent_rep_dim), device=input.device)
        dim = input.size()[1]

        embedding = self.encoder(input)
        for t in range(self.num_treatments):
            mask = treatment == t
            if torch.sum(mask) > 1:
                embedding_and_dose = torch.cat([embedding[mask], dosages[mask, t:t + 1]], dim=1)
                latent_rep[mask] += self.T_rep[t](embedding_and_dose)

        gene_reconstruction = self.decoder(latent_rep)
        gene_means = gene_reconstruction[:, :dim]
        gene_vars = nn.functional.softplus(gene_reconstruction[:, dim:]).add(1e-3)
        gene_reconstructions2 = torch.cat([gene_means, gene_vars], dim=1)

        return gene_reconstructions2, latent_rep

    def predict(self, X, treatment):
        self.eval()
        pred0 = self(X, treatment)[0]
        self.train()
        return pred0

    def predict_numpy(self, X, treatment):
        X = torch.Tensor(X)
        out = self.predict(X, treatment)
        return out[0].detach().numpy()

    def predict_latent(self, X, treatment):
        self.eval()
        pred0 = self(X, treatment)[1]
        self.train()
        return pred0


def evaluate_r2(autoencoder, dataset, genes_control, num_DEG=50, for_plot=False, set="ood"):
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
    for pert_category in np.unique(dataset.pert_categories):
        # print(pert_category)
        idx = dataset.pert_categories == pert_category

        de_idx = np.where(dataset.var_names.isin(np.array(dataset.de_genes[str(pert_category)])))[0][:num_DEG]

        # if np.sum(idx) > 30:
        if np.sum(idx) > 30:
            emb_drugs = torch.Tensor(dataset.drugs[idx][0:1].repeat(num, 1)).to(device)

            genes_predict = (autoencoder.predict(genes_control.to(device), emb_drugs).detach().cpu().numpy())

            mean_predict = genes_predict[:, :dim]
            var_predict = genes_predict[:, dim:]

            yp_m = mean_predict.mean(0)
            yp_v = var_predict.mean(0)
            # estimate metrics only for reasonably-sized drug/cell-type combos
            y_true = dataset[idx][0]

            # true means and variances
            yt_m = y_true.mean(axis=0)
            yt_v = y_true.var(axis=0)

            mean_score.append(r2_score(yt_m, yp_m))
            var_score.append(r2_score(yt_v, yp_v))

            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
            var_score_de.append(r2_score(yt_v[de_idx], yp_v[de_idx]))

            str_pert = pert_category.split("_")
            full_results.append(
                ["A549", str_pert[1], float(str_pert[2]), mean_score[-1], mean_score_de[-1], var_score[-1],
                 var_score_de[-1], "CODEX", np.sum(idx), set])

    if for_plot:
        return full_results

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de, var_score, var_score_de]
    ]


def fit_DNMLVAE_ES(args, dl_train_treated, dl_train_vehicle, control_X, test_data, ood_data):
    log = []
    log.append(
        ["epoch", "train_loss", "R2_mean_test", "R2_mean_DEG_test", "R2_var_test", "R2_var_DEG_test", "R2_mean_ood",
         "R2_mean_DEG_ood", "R2_var_ood", "R2_var_DEG_ood"])
    print(log[-1])
    net = CODEXDose(in_features=args["num_features"], num_nodes=args["layers"],
                    num_treatments=args["num_treatments"], batch_norm=args["batch_norm"],
                    dropout=args["dropout"], random_seed=args["seed"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net.to(device)

    optimizer_all = torch.optim.Adam(net.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    optimizer_latent = torch.optim.Adam(net.T_rep.parameters(), lr=args["learning_rate"],
                                        weight_decay=args["weight_decay"])

    best_net = None
    best_val_accuracy = -np.Inf
    loss_fkt = nn.GaussianNLLLoss()
    early_stopping_count = 0
    for epoch in range(args["epochs"]):  # loop over the dataset multiple times
        train_loss = 0.0
        for i, data in enumerate(dl_train_treated, 0):
            x = next(dl_train_vehicle.__iter__())[0]  # Get input data randomly sampled from vehicle cells
            y, treatment = data[0], data[1]

            if torch.cuda.is_available():
                x, y, treatment = x.cuda(), y.cuda(), treatment.cuda()

            # Step A
            lossA = 0.0
            optimizer_all.zero_grad()
            gene_reconstructions = net(x, treatment)[0]
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
                gene_reconstructions = net(x, treatment)[0]
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
            r2_test = evaluate_r2(net, test_data, control_X)
            r2 = evaluate_r2(net, ood_data, control_X)

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
