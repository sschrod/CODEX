import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

"""File to load and combine sciplex3 and combosciplex data"""


"""
docker run -it --rm -v /sybig/home/ssc/CODEX:/mnt ssc_cf python3 -i /mnt/data/Preprocess_Combosciplex.py
"""


def rank_genes_groups(
        adata,
        groupby,
        pool_doses=False,
        n_genes=50,
        rankby_abs=True,
        key_added="rank_genes_groups_cov",
        return_dict=False,
):
    """
    Function was adapted from https://github.com/facebookresearch/CPA/blob/main/cpa/data.py"
    The MIT License
    Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
    """

    control_group_cov = (adata[adata.obs["control"]].obs[groupby].values[0])
    sc.tl.rank_genes_groups(adata, groupby=groupby, rankby_abs=rankby_abs, reference="DMSO_DMSO", n_genes=n_genes)

    de_genes = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
    gene_dict = {}
    for group in de_genes:
        gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict



print("Loading Combosciplex data...")
adata1 = sc.read_mtx('/mnt/data/Combosciplex/GSE206741_count_matrix.mtx').T
obs1 = pd.read_csv('/mnt/data/Combosciplex/GSE206741_cell_metadata.tsv',sep='\t')
var_names1 = pd.read_csv('/mnt/data/Combosciplex/GSE206741_gene_metadata.tsv',sep='\t')
adata1.obs=obs1
adata1.obs['Drug1'] = [x.split('\xa0')[0] for x in adata1.obs['Drug1']]
adata1.obs['Drug2'] = [x.split('\xa0')[0] for x in adata1.obs['Drug2']]
adata1.var_names=var_names1["gene_short_name"]
print("Finished loading Combosciplex data")
print()
print(adata1.obs.shape)


### Match drug names
t1 = pd.concat([adata1.obs['Drug1'],adata1.obs['Drug2']]).unique()
adata =adata1

### Preprocess data following: https://github.com/theislab/cpa-reproducibility/blob/main/preprocessing/sciplex3.ipynb
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)


### Select untreated cells
adata.obs["control"] = np.logical_and(adata.obs["Drug1"]=='DMSO', adata.obs["Drug2"]=='DMSO') #14455 cells

### Select held-out-combinations for ood
ho_combinations=['Givinostat_Cediranib',
                 'panobinostat_SRT2104',
                 'Dacinostat_Danusertib',
                 'SRT2104_Alvespimycin',
                 'panobinostat_Alvespimycin']

ood = np.zeros_like(adata.obs['Drug1'])
pert_categories = np.empty(adata.obs['Drug1'].shape, dtype=object)

unique_treatments = pd.concat([adata.obs["Drug1"], adata.obs["Drug2"]]).unique()
num_treatments = unique_treatments[unique_treatments != "DMSO"].shape[0] #NOTE: treatment 0 is control
unique_treatments = np.append(["DMSO"], unique_treatments[unique_treatments != "DMSO"])

enc = LabelEncoder()
enc.classes_ = unique_treatments
adata.obs["Drug1_numeric"] = enc.transform(adata.obs["Drug1"])
adata.obs["Drug2_numeric"] = enc.transform(adata.obs["Drug2"])

combinations = np.unique(adata1.obs[['Drug1_numeric','Drug2_numeric']], axis=0)

for num, comb in enumerate(combinations):
    comb = [unique_treatments[comb[0]],unique_treatments[comb[1]]]
    tmp = np.logical_and(adata.obs['Drug1'] == comb[0], adata.obs['Drug2'] == comb[1])
    pert_categories[tmp] = comb[0]+"_"+comb[1]
    if comb[0]+"_"+comb[1] in ho_combinations:
        ood += tmp
        print(f"Added {comb[0]}_{comb[1]} to ood set")
    print("{} with {} cells".format(comb,np.sum(tmp)))


adata.obs["ood"] = ood == 1
adata.obs["pert_categories"] = pert_categories


adata.obs["categories"] = adata.obs["pert_categories"].astype("category")
rank_genes_groups(adata, groupby="categories")

###Summary statistics:
print(adata.obs[["Drug1","Drug2"]].value_counts())
print("Num cells: ", adata.shape)

### Save data
sc.write('/mnt/data/Combosciplex.h5ad', adata)


