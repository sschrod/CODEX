To run the experiments and reproduce the results of the paper additional data download is required.

### Drug-synergy experiments
To run the drug synergy_experiments clone the MARSY [1] GitHub repository https://github.com/Emad-COMBINE-lab/MARSY/tree/main, create a folder `./MARSY` and copy all files from `MARSY/data`.

### Dose experiment
To run the dose experiment download the preprocessed `GSM_new.h5ad` from 
https://drive.google.com/drive/folders/1yFB0gBr72_KLLp1asojxTgTqgz6cwpju (hosted by the authors of CPA [2]) and add the files to `data/CPA_datasets`

### Combosciplex experiment
To run the Combosciplex experiment download the raw data from https://0-www-ncbi-nlm-nih-gov.brum.beds.ac.uk/geo/query/acc.cgi?acc=GSE206741  GEO accession code — GSE206741, save to `data/Combosciplex/`
and run `Preprocess_Combosciplex.py` to preprocess the data.


### Gene-perturbation experiemnts
The gene-perturbation experiments download the data automatically from the dataverse hosted by the authors of GEARS [3].
Additionally, installation of GEARS is required (https://github.com/snap-stanford/GEARS) since CODEX uses the 
`PertData()` class for data loading. Then simply add `--downlad_data` the first time you run `CODEX_Norman19.py` and `CODEX_Replogle.py` to download and preprocess the data.


### References
[1] El Khili, Mohamed Reda, Safyan Aman Memon, and Amin Emad. "MARSY: a multitask deep-learning framework for prediction of drug combination synergy scores." Bioinformatics 39.4 (2023): btad177

[2] Lotfollahi, Mohammad, et al. "Predicting cellular responses to complex perturbations in high‐throughput screens." Molecular Systems Biology (2023): e11517.

[3] Roohani, Yusuf, Kexin Huang, and Jure Leskovec. "Predicting transcriptional outcomes of novel multigene perturbations with gears." Nature Biotechnology (2023): 1-9.


