# CODEX: COunterfactual Deep learning for the in-silico EXploration of cancer cell line perturbations


This repository implements CODEX, a general framework for the causal modelling of HTS data, linking perturbations to their downstream consequences. CODEX relies on a stringent causal modeling framework based on counterfactual reasoning. As such, CODEX predicts drug-specific cellular responses, comprising cell survival and molecular alterations, and facilitates the in-silico exploration of drug combinations.


## Setup

You can install CODEX directly via pip using the provided requirements file. Alternatively, you can use the provided `DOCKERFILE` to build a docker container. CODEX was tested with PyTorch 2.0.1 using CUDA 11.7.
To install the main requirements run
```
docker build -t codex -f DOCKERFILE .
```
If you want to run the gene perturbation experiments you need to install aditional requirements (uncomment in the Dockerfile), download GEARS (https://github.com/snap-stanford/GEARS) and add the package to the main path. 


## Run examples
To reproduce the results of the paper and run the provided examples, additional data download is required (see detailed explanation in `data/README.md`).

All parameters of the model can be set using flag commands. The model parameter flags are shared between all experiments, however, some experiments have additional flags corresponding to different experimental settings.
Example calls are included in each of main files.
Each run will save the log file, containing the train, validation, and test scores and the model corresponding to the best validation score in a unique folder named by the used hyper-parameters and the defined `--save_folder` in `models/`.
For final model selection, iterate over all log files corresponding to one experiment and load the model corresponding to the minimal validation score. 

## References
Stefan Schrod, Helena U Zacharias, Tim Beißbarth, Anne-Christin Hauschild, Michael Altenbuchinger, CODEX: COunterfactual Deep learning for the in silico EXploration of cancer cell line perturbations, Bioinformatics, Volume 40, Issue Supplement_1, July 2024, Pages i91–i99, https://doi.org/10.1093/bioinformatics/btae261


