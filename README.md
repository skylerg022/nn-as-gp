# Neural Nets as Gaussian Processes

## Introduction

This repository contains the code used for the research in my Master's project, **Substituting Neural Networks for Gaussian Processes** (see the project report [here](https://github.com/skylerg022/nn-as-gp/blob/main/masters_writeup_4_15.pdf)), as well as additional research work following the final writeup submission on April 15, 2022. This research work is evangelical in nature, preaching the gospel of neural networks to the statistical community as a viable option for predictive modeling in lieu of intractable Gaussian process modeling for big data.

The purpose of sharing this code is not to provide polished new functions to the community but rather to facilitate replicability of research results. Given the repository structure, you may extend this research to some new dataset with relative ease.

## Directory Structure



### Data

The land surface temperature data (Quant150K and Quant150K) should be available in this report by the end of April 2022.

The quantitative 1-million and 90K datasets come from the "Competition on Spatial Statistics for Large Datasets" by Huang et al. (2021) and are available at [https://doi.org/10.25781/KAUST-8VP2V](https://doi.org/10.25781/KAUST-8VP2V). After downloading the files, create a `data` directory in the respective project folders (for example, the directory `Quant1Mil_G5/data`) and move the data files to their respective `data` directories.

#### Finished Analyses

- Quant1Mil_G5
- Quant1Mil_NG1
- Quant150K_Sim

#### Unfinished Analyses

- Binary1Mil
- Quant150K
- Quant90K

