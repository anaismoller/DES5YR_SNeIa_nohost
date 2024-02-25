# The Dark Energy Survey 5-year photometrically classified type Ia supernovae without host-galaxy redshifts

Code to reproduce analysis in Möller et al. 2024. We select SNe Ia using only their light-curves from the [Dark Energy Survey](https://www.darkenergysurvey.org) 5-year data using the framework [SuperNNova](https://github.com/supernnova/SuperNNova) [(Möller & de Boissière 2019)](https://academic.oup.com/mnras/article-abstract/491/3/4277/5651173). 

As most of these SNe do not have a host-galaxy redshift, we derive redshifts directly from SN light-curves. We obtain an almost-complete sample of high-quality SNe Ia in DES SN fields which is in average higher redshift, bluer and broader light-curves, and fainter host-galaxies than other DES SNe Ia samples.

Using the DES 5-year data we also explore optimisation of follow-up decisions for Rubin LSST for both SNe Ia host-galaxies and live SNe Ia (early classification before maximum light).

This repository contains:
- *.py: analysis reproduction codes (prints outputs quoted in paper, saves plots and samples)

Simulations used for this project can be found in the [Möller et al. 2022](https://github.com/anaismoller/DES5YR_SNeIa_hostz/) repository under
- /reproduce/Pippin* : configuration files to recreate simulations used in the analysis using [pippin](https://github.com/dessn/Pippin) (Hinton & Brout 2020) analysis pipeline
