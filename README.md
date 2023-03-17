# CRUMB
Repository for the CRUMB dataset of FR galaxies

CRUMB (Collected Radiogalaxies Using MiraBest) is a machine learning image dataset of Fanaroff-Riley galaxies creating by merging [MiraBest](https://github.com/fmporter/MiraBest-full), [FR-DEEP](https://arxiv.org/abs/1903.11921), [AT17](https://arxiv.org/abs/1705.03413) and the supplementary MiraBest Hybrid dataset. All sources are cross-matched to ensure no duplicates images are present.

CRUMB consists of a total of 2104 images split into a training set of 1841 and a test set of 263. Images have two labels: a "basic" label - 0 (FRI), 1 (FRII) or 2 (hybid) - loaded as the dataset's default target, and a "complete" label which allows the user to select the class label from any of four the parent datasets. Please see the notebook for information on how to access these.
