<h1> G-MSM: Unsupervised Multi-Shape Matching with Graph-based Affinity Priors </h1>

The official implementation of the CVPR'2023 paper (\[[arXiv]https://arxiv.org/abs/2212.02910)\].

## Usage

### Setting up the repo
* We recommend installing the repo directly via anaconda:
```bash
conda env create --name gmsm_env -f gmsm_env.yml
conda activate gmsm_env
```

### Data
* The repository contains sample code to train our full model on SHREC'20. Please download the high-resolution data from [here](http://robertodyke.com/shrec2020/index2.html) and preprocess it, following the steps from the DeepShells [repository](https://github.com/marvin-eisenberger/deep-shells).
* Save the processed shapes under `data/shrec20`, or specify the path in `utils/data.py`.
* Other datasets can be trained analogously, following the same preprocessing steps. In our experiments, we considered the nearly-isometric datasets [FAUST remeshed](https://drive.google.com/file/d/1C-9GFsTl5xwa0RUmC_m1nnj87QUguh6j/view), [SCAPE remeshed](https://drive.google.com/file/d/157SoRhiVQzsWbSFlaV5N-vzkxKCvTIlf/view), [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/), [SHREC'19](http://profs.scienze.univr.it/~marin/shrec19/#Download), the topological datasets [SMAL](https://smal.is.tue.mpg.de/download.php), [TOPKIDS](https://cvg.cit.tum.de/data/datasets/topkids), [SHRECISO](https://shrec19.cs.cf.ac.uk/) as well as the non-isometric datasets [SHREC'20](http://robertodyke.com/shrec2020/index2.html), [SMAL](https://smal.is.tue.mpg.de/download.php), [TOSCA](http://tosca.cs.technion.ac.il/book/resources_data.html). For each dataset, running the [script](https://github.com/marvin-eisenberger/deep-shells/blob/master/preprocess_data/preprocess_dataset.m) `preprocess_data/preprocess_dataset.m` performs all the necessary preprocessing steps.
### Train
* To train our model, simply run
```bash
python3 main.py
```
* The outputs will be saved automatically in a new folder under `results/shrec20`.

### Test
* To test our model, run the evaluation script
```bash
python3 eval_scripts.py
```
* Results will be saved under `results/shrec20_pretrained`.
* The script outputs the query correspondences for all pairs in individual files, as well as the predicted shape graph and the validation losses for all pairs.
* By default, a pretrained version of our model is used to produce the query correspondences on SHREC'20. 