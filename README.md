<h1> G-MSM: Unsupervised Multi-Shape Matching with Graph-based Affinity Priors </h1>


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