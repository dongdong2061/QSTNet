# Quality-aware Spatio-temporal Transformer Network for RGBT Tracking

### Installation
Create and activate a conda environment:
```
conda create -n qstnet python=3.7
conda activate qstnet
```
Install the required packages:
```
bash install_qstnet.sh
```

### Data Preparation
Download the training datasets, It should look like:
```
$<PATH_of_Datasets>
    -- LasHeR/TrainingSet
        |-- 1boygo
        |-- 1handsth
        ...
```

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_BAT>
python tracking/create_default_local_file.py --workspace_dir . --data_dir <PATH_of_Datasets> --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Dowmload the pretrained [foundation model]([https://pan.baidu.com/s/1JX7xUlr-XutcsDsOeATU1A?pwd=4lvo](https://www.kaggle.com/datasets/zhaodongding/drgbt603-results/data?select=pretrained)) (OSTrack and DropTrack)
and put it under ./pretrained/.
```
bash train_qstnet.sh
```
You can train models with various modalities and variants by modifying ```train_qstnet.sh```.

### Testing

#### For RGB-T benchmarks
[LasHeR & RGBT234] \
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash eval_rgbt.sh
```
We refer you to use [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation, 
and refer you to use [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 evaluation.


## Acknowledgment
- This repo is based on [BAT](https://github.com/SparkTempest/BAT) which is an exellent work, helps us to quickly implement our ideas.
- Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library.

