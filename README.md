# UAD-IMAG

This repository contains the code for the [Deep Unsupervised Anomaly Detection in Brain Imaging:
Large-Scale Benchmarking and Bias Analysis](https://arxiv.org/pdf/2512.01534) publication. Deep unsupervised anomaly detection in brain magnetic resonance imaging offers a promising route to identify pathological deviations without requiring lesion-specific annotations. Yet, fragmented evaluations, heterogeneous datasets, and inconsistent metrics have hindered progress toward clinical translation. Here, we present a large-scale, multi-center benchmark of deep unsupervised anomaly detection for brain imaging.
The training cohort comprised 2,976 T1 and 2,972 T2-weighted scans (≈ 461,000 slices) from healthy individuals across six scanners, with ages ranging from 6 to 89 years. Validation used 92 scans to tune hyperparameters and estimate unbiased thresholds. Testing encompassed 2,221 T1w and 1,262 T2w scans spanning healthy datasets and diverse clinical cohorts. 

&nbsp;

<p align="center">
  <img src=https://github.com/AlexanderFrotscher/UAD-IMAG/blob/main/results.svg />
</p>

&nbsp;


Across all algorithms, the Dice-based segmentation performance ranged between ≈ 0.03 and ≈ 0.65, indicating substantial variability and underscoring that no single method achieved consistent superiority across lesion types or modalities for any task. To assess robustness, we systematically evaluated the impact of scanner variability, lesion type and size, and demographics (age, sex).
*Reconstruction-based* methods, particularly diffusion-inspired approaches, achieved the strongest lesion segmentation performance, while *feature-based* methods showed greater robustness under distributional shifts. However, systematic biases, such as lesion size, amplified scanner-related effects, while small and low-contrast lesions were missed more often, and false positives varied with age and sex.
Increasing healthy training data yielded only modest gains, underscoring the structural limitations of current unsupervised anomaly detection frameworks.


## File Structure

To use this repository your datasets have to be in the [BIDS](https://bids.neuroimaging.io/index.html) format:

```
 ├── Dataset
    │   ├── sub-ID
    │   │   ├── anat
    │   │   │   ├── sub-ID_T1w.nii
    │   │   │   ├── sub-ID_T2w.nii
    │   │   │   └── ...

```
Additionally multiple .csv files are needed that specify which individuals belong to train, test or validation split. Each .csv needs to contain the full path to the .nii file.

```
Path
/home/user/dataset/sub-ID/anat/sub-ID_T1w.nii.gz
/home/user/dataset/sub-ID2/anat/sub-ID2_T1w.nii.gz
...
```

## Preprocessing
For preprocessing the datasets, we applied a [pipeline](https://github.com/AlexanderFrotscher/UKB-MRI-Preprocessing) similar to the UK Biobank protocol. For training the methods, you need to generate a Lightning Memory-Mapped Database of the preprocessed slices with the [create_LMDB.py](utils/create_LMDB.py) script. Use the .csv that contains the paths to all files used during training.

```
python utils/create_LMDB.py -d path_to_csv.csv -p "path_to_store_lmdb"
```

## Training
The configuration for the training run can be set in the respective [config file](conf/ANDi_config.py). You need to specify the path to the created LMDB and the hyperparameters you want to use. Before starting the training, you will need to setup wandb and enter your account in the respective training script. Then the training for one model can be started with for example:

```
accelerate launch models/ANDi/train_ANDi.py
```

## Evaluation
For evaluation you need to set the path to the .csv that contains all volumes for evaluation in the respective [config file](conf/ANDi_eval.py). 
Then, run the evaluation for one algorithm, for example [eval_ANDi.py](models/ANDi/eval_ANDi.py):

```
accelerate launch models/ANDi/eval_ANDi.py
```
