# DualFair

This repository is the official implementation for "DualFair: Fair Representation Learning at Both Group and Individual Levels via Contrastive Self-supervision".

### Highlight
- **Accepted at WWW 2023.**
- We propose a self-supervised representation learning framework (DualFair) that simultaneously debiases sensitive attributes at both group and individual levels.
- We introduce the C-VAE model to generate counterfactual samples and propose *fairness-aware contrastive loss* to meet the two fairness criteria jointly.
- We design the *self-knowledge distillation loss* to maintain representation quality by minimizing the embedding discrepancy between original and perturbed instances.
- Experimental results on six real-world datasets confirm that DualFair generates a fair embedding for sensitive attributes while maintaining high representation quality. The ablation study further shows a synergistic effect of the two fairness criteria.


### Data
The following datasets are supported in this repository for evaluation:  
(1) UCI Adult contains 48,842 samples along with label information that indicates whether a given individual makes over 50K per year as a downstream task  
Link: https://archive.ics.uci.edu/ml/datasets/adult  

(2) UCI German credit includes 1,000 samples with 20 attributes and aims to predict credit approvals  
Link: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)  

(3) COMPAS includes 6,172 samples and is used to predict recidivism risk (i.e., the risk of a criminal defendant committing a crime again) of given individuals  
Link: https://www.kaggle.com/danofer/compass  

(4) LSAC contains 22,407 samples and estimates whether a given individual will pass the law school admission. We split each dataset into disjoint training and test sets. Embedding learning and counterfactual sample generation are based on the knowledge of the training set; evaluations are based on the test set.  
Link: http://www.seaphe.org/databases.php  


### Required packages
The code has been tested running under Python 3.7.3. with the following packages installed (along with their dependencies):

- numpy == 1.18.1
- pandas == 1.1.4
- torch == 1.2.0
- scikit-learn == 0.22.1
- rdt == 0.5.3
- tqdm == 4.44.1

<p>We recommend using the open data science platform <a href="https://www.continuum.io/downloads" rel="nofollow">Anaconda</a>.</p>


## Training
Training the counterfactual sample generator model with the following command.  
The trained sample generator will be saved in *./output/converter* directory.
```
python3 converter.py --dataset <DATASET> --sensitive <SENSITIVE_ATTRIBUTE> --gpu <GPU_ID>
```

Train DualFair model based on the trained counterfactual sample generator with the following command.  
Code automatically detects and loads the trained sample generator for training.  
The trained model will be saved in *./output/dualfair* directory.
```
python3 dualfair.py --dataset <DATASET> --sensitive <SENSITIVE_ATTRIBUTE> --gpu <GPU_ID>
```


## Evaluation
You can evaluate the trained model with the following command.  
Please provide a file name of the saved model checkpoint in ./output/dualfair.  
File name will have a following format - "dualfair_<DATASET>_<SENSITIVE_ATTRIBUTE>_seed_<SEED_NUM>_<TIME>"  
```
python3 evaluate.py --save_pre <FILE_NAME> --dataset <DATASET> --sensitive <SENSITIVE_ATTRIBUTE>
```
