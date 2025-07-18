[![language](https://img.shields.io/badge/language-Python-3776AB)](https://www.python.org/)
[![OS](https://img.shields.io/badge/OS-CentOS%20%7C%20Ubuntu-2C3E50)](https://www.centos.org/)
[![arch](https://img.shields.io/badge/arch-x86__64-blue)](https://en.wikipedia.org/wiki/X86-64)
[![GitHub last commit](https://img.shields.io/github/last-commit/zhichunlizzx/MambaHM)](https://github.com/zhichunlizzx/MambaHM/commits)

# MambaHM
This package provides an implementation for training, testing, and evaluation of the MambaHM framework.
![Hi](https://github.com/zhichunlizzx/MambaHM/blob/master/model.png?v=4&s=200 "dREG gateway")

## 🚀 About
**MambaHM** is a model specifically designed for histone modification signal prediction. It leverages raw ATAC-seq data and one-hot encoded DNA sequences as input features. The key advantage of MambaHM lies in its use of the computationally efficient Mamba architecture, which enables high-resolution predictions at 16 bp while maintaining strong predictive performance.


## 🔧 Setup
Requirements:
*   mamba-ssm(2.2.2)
*   torch(2.5.1)
*   h5py(3.12.1)
*   pyBigWig(0.3.24)
*   numpy(1.26.4)
*   tensorflow(2.4.0)

See `MambaHM.yml`.

Create the environment with the following command:

```shell
conda env create -f MambaHM.yml -n my_env
```

Pre-trained model weights of MambaHM are available here: https://github.com/zhichunlizzx/MambaHM/blob/master/model/model.pth.

## 📝How to train and predict
The training of the model requires the following types of data:
*   ATAC-seq data ("xx.bw, xx_minus.bw", optional — at least one of ATAC-seq or reference genome data must be provided)
*   Reference genome data ("hg19.fa", optional — at least one of reference genome or ATAC-seq data must be provided)
*   Target ground truth (10 types of histone modification ChIP-seq data)
*   Genome blacklist (optional)

When using a trained model for prediction, it is not necessary to provide the target ground truth. The detailed process of training can be found in `train.ipynb`.

## 📊 Evaluation
This package provides evaluation and prediction methods for four subtasks of MambaHM, see detail in `evaluation_and_predict.ipynb`.

## 🧬 Targets of downstream tasks
|index|Cell Type|Item|Type|
|:-|:-|:-|:-|
|1|K562|H3K4me1, H3K122ac, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3, H4K20me1|ChIP-seq|
|2|GM12878|H3K4me1, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3, H4K20me1|ChIP-seq|
|3|HCT116|H3K4me1, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3, H4K20me1|ChIP-seq|
|4|HeLa-S3|H3K4me1, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3, H4K20me1|ChIP-seq|
|5|CD4|H3K4me1, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3|ChIP-seq|
|6|IMR-90|H3K4me1, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3, H4K20me1|ChIP-seq|
|7|MCF-7|H3K4me1, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3, H4K20me1|ChIP-seq|
|8|HepG2|H3K4me1, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3, H4K20me1|ChIP-seq|
|9|A549|H3K4me1, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3, H4K20me1|ChIP-seq|
|10|Heart|H3K4me1, H3K4me3, H3K27me3, H3K36me3, H3K9ac, H3K9me3|ChIP-seq|
|11|Spleen|H3K4me1, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9me3|ChIP-seq|
|12|Heart mm10|H3K4me1, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3|ChIP-seq|
|13|HindBrain mm10|H3K4me1, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3|ChIP-seq|
|14|G1E mm10|H3K4me1, H3K4me3, H3K27me3, H3K36me3, H3K9me3|ChIP-seq|







