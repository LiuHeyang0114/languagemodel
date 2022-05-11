### QRNN Implementation

#### We recommend that you create a conda environment to implement QRNN

Since pytorch-qrnn and the model code may have download or run errors due to version problems, we recommend downloading torch under a newer version and running it under an older version.
```
conda create -n qrnn python=3
conda activate qrnn
conda install pytorch==0.4
pip install cupy pynvrtc git+https://github.com/salesforce/pytorch-qrnn
conda install pytorch=0.1.12 -c soumith
```


#### Result


|structure|Parameters(M)|Valid loss|Valid ppl|Test ppl|
|---|---|---|---|---|
LSTM | 32.61 | 4.86 | 129.57 | 138.87|
GRU | 31.55 | 4.89 | 132.53 | 141.19|
QRNN | 30.90 | 4.94 | 136.69 | 148.29|


For more details about QRNN, please refer https://arxiv.org/abs/1611.01576.