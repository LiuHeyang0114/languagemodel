## Language Model Project

### Best performance achieved without changing the model structure
You can reproduce the experiment by running run.sh

|structure|nhid|emsize|nlayers|lr|clip|dropout|nhead|parameters(M)|Valid ppl|Test ppl|
|---|---|---|---|---|---|---|---|---|---|---|
RNN(Relu) | 512 | 512 | 2 | 1 | 0.6 | 0.1| |29.45 | 162.34 | 173.66|
RNN(Tanh) | 512 | 512 | 4 |1  | 0.4 | 0.1 | |30.50 | 184.31 | 195.25|
LSTM | 512 | 512 | 2 | 20 | 0.1 | 0.2 | |32.61 | 129.57 | 138.87|
GRU | 512 | 512 | 2 | 10 | 0.4 | 0.2| |31.55 | 132.53 | 141.19 |
Transformer | 512 | 512 | 6 | 1 | 0.25 | 0.1 | 16 |37.87 | 152.79 | 165.10|

### Training with Bayesian Optimization

We use the GRU model as an example and provide training code that includes Bayesian optimization. You can run python main_bo.py directly to complete training and testing. If you want to use it on other models, please change the model_name in the train_evaluate function and specify the approximate range of tuning parameters in the parameters at the end of the code.

|structure|lr range|lr|clip|dropout|nlayers|ppl|
|---|---|---|---|---|---|---|
RNN(Relu) | [0.1, 2] | 0.970 | 0.472 | 0.079 | 3  | 176.27 |
RNN(Tanh) | [1,10] | 1.452 | 0.486 | 0.221 | 2 | 193.234 |
LSTM | [1,50] | 20.978 | 0.138 | 0.084 | 3 | 136.948 |
GRU | [1,50] | 8.530 | 0.426 | 0.253 | 2 | 140.227 |
Transformer | [0.1,5] | 1.806 | 0.425 |  0.102 | 8  | 163.445 |