# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

from ax.service.managed_loop import optimize

# Set the random seed manually for reproducibility.

use_cuda = True
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
device = torch.device("cuda" if use_cuda else "cpu")
###############################################################################
# Load data
###############################################################################
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


corpus = data.Corpus('data/gigaspeech')
train_data = batchify(corpus.train, 20)
val_data = batchify(corpus.valid, 10)
test_data = batchify(corpus.test, 10)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.



###############################################################################
# Build the model
###############################################################################



###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(35, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target



def train_evaluate(parameterization):
# Loop over epochs.
    model_name = 'GRU'
    torch.manual_seed(1111)
    save_path = 'model.pt'


    eval_batch_size = 10


    nhead = 16
    emsize = 512
    nhid = 512
    tied= True
    nlayers = parameterization.get("nlayers", 2)
    dropout = parameterization.get("dropout", 0.2)
    clip = parameterization.get("clip", 0.25)
    lr = parameterization.get("lr", 20)
    best_val_loss = None
    print('nlayers : %d ;dropout : %f ;clip : %f ; lr : %f .' %(nlayers,dropout,clip,lr))
    ntokens = len(corpus.dictionary)
    if model_name == 'Transformer':
        net = model.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    else:
        net = model.RNNModel(model_name, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)

    print ("Vocabulary Size: ", ntokens)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print ("Total number of model parameters: {:.2f}M".format(num_params*1.0/1e6))


    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, 7):
            epoch_start_time = time.time()

            dry_run = True
            criterion = nn.NLLLoss()
            net.train()
            total_loss = 0.
            start_time = time.time()
            ntokens = len(corpus.dictionary)
            if model_name != 'Transformer':
                hidden = net.init_hidden(20)
            for batch, i in enumerate(range(0, train_data.size(0) - 1, 35)):
                data, targets = get_batch(train_data, i)
                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                net.zero_grad()
                if model_name == 'Transformer':
                    output = net(data)
                    output = output.view(-1, ntokens)
                else:
                    hidden = repackage_hidden(hidden)
                    output, hidden = net(data, hidden)
                loss = criterion(output, targets)
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                for p in net.parameters():
                    p.data.add_(p.grad, alpha=-lr)

                total_loss += loss.item()

                if batch % 500 == 0 and batch > 0:
                    cur_loss = total_loss / 500
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                          'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // 35, lr,
                                      elapsed * 1000 / 500, cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()
               # if dry_run:
               #     break

            criterion = nn.NLLLoss()
            net.eval()
            total_loss = 0.
            ntokens = len(corpus.dictionary)
            if model_name != 'Transformer':
                hidden = net.init_hidden(eval_batch_size)
            with torch.no_grad():
                for i in range(0, val_data.size(0) - 1, 35):
                    data, targets = get_batch(val_data, i)
                    if model_name == 'Transformer':
                        output = net(data)
                        output = output.view(-1, ntokens)
                    else:
                        output, hidden = net(data, hidden)
                        hidden = repackage_hidden(hidden)
                    total_loss += len(data) * criterion(output, targets).item()
            val_loss = total_loss / (len(val_data) - 1)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(save_path, 'wb') as f:
                    torch.save(net, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(save_path, 'rb') as f:
        net = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if model_name in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            net.rnn.flatten_parameters()

    # Run on test data.
    criterion = nn.NLLLoss()
    net.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if model_name != 'Transformer':
        hidden = net.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, test_data.size(0) - 1, 35):
            data, targets = get_batch(test_data, i)
            if model_name == 'Transformer':
                output = net(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = net(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    test_loss = total_loss / (len(test_data) - 1)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    return math.exp(test_loss)


best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1, 50], "value_type":"float", "log_scale": True},
        {"name": "clip", "type": "range", "bounds": [0, 0.6], "value_type":"float"},
        {"name": "dropout", "type": "range", "bounds": [0, 0.4], "value_type":"float"},
        {"name": "nlayers", "type": "choice", "values": [2, 3, 4, 6, 8]},
    ],
    evaluation_function=train_evaluate,
    objective_name='test ppl',
    minimize = True
)
means, covariances = values
means, covariances
print(means, covariances)

print(best_parameters)

#if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
#    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
