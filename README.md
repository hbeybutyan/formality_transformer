# Text Formality transfer

The repository conatains implementation of transformer based model for text formality transfer
The model consists of 3 main parts:
- Transformer
- Autoencoders
- Language discriminator

## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/) (currently tested on version 0.5)
* [Moses](http://www.statmt.org/moses/) (clean and tokenize text / train PBSMT model)
* [fastBPE](https://github.com/glample/fastBPE) (generate and apply BPE codes)
* [fastText](https://github.com/facebookresearch/fastText) (generate embeddings)

## Download / preprocess data

The `get_fin_data.sh` script will take care of installing everything (except PyTorch) and preprocessing the data.

The script downloads a monolingual and parallel sentences which are obtained from [L6 -Yahoo! Answers Comprehensive Questions and Answers version 1.0 ](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l) and [GYAFC corpus] (https://github.com/raosudha89/GYAFC-corpus). Please follow the instructions in the corresponsing links to get the datasets.

After datasetas are obtained you need to classify the sentences in the L6 corpus to formal/informal categories, join the sets of GYAFC corpus and make those available for download from your machine (e.g. upload those to Google drive and get links). We plan to describe all the required steps in near future.


`get_enfr_data.sh` contains a few parameters defined at the beginning of the file:
- `N_MONO` number of monolingual sentences for each language
- `CODES` number of BPE codes (default 60000)
- `N_THREADS` number of threads in data preprocessing (default 48)
- `N_EPOCHS` number of fastText epochs (default 10)

## Train the model

Use the following command with corresponding parameters to train the model:

```
python main.py 

## main parameters
--exp_name test                             # experiment name
--exp_id test_ud                            # experiment id
--dump_path ./dumped/                       # experiment dump path
--save_periodic False                       # save the model periodically

## Model parameters
--emb_dim 512                               # embedding layer size
--n_enc_layers 4                            # use 4 layers in the encoder
--n_dec_layers 4                            # use 4 layers in the decoder
--hidden_dim 512                            # hidden layer size
--dropout 0                                 # dropout
--label-smoothing 0                         # Label smoothing
--attention_dropout 0                       # attention dropout
--relu_dropout 0                            # relu dropout
--encoder_attention_heads 8                 # encoder attention heads
--decoder_attention_heads 8                 # decoder attention heads
--encoder_normilize_before False            # encoder normilize before
--decoder_normilize_before False            # decoder normilize before
## parameters sharing
--share_enc 3                               # share 3 out of the 4 encoder layers
--share_dec 3                               # share 3 out of the 4 decoder layers
--share_lang_emb True                       # share lookup tables between different styles (enc/dec/proj)
--share_output_emb True                     # share projection output layers of different styles
--share_enc_dec_emb False                   # share encoder decoder embeddings
--share_decpro_emb False                    # share decoder embeddings with decoder output projection

## encoder input
--word_shuffle 0                            # randomly shuffle input words (0 to disable)
--word_dropout 0                            # randomly dropout input words (0 to disable)
--word_blank 0                              # randomly blank input words (0 to disable)

## discriminator parameters
--dis_layers" 3                             # number of hidden layers in the discriminator
--dis_hidden_dim 128                        # discriminator hidden layers dimension
--dis_dropout 0                             # discriminator dropout
--dis_clip 0                                # clip discriminator weights (0 to disable)
--dis_smooth 0                              # GAN smooth predictions
--dis_input_proj True                       # feed the discriminator with the projected output


## datasets parameters
--langs 'f,in'                              # training datasets (Formal, Informal)
--n_mono -1                                 # number of sentences in monostyle corpus to use (-1 for everything)
--n_para -1                                 # number of parallel sentences (-1 for everything)
--mono_dataset $MONO_DATASET                # monostyle dataset
--para_dataset $PARA_DATASET                # parallel dataset

--vocab ""                                  # vocabulary (style1:path1;style2:path2)
--vocab_min_count 0                         # vocabulary minimum word count
--max_len 175                               # maximum length of sentences (after BPE)
--max_vocab -1                              # maximum vocabulary size (-1 to disable)

## denoising auto-encoder parameters
--n_dis 0                                   # number of discriminator training iterations in each epoch
--para_directions ""                        # training directions (style1-style2,style2-style1)
--mono_directions 'f,in'                    # train the auto-encoder on Formal and Informal corpuses

## training parameters
--cuda False                                # run with cuda
--batch_size 32                             # batch size
--group_by_size True                        # sort sentences by size during the training
--lambda_xe_mono 0                          # cross-entropy reconstruction coefficient (autoencoding)
--lambda_xe_para 0                          # cross-entropy transformer coefficient (parallel data)
--lambda_dis 0                              # discriminator loss coefficient
--enc_optimizer "adam,lr=0.0003"            # encoder optimizer (SGD / RMSprop / Adam, etc.)
--dec_optimizer "enc_optimizer"             # decoder optimizer (SGD / RMSprop / Adam, etc.)
--dis_optimizer "rmsprop,lr=0.0005"         # discriminator optimizer (SGD / RMSprop / Adam, etc.)
--clip_grad_norm 5                          # clip gradients norm (0 to disable)
--epoch_size 100000                         # epoch size / evaluation frequency
--max_epoch 100000                          # maximum epoch size
--stopping_criterion ""                     # stopping criterion, and number of non-increase before stopping the experiment

## reload model
--pretrained_emb $PRETRAINED                # cross-lingual embeddings path
--pretrained_out True                       # also pretrain output layers
--model_file                                # if specified the file will be loaded instead of checkpoint

--reload_model ""                           # reload a pre-trained model
--reload_enc False                          # reload a pre-trained encoder
--reload_dec False                          # reload a pre-trained decoder
--reload_dis False                          # reload a pre-trained discriminator

## freeze network parameters
--freeze_enc_emb False                      # freeze encoder embeddings
--freeze_dec_emb False                      # freeze decoder embeddings

## evaluation 
--eval_only False                           # only run evaluations
--beam_size 0                               # beam width (<= 0 means greedy)
--length_penalty 1.0                        # length penalty: <1.0 favors shorter, >1.0 favors longer sentences

## generation
--generation_set ''                         # a set of preprocessed sentences which must be style transformed
--generation_source_style ''                # a style of sentences in generation set
--valid_domain ''                           # a comtext domain of generation set. This is used for output naming only

## Where
MONO_DATASET='f:./data/mono/all.f.tok.60000.pth,,;in:./data/mono/all.in.tok.60000.pth,,'
PARA_DATASET='f-in:./data/para/train.XX.60000.pth,./data/para/val.XX.60000.pth,./data/para/test.XX.60000.pth'
PRETRAINED='./data/mono/all.f-in.60000.vec'


```

With all this in mind a good candidate for a training command is:

```
python main.py --exp_name test --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs 'f,in' --n_mono -1 --n_para -1 --n_dis 1 --mono_dataset 'f:./data/mono/all.f.tok.60000.pth,,;in:./data/mono/all.in.tok.60000.pth,,' --para_dataset 'f-in:./data/para/train.XX.60000.pth,./data/para/val.XX.60000.pth,./data/para/test.XX.60000.pth' --mono_directions 'f,in' ----para_directions 'f-in' word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pretrained_emb './data/mono/all.f-in.60000.vec' --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_para 1 --lambda_dis 1 --enc_optimizer adam,lr=0.0001 --epoch_size 50000 --stopping_criterion bleu_f_in_valid,10
```

Here are used code samples from the 
[Phrase-Based & Neural Unsupervised Machine Translation](https://github.com/facebookresearch/UnsupervisedMT).


