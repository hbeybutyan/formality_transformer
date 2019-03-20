# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch
from torch import nn

from .discriminator import Discriminator
from ..modules.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyLoss
from .pretrain_embeddings import initialize_embeddings
from ..utils import reload_model


logger = getLogger()


def build_transformer_enc_dec(params):
    from .transformer import TransformerEncoder, TransformerDecoder

    params.left_pad_source = False
    params.left_pad_target = False

    assert hasattr(params, 'dropout')
    assert hasattr(params, 'attention_dropout')
    assert hasattr(params, 'relu_dropout')

    params.encoder_embed_dim = params.emb_dim
    params.encoder_ffn_embed_dim = params.transformer_ffn_emb_dim
    params.encoder_layers = params.n_enc_layers
    assert hasattr(params, 'encoder_attention_heads')
    assert hasattr(params, 'encoder_normalize_before')

    params.decoder_embed_dim = params.emb_dim
    params.decoder_ffn_embed_dim = params.transformer_ffn_emb_dim
    params.decoder_layers = params.n_dec_layers
    assert hasattr(params, 'decoder_attention_heads')
    assert hasattr(params, 'decoder_normalize_before')

    logger.info("============ Building transformer attention model - Encoder ...")
    encoder = TransformerEncoder(params)
    logger.info("")
    logger.info("============ Building transformer attention model - Decoder ...")
    decoder = TransformerDecoder(params, encoder)
    logger.info("")
    return encoder, decoder


def build_attention_model(params, data, cuda=True):
    """
    Build a encoder / decoder, and the decoder reconstruction loss function.
    """
    # encoder / decoder / discriminator\
    encoder, decoder = build_transformer_enc_dec(params)
    logger.info("============ Building attention model - Discriminator ...")
    discriminator = Discriminator(params)
    logger.info("")

    # loss function for decoder reconstruction
    loss_fn = []
    for n_words in params.n_words:
        loss_weight = torch.FloatTensor(n_words).fill_(1)
        loss_weight[params.pad_index] = 0
        if params.label_smoothing <= 0:
            loss_fn.append(nn.CrossEntropyLoss(loss_weight, size_average=True))
        else:
            loss_fn.append(LabelSmoothedCrossEntropyLoss(
                params.label_smoothing,
                params.pad_index,
                size_average=True,
                weight=loss_weight,
            ))
    decoder.loss_fn = nn.ModuleList(loss_fn)
    # language model

    # cuda - models on CPU will be synchronized and don't need to be reloaded
    if cuda:
        encoder.cuda()
        decoder.cuda()
        if len(params.vocab) > 0:
            decoder.vocab_mask_neg = [x.cuda() for x in decoder.vocab_mask_neg]
        if discriminator is not None:
            discriminator.cuda()

        # initialize the model with pretrained embeddings
        assert not (getattr(params, 'cpu_thread', False)) ^ (data is None)
        if data is not None:
            initialize_embeddings(encoder, decoder, params, data)

        # reload encoder / decoder / discriminator
        if params.reload_model != '':
            assert os.path.isfile(params.reload_model)
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model)
            if params.reload_enc:
                logger.info("Reloading encoder...")
                enc = reloaded.get('enc', reloaded.get('encoder'))
                reload_model(encoder, enc, encoder.ENC_ATTR)
            if params.reload_dec:
                logger.info("Reloading decoder...")
                dec = reloaded.get('dec', reloaded.get('decoder'))
                reload_model(decoder, dec, decoder.DEC_ATTR)
            if params.reload_dis:
                assert discriminator is not None
                logger.info("Reloading discriminator...")
                dis = reloaded.get('dis', reloaded.get('discriminator'))
                reload_model(discriminator, dis, discriminator.DIS_ATTR)
    return encoder, decoder, discriminator
