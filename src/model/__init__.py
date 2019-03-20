# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from collections import namedtuple


LSTM_PARAMS = ['weight_ih_l%i', 'weight_hh_l%i', 'bias_ih_l%i', 'bias_hh_l%i']
BILSTM_PARAMS = LSTM_PARAMS + ['%s_reverse' % x for x in LSTM_PARAMS]

LatentState = namedtuple('LatentState', 'dec_input, dis_input, input_len')


def check_mt_model_params(params):
    """
    Check models parameters.
    """

    # shared layers
    assert 0 <= params.dropout < 1
    assert 0 <= params.share_dec <= params.n_dec_layers
    assert params.emb_dim == params.hidden_dim
    assert not params.share_output_emb or params.share_lang_emb
    assert (not (params.share_decpro_emb and params.share_lang_emb)) or params.share_output_emb

    # attention model
    assert params.n_dec_layers > 1
    assert params.emb_dim % params.encoder_attention_heads == 0
    assert params.emb_dim % params.decoder_attention_heads == 0

    # pretrained embeddings / freeze embeddings
    if params.pretrained_emb == '':
        assert not params.freeze_enc_emb or params.reload_enc
        assert not params.freeze_dec_emb or params.reload_dec
        assert not params.pretrained_out
    else:
        split = params.pretrained_emb.split(',')
        if len(split) == 1:
            assert os.path.isfile(params.pretrained_emb)
        else:
            assert len(split) == params.n_langs
            assert not params.share_lang_emb
            assert all(os.path.isfile(x) for x in split)
        if params.share_encdec_emb:
            assert params.freeze_enc_emb == params.freeze_dec_emb
        else:
            assert not (params.freeze_enc_emb and params.freeze_dec_emb)
        assert not (params.share_decpro_emb and params.freeze_dec_emb)
        assert not (params.share_decpro_emb and not params.pretrained_out)
        assert not params.pretrained_out or params.emb_dim == params.hidden_dim

    # discriminator parameters
    assert params.dis_layers >= 0
    assert params.dis_hidden_dim > 0
    assert 0 <= params.dis_dropout < 1
    assert params.dis_clip >= 0

    # reload MT model
    assert params.reload_model == '' or os.path.isfile(params.reload_model)
    assert not (params.reload_model != '') ^ (params.reload_enc or params.reload_dec or params.reload_dis)


def build_mt_model(params, data):
    """
    Build machine translation model.
    """
    from .attention import build_attention_model
    return build_attention_model(params, data, cuda=params.cuda)