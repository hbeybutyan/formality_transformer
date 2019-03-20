# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

hashs = {}


def assert_equal(x, y):
    assert x.size() == y.size()
    assert (x.data - y.data).abs().sum() == 0


def hash_data(x):
    """
    Compute a hash on tensor data.
    """
    # TODO: make a better hash function (although this is good enough for embeddings)
    return (x.data.sum(), x.data.abs().sum())


def test_sharing(encoder, decoder, params):
    """
    Test parameters sharing between the encoder,
    the decoder, and the language model.
    Test that frozen parameters are not being updated.
    """
    # frozen parameters
    if params.freeze_enc_emb:
        for i in range(params.n_langs):
            k = 'enc_emb_%i' % i
            if k in hashs:
                assert hash_data(encoder.embeddings[i].weight) == hashs[k]
            else:
                hashs[k] = hash_data(encoder.embeddings[i].weight)
    if params.freeze_dec_emb:
        for i in range(params.n_langs):
            k = 'dec_emb_%i' % i
            if k in hashs:
                assert hash_data(decoder.embeddings[i].weight) == hashs[k]
            else:
                hashs[k] = hash_data(decoder.embeddings[i].weight)

    #
    # encoder
    #
    # embedding layers
    if params.share_lang_emb:
        for i in range(1, params.n_langs):
            assert_equal(encoder.embeddings[i].weight, encoder.embeddings[0].weight)

    #
    # decoder
    #
    # embedding layers
    if params.share_encdec_emb:
        for i in range(params.n_langs):
            assert_equal(encoder.embeddings[i].weight, decoder.embeddings[i].weight)
    elif params.share_lang_emb:
        for i in range(1, params.n_langs):
            assert_equal(decoder.embeddings[i].weight, decoder.embeddings[0].weight)
    # projection layers
    if params.share_decpro_emb:
        for i in range(params.n_langs):
            assert_equal(decoder.proj[i].weight, decoder.embeddings[i].weight)
        if params.share_lang_emb:
            assert params.share_output_emb
            for i in range(1, params.n_langs):
                assert_equal(decoder.proj[i].bias, decoder.proj[0].bias)
    elif params.share_output_emb:
        assert params.share_lang_emb
        for i in range(1, params.n_langs):
            assert_equal(decoder.proj[i].weight, decoder.proj[0].weight)
            assert_equal(decoder.proj[i].bias, decoder.proj[0].bias)