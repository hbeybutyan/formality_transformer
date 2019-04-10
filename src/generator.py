# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger

from .utils import restore_segmentation


logger = getLogger()


class Generator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.data = data
        self.dico = data['dico']
        self.params = params

    def get_iterator(self, lang1, lang2):
        """
        Create a new iterator for a dataset.
        """
        dataset = self.data['gen']
        dataset.batch_size = 32
        for batch in dataset.get_iterator(shuffle=False, group_by_size=False)():
            yield batch if lang1 < lang2 else batch[::-1]

    def generate(self, lang1, lang2):
        """
        Transform style1 -> style2 .
        """
        logger.info("Generating sentences %s -> %s ..." % (lang1, lang2))
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # hypothesis
        txt = []
        for batch in self.get_iterator(lang1, lang2):

            # batch
            sent1, len1 = batch
            if params.cuda:
                sent1 = sent1.cuda()

            # encode / generate
            encoded = self.encoder(sent1, len1, lang1_id)
            sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)
            # convert to text
            txt.extend(convert_to_text(sent2_, len2_, self.dico[lang2], lang2_id, self.params))

        # hypothesis / reference paths
        hyp_name = 'gen.{0}-{1}.txt'.format(lang1, lang2)
        hyp_path = os.path.join(params.dump_path, hyp_name)

        # export sentences to hypothesis file / restore BPE segmentation
        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt) + '\n')
        restore_segmentation(hyp_path)


def convert_to_text(batch, lengths, dico, lang_id, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()
    bos_index = params.bos_index[lang_id]

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == bos_index).sum() == bs
    assert (batch == params.eos_index).sum() == bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences
