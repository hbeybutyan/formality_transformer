# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
from logging import getLogger
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .utils import clip_parameters
from .utils import get_optimizer, parse_lambda_config, update_lambdas
from .multiprocessing_event_loop import MultiprocessingEventLoop
from .test import test_sharing


logger = getLogger()


class TrainerMT(MultiprocessingEventLoop):

    VALIDATION_METRICS = []

    def __init__(self, encoder, decoder, discriminator, data, params):
        """
        Initialize trainer.
        """
        super().__init__(range(30))
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.data = data
        self.params = params

        # define encoder parameters (the ones shared with the
        # decoder are optimized by the decoder optimizer)
        enc_params = list(encoder.parameters())
        for i in range(params.n_langs):
            if params.share_lang_emb and i > 0:
                break
            assert enc_params[i].size() == (params.n_words[i], params.emb_dim)
        if self.params.share_encdec_emb:
            to_ignore = 1 if params.share_lang_emb else params.n_langs
            enc_params = enc_params[to_ignore:]

        # optimizers
        if params.dec_optimizer == 'enc_optimizer':
            params.dec_optimizer = params.enc_optimizer
        self.enc_optimizer = get_optimizer(enc_params, params.enc_optimizer) if len(enc_params) > 0 else None
        self.dec_optimizer = get_optimizer(decoder.parameters(), params.dec_optimizer)
        self.dis_optimizer = get_optimizer(discriminator.parameters(), params.dis_optimizer) if discriminator is not None else None

        # models / optimizers
        self.model_opt = {
            'enc': (self.encoder, self.enc_optimizer),
            'dec': (self.decoder, self.dec_optimizer),
            'dis': (self.discriminator, self.dis_optimizer),
        }

        # define validation metrics / stopping criterion used for early stopping
        logger.info("Stopping criterion: %s" % params.stopping_criterion)
        if params.stopping_criterion == '':
            for lang1, lang2 in self.data['para'].keys():
                for data_type in ['valid', 'test']:
                    self.VALIDATION_METRICS.append('bleu_%s_%s_%s' % (lang1, lang2, data_type))
            self.stopping_criterion = None
            self.best_stopping_criterion = None
        else:
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            self.stopping_criterion = split[0]
            self.best_stopping_criterion = -1e12
            assert len(self.VALIDATION_METRICS) == 0
            self.VALIDATION_METRICS.append(self.stopping_criterion)

        # training variables
        self.best_metrics = {metric: -1e12 for metric in self.VALIDATION_METRICS}
        self.epoch = 0
        self.n_total_iter = 0
        self.freeze_enc_emb = self.params.freeze_enc_emb
        self.freeze_dec_emb = self.params.freeze_dec_emb

        # training statistics
        self.n_iter = 0
        self.n_sentences = 0
        self.stats = {
            'dis_costs': [],
            'processed_s': 0,
            'processed_w': 0,
        }
        for lang in params.mono_directions:
            self.stats['xe_costs_%s_%s' % (lang, lang)] = []
        for lang1, lang2 in params.para_directions:
            self.stats['xe_costs_%s_%s' % (lang1, lang2)] = []
        self.last_time = time.time()

        # data iterators
        self.iterators = {}

        # initialize BPE subwords
        self.init_bpe()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params, 'lambda_xe_mono')
        parse_lambda_config(params, 'lambda_xe_para')
        parse_lambda_config(params, 'lambda_dis')
        parse_lambda_config(params, 'lambda_xe_2_way')

    def init_bpe(self):
        """
        Index BPE words.
        """
        self.bpe_end = []
        for lang in self.params.langs:
            dico = self.data['dico'][lang]
            self.bpe_end.append(np.array([not dico[i].endswith('@@') for i in range(len(dico))]))

    def get_iterator(self, iter_name, lang1, lang2):
        """
        Create a new iterator for a dataset.
        """
        key = ','.join([x for x in [iter_name, lang1, lang2] if x is not None])
        logger.info("Creating new training %s iterator ..." % key)
        if lang2 is None:
            dataset = self.data['mono'][lang1]['train']
        else:
            k = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            dataset = self.data['para'][k]['train']
        iterator = dataset.get_iterator(shuffle=True, group_by_size=self.params.group_by_size)()
        self.iterators[key] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2):
        """
        Return a batch of sentences from a dataset.
        """
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        key = ','.join([x for x in [iter_name, lang1, lang2] if x is not None])
        iterator = self.iterators.get(key, None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, lang1, lang2)
            batch = next(iterator)
        return batch if (lang2 is None or lang1 < lang2) else batch[::-1]

    def word_shuffle(self, x, l, lang_id):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = word_idx[:l[i] - 1, i] + noise[word_idx[:l[i] - 1, i], i]
            scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l, lang_id):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        bos_index = self.params.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        # be sure to drop entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[word_idx[j, i], i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(self.params.eos_index)
            assert len(new_s) >= 3 and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l, lang_id):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        bos_index = self.params.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        # be sure to blank entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[word_idx[j, i], i] else self.params.blank_index for j, w in enumerate(words)]
            new_s.append(self.params.eos_index)
            assert len(new_s) == l[i] and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_noise(self, words, lengths, lang_id):
        """
        Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths, lang_id)
        words, lengths = self.word_dropout(words, lengths, lang_id)
        words, lengths = self.word_blank(words, lengths, lang_id)
        return words, lengths

    def zero_grad(self, models):
        """
        Zero gradients.
        """
        if type(models) is not list:
            models = [models]
        models = [self.model_opt[name] for name in models]
        for _, optimizer in models:
            if optimizer is not None:
                optimizer.zero_grad()

    def update_params(self, models):
        """
        Update parameters.
        """
        if type(models) is not list:
            models = [models]
        # don't update encoder when it's frozen
        models = [self.model_opt[name] for name in models]
        # clip gradients
        for model, _ in models:
            clip_grad_norm_(model.parameters(), self.params.clip_grad_norm)

        # optimizer
        for _, optimizer in models:
            if optimizer is not None:
                optimizer.step()

    def get_lrs(self, models):
        """
        Get current optimizer learning rates.
        """
        if type(models) is not list:
            models = [models]
        lrs = {}
        for name in models:
            optimizer = self.model_opt[name][1]
            if optimizer is not None:
                lrs[name] = optimizer.param_groups[0]['lr']
        return lrs

    def discriminator_step(self):
        """
        Train the discriminator on the latent space.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.train()

        # train on monolingual data only
        if self.params.n_mono == 0:
            raise Exception("No data to train the discriminator!")

        # batch / encode
        encoded = []
        for lang_id, lang in enumerate(self.params.langs):
            sent1, len1 = self.get_batch('dis', lang, None)
            with torch.no_grad():
                if self.params.cuda:
                    encoded.append(self.encoder(sent1.cuda(), len1, lang_id))
                else:
                    encoded.append(self.encoder(sent1, len1, lang_id))


        # discriminator
        dis_inputs = [x.dis_input.view(-1, x.dis_input.size(-1)) for x in encoded]
        ntokens = [dis_input.size(0) for dis_input in dis_inputs]
        encoded = torch.cat(dis_inputs, 0)
        predictions = self.discriminator(encoded.data)

        # loss
        self.dis_target = torch.cat([torch.zeros(sz).fill_(i) for i, sz in enumerate(ntokens)])
        if self.params.cuda:
            self.dis_target = self.dis_target.contiguous().long().cuda()
        else:
            self.dis_target = self.dis_target.contiguous().long()
        y = self.dis_target

        loss = F.cross_entropy(predictions, y)
        self.stats['dis_costs'].append(loss.item())

        # optimizer
        self.zero_grad('dis')
        loss.backward()
        self.update_params('dis')
        clip_parameters(self.discriminator, self.params.dis_clip)

    def enc_dec_step(self, lang1, lang2, lambda_xe):
        """
        Source / target autoencoder training (parallel data):
            - encoders / decoders training on cross-entropy
            - encoders training on discriminator feedback
        """
        params = self.params
        assert lang1 in params.langs and lang2 in params.langs
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        loss_fn = self.decoder.loss_fn[lang2_id]
        n_words = params.n_words[lang2_id]
        self.encoder.train()
        self.decoder.train()
        if self.discriminator is not None:
            self.discriminator.eval()

        # batch
        if lang1 == lang2:
            sent1, len1 = self.get_batch('encdec', lang1, None)
            sent2, len2 = sent1, len1
        else:
            (sent1, len1), (sent2, len2) = self.get_batch('encdec', lang1, lang2)

        # prepare the encoder / decoder inputs
        if lang1 == lang2:
            sent1, len1 = self.add_noise(sent1, len1, lang1_id)
        if params.cuda:
            sent1, sent2 = sent1.cuda(), sent2.cuda()

        # encoded states
        encoded = self.encoder(sent1, len1, lang1_id)
        #self.stats['enc_norms_%s' % lang1].append(encoded.dis_input.data.norm(2, 1).mean().item())

        # cross-entropy scores / loss
        scores = self.decoder(encoded, sent2[:-1], lang2_id)
        xe_loss = loss_fn(scores.view(-1, n_words), sent2[1:].view(-1))
        self.stats['xe_costs_%s_%s' % (lang1, lang2)].append(xe_loss.item())

        # discriminator feedback loss
        if params.lambda_dis:
            predictions = self.discriminator(encoded.dis_input.view(-1, encoded.dis_input.size(-1)))
            fake_y = torch.LongTensor(predictions.size(0)).random_(1, params.n_langs)
            fake_y = (fake_y + lang1_id) % params.n_langs
            if params.cuda:
                fake_y = fake_y.cuda()
            dis_loss = F.cross_entropy(predictions, fake_y)

        # total loss
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss
        if params.lambda_dis:
            loss = loss + params.lambda_dis * dis_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.zero_grad(['enc', 'dec'])
        loss.backward()
        self.update_params(['enc', 'dec'])

        # number of processed sentences / words
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += len2.sum()


    def enc_dec_step_mono(self, lang1, lang2, lambda_xe):
        """
        Source / source autoencoder training:
            - encoders / decoders training on cross-entropy
        """
        params = self.params
        assert lang1 in params.langs and lang2 in params.langs
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        loss_fn = self.decoder.loss_fn[lang2_id]
        n_words = params.n_words[lang2_id]
        self.encoder.train()
        self.decoder.train()
        if self.discriminator is not None:
            self.discriminator.eval()

        # batch
        if lang1 == lang2:
            sent1, len1 = self.get_batch('encdec', lang1, None)
            sent2, len2 = sent1, len1
        else:
            (sent1, len1), (sent2, len2) = self.get_batch('encdec', lang1, lang2)

        # prepare the encoder / decoder inputs
        if lang1 == lang2:
            sent1, len1 = self.add_noise(sent1, len1, lang1_id)
        if params.cuda:
            sent1, sent2 = sent1.cuda(), sent2.cuda()

        # encoded states
        encoded = self.encoder(sent1, len1, lang1_id)
        #self.stats['enc_norms_%s' % lang1].append(encoded.dis_input.data.norm(2, 1).mean().item())

        # cross-entropy scores / loss
        sent_back, len_back, _ = self.decoder.generate(encoded, lang2_id)
        encoded_back = self.encoder(sent_back, len_back, lang1_id)
        scores_back = self.decoder(encoded_back, sent1[:-1], lang1_id)
        xe_loss = loss_fn(scores_back.view(-1, n_words), sent1[1:].view(-1))
        self.stats['xe_costs_%s_%s' % (lang1, lang2)].append(xe_loss.item())

        # discriminator feedback loss
        if params.lambda_dis:
            predictions = self.discriminator(encoded.dis_input.view(-1, encoded.dis_input.size(-1)))
            fake_y = torch.LongTensor(predictions.size(0)).random_(1, params.n_langs)
            fake_y = (fake_y + lang1_id) % params.n_langs
            if params.cuda:
                fake_y = fake_y.cuda()
            dis_loss = F.cross_entropy(predictions, fake_y)

        # total loss
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss
        if params.lambda_dis:
            loss = loss + params.lambda_dis * dis_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.zero_grad(['enc', 'dec'])
        loss.backward()
        self.update_params(['enc', 'dec'])

        # number of processed sentences / words
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += len2.sum()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        n_batches = len(self.params.mono_directions) + len(self.params.para_directions)
        self.n_sentences += n_batches * self.params.batch_size
        self.print_stats()
        update_lambdas(self.params, self.n_total_iter)

    def print_stats(self):
        """
        Print statistics about the training.
        """
        # average loss / statistics
        if self.n_iter % 50 == 0:
            mean_loss = [
                ('DIS', 'dis_costs'),
            ]
            for lang in self.params.mono_directions:
                mean_loss.append(('XE-%s-%s' % (lang, lang), 'xe_costs_%s_%s' % (lang, lang)))
            for lang1, lang2 in self.params.para_directions:
                mean_loss.append(('XE-%s-%s' % (lang1, lang2), 'xe_costs_%s_%s' % (lang1, lang2)))

            s_iter = "%7i - " % self.n_iter
            s_stat = ' || '.join(['{}: {:7.4f}'.format(k, np.mean(self.stats[l]))
                                 for k, l in mean_loss if len(self.stats[l]) > 0])
            for _, l in mean_loss:
                del self.stats[l][:]

            # processing speed
            new_time = time.time()
            diff = new_time - self.last_time
            s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(self.stats['processed_s'] * 1.0 / diff,
                                                                   self.stats['processed_w'] * 1.0 / diff)
            self.stats['processed_s'] = 0
            self.stats['processed_w'] = 0
            self.last_time = new_time

            lrs = self.get_lrs(['enc', 'dec'])
            s_lr = " - LR " + ",".join("{}={:.4e}".format(k, lr) for k, lr in lrs.items())

            # log speed + stats
            logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving model to %s ...' % path)
        torch.save({
            'enc': self.encoder,
            'dec': self.decoder,
            'dis': self.discriminator,
        }, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        checkpoint_data = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'discriminator': self.discriminator,
            'enc_optimizer': self.enc_optimizer,
            'dec_optimizer': self.dec_optimizer,
            'dis_optimizer': self.dis_optimizer,
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(checkpoint_data, checkpoint_path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        # reload checkpoint
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            return
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path)
        self.encoder = checkpoint_data['encoder']
        self.decoder = checkpoint_data['decoder']
        self.discriminator = checkpoint_data['discriminator']
        self.enc_optimizer = checkpoint_data['enc_optimizer']
        self.dec_optimizer = checkpoint_data['dec_optimizer']
        self.dis_optimizer = checkpoint_data['dis_optimizer']
        self.epoch = checkpoint_data['epoch']
        self.n_total_iter = checkpoint_data['n_total_iter']
        self.best_metrics = checkpoint_data['best_metrics']
        self.best_stopping_criterion = checkpoint_data['best_stopping_criterion']
        self.model_opt = {
            'enc': (self.encoder, self.enc_optimizer),
            'dec': (self.decoder, self.dec_optimizer),
            'dis': (self.discriminator, self.dis_optimizer),
        }
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)

    def test_sharing(self):
        """
        Test to check that parameters are shared correctly.
        """
        test_sharing(self.encoder, self.decoder, self.params)
        logger.info("Test: Parameters are shared correctly.")

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        for metric in self.VALIDATION_METRICS:
            if scores[metric] > self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if self.params.save_periodic and self.epoch % 20 == 0 and self.epoch > 0:
            self.save_model('periodic-%i' % self.epoch)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None:
            assert self.stopping_criterion in scores
            if scores[self.stopping_criterion] > self.best_stopping_criterion:
                self.best_stopping_criterion = scores[self.stopping_criterion]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            if scores[self.stopping_criterion] < self.best_stopping_criterion:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                exit()
        self.epoch += 1
        self.save_checkpoint()
