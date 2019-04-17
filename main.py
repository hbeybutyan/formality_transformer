# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import argparse

from src.data.loader import check_all_data_params, load_data, load_generation_set
from src.utils import bool_flag, initialize_exp
from src.model import check_mt_model_params, build_mt_model
from src.trainer import TrainerMT
from src.evaluator import EvaluatorMT
from src.generator import Generator

def get_parser():
    # parse parameters
    parser = argparse.ArgumentParser(description='Language transfer')
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--model_file", type=str, default="",
                        help="Saved model file name to load.")
    parser.add_argument("--save_periodic", type=bool_flag, default=False,
                        help="Save the model periodically")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random generator seed (-1 for random)")
    # autoencoder parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_enc_layers", type=int, default=4,
                        help="Number of layers in the encoders")
    parser.add_argument("--n_dec_layers", type=int, default=4,
                        help="Number of layers in the decoders")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden layer size")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--label-smoothing", type=float, default=0,
                        help="Label smoothing")
    parser.add_argument("--transformer_ffn_emb_dim", type=int, default=2048,
                        help="Transformer fully-connected hidden dim size")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="attention_dropout")
    parser.add_argument("--relu_dropout", type=float, default=0,
                        help="relu_dropout")
    parser.add_argument("--encoder_attention_heads", type=int, default=8,
                        help="encoder_attention_heads")
    parser.add_argument("--decoder_attention_heads", type=int, default=8,
                        help="decoder_attention_heads")
    parser.add_argument("--encoder_normalize_before", type=bool_flag, default=False,
                        help="encoder_normalize_before")
    parser.add_argument("--decoder_normalize_before", type=bool_flag, default=False,
                        help="decoder_normalize_before")
    parser.add_argument("--share_lang_emb", type=bool_flag, default=False,
                        help="Share embedding layers between languages (enc / dec / proj)")
    parser.add_argument("--share_encdec_emb", type=bool_flag, default=False,
                        help="Share encoder embeddings / decoder embeddings")
    parser.add_argument("--share_decpro_emb", type=bool_flag, default=False,
                        help="Share decoder embeddings with decoder output projection")
    parser.add_argument("--share_output_emb", type=bool_flag, default=False,
                        help="Share decoder output embeddings of different languages")
    parser.add_argument("--share_enc", type=int, default=0,
                        help="Number of layers to share in the encoders of different languages")
    parser.add_argument("--share_dec", type=int, default=0,
                        help="Number of layers to share in the decoders of different languages")
    # encoder input perturbation
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")
    # discriminator parameters
    parser.add_argument("--dis_layers", type=int, default=3,
                        help="Number of hidden layers in the discriminator")
    parser.add_argument("--dis_hidden_dim", type=int, default=128,
                        help="Discriminator hidden layers dimension")
    parser.add_argument("--dis_dropout", type=float, default=0,
                        help="Discriminator dropout")
    parser.add_argument("--dis_clip", type=float, default=0,
                        help="Clip discriminator weights (0 to disable)")
    parser.add_argument("--dis_smooth", type=float, default=0,
                        help="GAN smooth predictions")
    parser.add_argument("--dis_input_proj", type=bool_flag, default=True,
                        help="Feed the discriminator with the projected output (attention only)")
    # dataset
    parser.add_argument("--langs", type=str, default="",
                        help="Languages/styles (lang1,lang2)")
    parser.add_argument("--vocab", type=str, default="",
                        help="Vocabulary (lang1:path1;lang2:path2)")
    parser.add_argument("--vocab_min_count", type=int, default=0,
                        help="Vocabulary minimum word count")
    parser.add_argument("--mono_dataset", type=str, default="",
                        help="Monolingual dataset (lang1:train1,valid1,test1;lang2:train2,valid2,test2)")
    parser.add_argument("--para_dataset", type=str, default="",
                        help="Parallel dataset (lang1-lang2:train12,valid12,test12;lang1-lang3:train13,valid13,test13)")
    parser.add_argument("--generation_set", type=str, default="",
                        help="Set of input sentences to transfer style.")
    parser.add_argument("--generation_source_style", type=str, default="",
                        help="Formality of source set used to transform.")
    parser.add_argument("--valid_domain", type=str, default="",
                        help="From which domain are the validation files. fr - family&relations, em - entertainment&music, oth - other")
    parser.add_argument("--n_mono", type=int, default=0,
                        help="Number of monolingual sentences (-1 for everything)")
    parser.add_argument("--n_para", type=int, default=0,
                        help="Number of parallel sentences (-1 for everything)")
    parser.add_argument("--max_len", type=int, default=175,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    # training steps
    parser.add_argument("--n_dis", type=int, default=0,
                        help="Number of discriminator training iterations")
    parser.add_argument("--mono_directions", type=str, default="",
                        help="Training directions (lang1,lang2)")
    parser.add_argument("--para_directions", type=str, default="",
                        help="Training directions (lang1-lang2,lang2-lang1)")
    # training parameters
    parser.add_argument("--cuda", type=bool_flag, default=False,
                        help="Run with cuda.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--lambda_xe_mono", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (autoencoding)")
    parser.add_argument("--lambda_xe_2_way", type=str, default="0",
                        help="Cross-entropy 2 way reconstruction coefficient (lang1 -> lang2 -> lang1)")
    parser.add_argument("--lambda_xe_para", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (parallel data)")
    parser.add_argument("--lambda_dis", type=str, default="0",
                        help="Discriminator loss coefficient")
    parser.add_argument("--enc_optimizer", type=str, default="adam,lr=0.0003",
                        help="Encoder optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--dec_optimizer", type=str, default="enc_optimizer",
                        help="Decoder optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--dis_optimizer", type=str, default="rmsprop,lr=0.0005",
                        help="Discriminator optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    # reload models
    parser.add_argument("--pretrained_emb", type=str, default="",
                        help="Reload pre-trained source and target word embeddings")
    parser.add_argument("--pretrained_out", type=bool_flag, default=False,
                        help="Pretrain the decoder output projection matrix")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pre-trained model")
    parser.add_argument("--reload_enc", type=bool_flag, default=False,
                        help="Reload a pre-trained encoder")
    parser.add_argument("--reload_dec", type=bool_flag, default=False,
                        help="Reload a pre-trained decoder")
    parser.add_argument("--reload_dis", type=bool_flag, default=False,
                        help="Reload a pre-trained discriminator")
    # freeze network parameters
    parser.add_argument("--freeze_enc_emb", type=bool_flag, default=False,
                        help="Freeze encoder embeddings")
    parser.add_argument("--freeze_dec_emb", type=bool_flag, default=False,
                        help="Freeze decoder embeddings")
    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--beam_size", type=int, default=0,
                        help="Beam width (<= 0 means greedy)")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="Length penalty: <1.0 favors shorter, >1.0 favors longer sentences")
    return parser


def main(params):
    # check parameters
    assert params.exp_name
    check_all_data_params(params)
    check_mt_model_params(params)

    # initialize experiment / load data / build model
    logger = initialize_exp(params)
    data = load_data(params)
    encoder, decoder, discriminator = build_mt_model(params, data)

    # initialize trainer / reload checkpoint / initialize evaluator
    trainer = TrainerMT(encoder, decoder, discriminator, data, params)
    if params.model_file:
        trainer.load_saved_model(params.model_file)
    else:
        trainer.reload_checkpoint()
    trainer.test_sharing()  # check parameters sharing
    evaluator = EvaluatorMT(trainer, data, params)

    # generation mode
    if params.generation_set:
        assert params.valid_domain in ["fr", "em", "oth"]
        generator = Generator(trainer, load_generation_set(params, data), params)
        if params.generation_source_style == params.langs[0]:
            generator.generate(params.langs[0], params.langs[1])
        else:
            generator.generate(params.langs[1], params.langs[0])
        exit()

    # evaluation mode
    if params.eval_only:
        evaluator.run_all_evals(0)
        exit()

    # define epoch size
    if params.epoch_size == -1:
        params.epoch_size = params.n_para
    assert params.epoch_size > 0

    # start training
    for _ in range(trainer.epoch, params.max_epoch):

        logger.info("====================== Starting epoch %i ... ======================" % trainer.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < params.epoch_size:

            # discriminator training
            for _ in range(params.n_dis):
                trainer.discriminator_step()

            # autoencoder training (monolingual data)
            if params.lambda_xe_mono > 0:
                for lang in params.mono_directions:
                    trainer.enc_dec_step(lang, lang, params.lambda_xe_mono)
            # 2 way training
            if params.lambda_xe_2_way > 0:
                for lang1, lang2 in params.para_directions:
                    trainer.enc_dec_step_mono(lang1, lang2, params.lambda_xe_2_way)
            # MT training (parallel data)
            if params.lambda_xe_para > 0:
                for lang1, lang2 in params.para_directions:
                    trainer.enc_dec_step(lang1, lang2, params.lambda_xe_para)

            trainer.iter()

        # end of epoch
        logger.info("====================== End of epoch %i ======================" % trainer.epoch)

        # evaluate discriminator / perplexity / BLEU
        scores = evaluator.run_all_evals(trainer.epoch)

        # print / JSON log
        for k, v in scores.items():
            logger.info('%s -> %.6f' % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        # save best / save periodic / end epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)
        trainer.test_sharing()


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)
