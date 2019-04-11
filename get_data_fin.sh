# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

#
# Data preprocessing configuration
#

N_MONO=1000000   # number of monolingual sentences for each language
CODES=40000      # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epochs


#
# Initialize tools and data paths
#

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para
VALID_FR_PATH=$PARA_PATH/validation_fr
VALID_EM_PATH=$PARA_PATH/validation_em

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH
mkdir -p $VALID_FR_PATH
mkdir -p $VALID_EM_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# files full paths
SRC_DWN=$MONO_PATH/formal
TGT_DWN=$MONO_PATH/informal

SRC_RAW=$MONO_PATH/all.f
TGT_RAW=$MONO_PATH/all.in
SRC_TOK=$MONO_PATH/all.f.tok
TGT_TOK=$MONO_PATH/all.in.tok
BPE_CODES=$MONO_PATH/bpe_codes
CONCAT_BPE=$MONO_PATH/all.f-in.$CODES
SRC_VOCAB=$MONO_PATH/vocab.f.$CODES
TGT_VOCAB=$MONO_PATH/vocab.in.$CODES
FULL_VOCAB=$MONO_PATH/vocab.f-in.$CODES

PARA_SRC=$PARA_PATH/train.f
PARA_TGT=$PARA_PATH/train.in
PARA_SRC_RAW=$PARA_PATH/formal.train_raw
PARA_TGT_RAW=$PARA_PATH/informal.train_raw

SRC_VALID=$PARA_PATH/val.f
TGT_VALID=$PARA_PATH/val.in
SRC_TEST=$PARA_PATH/test.f
TGT_TEST=$PARA_PATH/test.in

SRC_VALID_RAW=$PARA_PATH/formal.val_raw
TGT_VALID_RAW=$PARA_PATH/informal.val_raw
SRC_TEST_RAW=$PARA_PATH/formal.test_raw
TGT_TEST_RAW=$PARA_PATH/informal.test_raw


SRC_IN_FR_RAW=$VALID_FR_PATH/informal.fr_raw
SRC_IN_FR=$VALID_FR_PATH/informal.fr
REF_F_FR_REF0=$VALID_FR_PATH/formal.ref0.fr
REF_F_FR_REF1=$VALID_FR_PATH/formal.ref1.fr
REF_F_FR_REF2=$VALID_FR_PATH/formal.ref2.fr
REF_F_FR_REF3=$VALID_FR_PATH/formal.ref3.fr

SRC_F_FR_RAW=$VALID_FR_PATH/formal.fr_raw
SRC_F_FR=$VALID_FR_PATH/formal.fr
REF_IN_FR_REF0=$VALID_FR_PATH/informal.ref0.fr
REF_IN_FR_REF1=$VALID_FR_PATH/informal.ref1.fr
REF_IN_FR_REF2=$VALID_FR_PATH/informal.ref2.fr
REF_IN_FR_REF3=$VALID_FR_PATH/informal.ref3.fr


SRC_IN_EM_RAW=$VALID_EM_PATH/informal.em_raw
SRC_IN_EM=$VALID_EM_PATH/informal.em
REF_F_EM_REF0=$VALID_EM_PATH/formal.ref0.em
REF_F_EM_REF1=$VALID_EM_PATH/formal.ref1.em
REF_F_EM_REF2=$VALID_EM_PATH/formal.ref2.em
REF_F_EM_REF3=$VALID_EM_PATH/formal.ref3.em

SRC_F_EM_RAW=$VALID_EM_PATH/formal.em_raw
SRC_F_EM=$VALID_EM_PATH/formal.em
REF_IN_EM_REF0=$VALID_EM_PATH/informal.ref0.em
REF_IN_EM_REF1=$VALID_EM_PATH/informal.ref1.em
REF_IN_EM_REF2=$VALID_EM_PATH/informal.ref2.em
REF_IN_EM_REF3=$VALID_EM_PATH/informal.ref3.em



#
# Download and install tools
#

# Download Moses
cd $TOOLS_PATH
if [ ! -d "$MOSES" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
echo "Moses found in: $MOSES"

# Download fastBPE
cd $TOOLS_PATH
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi
echo "fastBPE found in: $FASTBPE_DIR"


# Compile fastBPE
cd $TOOLS_PATH
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd $FASTBPE_DIR
  g++ -std=c++11 -pthread -O3 fast.cc -o fast
fi
echo "fastBPE compiled in: $FASTBPE"

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"


#
# Download monolingual data
#

cd $MONO_PATH
if [ ! -f "$SRC_DWN" ]; then
  echo "Downloading Formal files..."
  wget -O mono1.tar.gz 'https://drive.google.com/uc?export=download&id=1Lz3dgAvsqUFc2SKCI_on-PZtwUrqgmyt'
  wget -O mono2.tar.gz 'https://drive.google.com/uc?export=download&id=1ABmPPP6imT4Mp_8B8wrMj6CxTGd1-LgB'
  wget -O mono3.tar.gz 'https://drive.google.com/uc?export=download&id=1JRlCOhkMslqmHs7YvBcsGyWK2t01PymH'
  wget -O mono4.tar.gz 'https://drive.google.com/uc?export=download&id=1OGPT9MXfGo4dAiZB3NmfjQ17lHR16iCX'
  wget -O mono5.tar.gz 'https://drive.google.com/uc?export=download&id=1jthAVesgoh3-zUH09gZtRrCyo7X1mNlL'
  wget -O mono6.tar.gz 'https://drive.google.com/uc?export=download&id=1kjaQLWqhAlpIEaG9NDerAEXvzCMYuu6U'
  wget -O mono7.tar.gz 'https://drive.google.com/uc?export=download&id=1AVp-KMYo64yE3p-IiZ8k8VQSa-a2sQNY'
  wget -O mono8.tar.gz 'https://drive.google.com/uc?export=download&id=1jsDTmxwQV6ymWwJtVI5LZBtKo4qGAcd5'
  wget -O mono9.tar.gz 'https://drive.google.com/uc?export=download&id=1V9GuF7f5v0AuZCW0seFiwAU_XyLygHBt'
  wget -O mono10.tar.gz 'https://drive.google.com/uc?export=download&id=1TYODjpTHP2jknRNf2Ow9DGd5nLEycGC1'
  wget -O mono11.tar.gz 'https://drive.google.com/uc?export=download&id=1pn_kyDDGhnJroDiAknYHpYs8WsC5l27F'
  wget -O mono12.tar.gz 'https://drive.google.com/uc?export=download&id=1DthmXZvcP2RihlRAP9UvUnuDS78ecI2T'
  wget -O mono13.tar.gz 'https://drive.google.com/uc?export=download&id=1eqRDpSEK4jxjZ5BD5HKd7_6r2lmB3p9o'
  wget -O mono14.tar.gz 'https://drive.google.com/uc?export=download&id=1WCAJWZc157JETO7q1DNifscNZS-GWOLd'
  wget -O mono15.tar.gz 'https://drive.google.com/uc?export=download&id=1kcr7RSsIIjWKc-YHZj6RJNkX3IW7-3_C'
  wget -O mono16.tar.gz 'https://drive.google.com/uc?export=download&id=1tQlOvx6LDNZhDOkD0Pb-Z6xpTfQoPwcl'
  wget -O mono17.tar.gz 'https://drive.google.com/uc?export=download&id=1KZR5C-X8eUQqQcqGsa7NDr1lkV5tpQQE'
  wget -O mono18.tar.gz 'https://drive.google.com/uc?export=download&id=11YWeH2qTg6G2lW8ahRiY3ZENZWVPYlo8'
  wget -O mono19.tar.gz 'https://drive.google.com/uc?export=download&id=1TVs02Zt2YVefh4AOYKeX2UNqh7vBLHuJ'
  wget -O mono20.tar.gz 'https://drive.google.com/uc?export=download&id=1cNQTUhI3kMh_GPHVqm_W0oFbZmToUdZf'
  wget -O mono21.tar.gz 'https://drive.google.com/uc?export=download&id=1C__6ogurs4u_xY7nJ9vmP1xJczcNzP7-'
  wget -O mono22.tar.gz 'https://drive.google.com/uc?export=download&id=1NZgD0nCk0udvnl5ZGfO3HWFucLPeKoVu'
  wget -O mono23.tar.gz 'https://drive.google.com/uc?export=download&id=18C7R6bQwp4ooiI7J0KxcD4uBn7eK0R-4'
  wget -O mono24.tar.gz 'https://drive.google.com/uc?export=download&id=17WsJT-VVutz1wqXLfC64ZzK6-w6sSuuQ'
  wget -O mono25.tar.gz 'https://drive.google.com/uc?export=download&id=15fHMSDtIwTyRRjokDEiGcK5Eeo24yJn9'
  wget -O mono26.tar.gz 'https://drive.google.com/uc?export=download&id=1zyTYg-cYw6-Sj2j1U-QrKIVqRk0F1Tj2'
  wget -O mono27.tar.gz 'https://drive.google.com/uc?export=download&id=1u3wejKHUEPZWLpkuGJbOL3UpATllUAG8'
  wget -O mono28.tar.gz 'https://drive.google.com/uc?export=download&id=170bDAwe83ZHES_3R6vG3WZ6o_2e7TNpA'
  wget -O mono29.tar.gz 'https://drive.google.com/uc?export=download&id=1AtmBPwsZhXuluBbNdEp06n-W4IUH7OYy'
  wget -O mono30.tar.gz 'https://drive.google.com/uc?export=download&id=1iL7LFiNdAfDymqYOUX6fSsADC1ZdaQHb'
  wget -O mono31.tar.gz 'https://drive.google.com/uc?export=download&id=1ndWfRnO8hPFAnGkEvGiUAEfuxfEb9vCz'
  wget -O mono32.tar.gz 'https://drive.google.com/uc?export=download&id=1kviWzD0TosLCtZMJKE0wmyBlJaByuHkJ'

  for FILENAME in mono*tar.gz; do
    OUTPUT="${FILENAME::-7}"
    if [ ! -f "$OUTPUT" ]; then
      echo "Decompressing $FILENAME..."
      tar -xzf $FILENAME
    else
      echo "$OUTPUT already decompressed."
    fi
  done
  for DIRNAME in output*; do
    cd $DIRNAME
    if ls formal_* 1> /dev/null 2>&1; then
      cat $(ls formal_* | grep -v gz) | sed -r '/^\s*$/d' >> $MONO_PATH/formal
    fi
    if ls informal_* 1> /dev/null 2>&1; then
      cat $(ls informal_* | grep -v gz) | sed -r '/^\s*$/d' >> $MONO_PATH/informal
    fi
    cd $MONO_PATH
  done
fi


if ! [[ -f "$SRC_RAW" && -f "$TGT_RAW" ]]; then
  echo "Cutting monolingual data..."
  cat formal | head -n $N_MONO > $SRC_RAW
  cat informal | head -n $N_MONO > $TGT_RAW
fi
cd $MONO_PATH

#if [ ! -f "$TGT_DWN" ]; then
#  echo "Downloading Informal files..."
#  wget -O informal 'https://drive.google.com/uc?export=download&id=1cHXY4G73RN3Rvcn7xWLG9J_GKjH5oWk3'
#fi
# decompress monolingual data
##HBB## USE THIS IF DECOMPRESSION IS NEEDED
##HBB##for FILENAME in news*gz; do
##HBB##  OUTPUT="${FILENAME::-3}"
##HBB##  if [ ! -f "$OUTPUT" ]; then
##HBB##    echo "Decompressing $FILENAME..."
##HBB##    gunzip -k $FILENAME
##HBB##  else
##HBB##    echo "$OUTPUT already decompressed."
##HBB##  fi
##HBB##done


# concatenate monolingual data files
##HBB##if ! [[ -f "$SRC_RAW" && -f "$TGT_RAW" ]]; then
##HBB##  echo "Concatenating monolingual data..."
##HBB##  cat $(ls news*en* | grep -v gz) | head -n $N_MONO > $SRC_RAW
##HBB##  cat $(ls news*fr* | grep -v gz) | head -n $N_MONO > $TGT_RAW
##HBB##fi
#mv $SRC_DWN $SRC_RAW
#mv $TGT_DWN $TGT_RAW
echo "Formal monolingual data is in: $SRC_RAW"
echo "Informal monolingual data is in: $TGT_RAW"


# check number of lines
if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your Formal monolingual data."; exit; fi
if ! [[ "$(wc -l < $TGT_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your Informal monolingual data."; exit; fi

# tokenize data
if ! [[ -f "$SRC_TOK" && -f "$TGT_TOK" ]]; then
  echo "Tokenize monolingual data..."
  cat $SRC_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TOK
  cat $TGT_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_TOK
fi
echo "Formal data tokenized in: $SRC_TOK"
echo "Informal data tokenized in: $TGT_TOK"

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $SRC_TOK $TGT_TOK > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

# apply BPE codes
if ! [[ -f "$SRC_TOK.$CODES" && -f "$TGT_TOK.$CODES" ]]; then
  echo "Applying BPE codes..."
  $FASTBPE applybpe $SRC_TOK.$CODES $SRC_TOK $BPE_CODES
  $FASTBPE applybpe $TGT_TOK.$CODES $TGT_TOK $BPE_CODES
fi
echo "BPE codes applied to formal in: $SRC_TOK.$CODES"
echo "BPE codes applied to informal in: $TGT_TOK.$CODES"

# extract vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" && -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TOK.$CODES > $SRC_VOCAB
  $FASTBPE getvocab $TGT_TOK.$CODES > $TGT_VOCAB
  $FASTBPE getvocab $SRC_TOK.$CODES $TGT_TOK.$CODES > $FULL_VOCAB
fi
echo "Formal vocab in: $SRC_VOCAB"
echo "Informal vocab in: $TGT_VOCAB"
echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$SRC_TOK.$CODES.pth" && -f "$TGT_TOK.$CODES.pth" ]]; then
  echo "Binarizing data..."
  $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TOK.$CODES
  $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TOK.$CODES
fi
echo "Formal binarized data in: $SRC_TOK.$CODES.pth"
echo "Informal binarized data in: $TGT_TOK.$CODES.pth"


#
# Download parallel data (for evaluation only)
#

cd $PARA_PATH

echo "Downloading parallel data..."
wget -O $PARA_SRC.fandr 'https://drive.google.com/uc?export=download&id=1Xjt9fjwocF_KaGM1I5scc7UlCvbEgf9w'
wget -O $PARA_TGT.fandr 'https://drive.google.com/uc?export=download&id=1G_KWnjVuJrr3GeJTpe8Al7ngPzIBxyR8'
wget -O $PARA_SRC.eandm 'https://drive.google.com/uc?export=download&id=1XQbMTj_4CQKfdeJKS1wuT6ZDkchYuGQB'
wget -O $PARA_TGT.eandm 'https://drive.google.com/uc?export=download&id=1npcrOB-t2TB2SS_MMq_Q_RD-mPbH3Nzu'

wget -O $SRC_VALID_RAW 'https://drive.google.com/uc?export=download&id=11uukepFDeqxTcLDBhADKuvpTItwsUACn'
wget -O $TGT_VALID_RAW 'https://drive.google.com/uc?export=download&id=1ZjSDMLKk65cKy0U--iC20dQioq7BjX3O'
wget -O $SRC_TEST_RAW 'https://drive.google.com/uc?export=download&id=15EKsj-oxfb-30SL5Aq-dQWvAe2eO-lkO'
wget -O $TGT_TEST_RAW 'https://drive.google.com/uc?export=download&id=1N1-pZiXmQhmJhroTOkxsG0JKjAqwHOEX'

cat $PARA_SRC.fandr > $PARA_SRC_RAW
cat $PARA_TGT.fandr > $PARA_TGT_RAW
cat $PARA_SRC.eandm >> $PARA_SRC_RAW
cat $PARA_TGT.eandm >> $PARA_TGT_RAW

#echo "Extracting parallel data..."
#tar -xzf dev.tgz

# check valid and test files are here
if ! [[ -f "$PARA_SRC_RAW" ]]; then echo "$PARA_SR_RAW is not found!"; exit; fi
if ! [[ -f "$PARA_TGT_RAW" ]]; then echo "$PARA_TGT_RAW is not found!"; exit; fi
if ! [[ -f "$SRC_VALID_RAW" ]]; then echo "$SRC_VALID_RAW is not found!"; exit; fi
if ! [[ -f "$TGT_VALID_RAW" ]]; then echo "$TGT_VALID_RAW is not found!"; exit; fi
if ! [[ -f "$SRC_TEST_RAW" ]]; then echo "$SRC_TEST_RAW is not found!"; exit; fi
if ! [[ -f "$TGT_TEST_RAW" ]]; then echo "$TGT_TEST_RAW is not found!"; exit; fi

echo "Tokenizing valid and test data..."

cat $PARA_SRC_RAW | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $PARA_SRC
cat $PARA_TGT_RAW | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $PARA_TGT
cat $SRC_VALID_RAW | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_VALID
cat $TGT_VALID_RAW | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_VALID
cat $SRC_TEST_RAW | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TEST
cat $TGT_TEST_RAW | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_TEST

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $PARA_SRC.$CODES $PARA_SRC $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $PARA_TGT.$CODES $PARA_TGT $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $SRC_VALID.$CODES $SRC_VALID $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_VALID.$CODES $TGT_VALID $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $SRC_TEST.$CODES $SRC_TEST $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_TEST.$CODES $TGT_TEST $BPE_CODES $TGT_VOCAB

echo "Binarizing data..."
rm -f $PARA_SRC.$CODES.pth $PARA_TGT.$CODES.pth $SRC_VALID.$CODES.pth $TGT_VALID.$CODES.pth $SRC_TEST.$CODES.pth $TGT_TEST.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $PARA_SRC.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $PARA_TGT.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST.$CODES



echo "Downloading parallel data for validation..."
# Family and Relations
cd $VALID_FR_PATH
wget -O $SRC_IN_FR_RAW 'https://drive.google.com/uc?export=download&id=1535YHKL6fJZyCkVoLO9QQlR3GgW2NIM5'
wget -O $REF_F_FR_REF0 'https://drive.google.com/uc?export=download&id=18TMEFPxDuOxHWDVQgFkJTSa0axQgvfMT'
wget -O $REF_F_FR_REF1 'https://drive.google.com/uc?export=download&id=1f0TUNFAa3Nka9et20HrwK0W53UaWBBiD'
wget -O $REF_F_FR_REF2 'https://drive.google.com/uc?export=download&id=13EKoe2xjRVfygjJlpMqjitVUpVbDr-5H'
wget -O $REF_F_FR_REF3 'https://drive.google.com/uc?export=download&id=1KHEpIKDNZcl1nsq2E7F3lfkPMB0CTooC'

# check files are here
if ! [[ -f "$SRC_IN_FR_RAW" ]]; then echo "$SRC_IN_FR_RAW is not found!"; exit; fi
if ! [[ -f "$REF_F_FR_REF0" ]]; then echo "$REF_F_FR_REF0 is not found!"; exit; fi
if ! [[ -f "$REF_F_FR_REF1" ]]; then echo "$REF_F_FR_REF1 is not found!"; exit; fi
if ! [[ -f "$REF_F_FR_REF2" ]]; then echo "$REF_F_FR_REF2 is not found!"; exit; fi
if ! [[ -f "$REF_F_FR_REF3" ]]; then echo "$REF_F_FR_REF3 is not found!"; exit; fi


echo "Tokenizing source data, the ref files will be used with sacrebleu which does not need preprocessing..."

cat $SRC_IN_FR_RAW | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_IN_FR

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $SRC_IN_FR.$CODES $SRC_IN_FR $BPE_CODES $SRC_VOCAB

echo "Binarizing data..."
rm -f $SRC_IN_FR.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_IN_FR.$CODES


wget -O $SRC_F_FR_RAW 'https://drive.google.com/uc?export=download&id=1GIAnRgFdU64Vp9dS5bCbDYy9W8EiH3o9'
wget -O $REF_IN_FR_REF0 'https://drive.google.com/uc?export=download&id=1yK04PJPJ2SW9EoVxw6pa9bOT4OPJIl7J'
wget -O $REF_IN_FR_REF1 'https://drive.google.com/uc?export=download&id=1v5vhsxRa5x0mEn7fR32SvxvKoS0FqnkS'
wget -O $REF_IN_FR_REF2 'https://drive.google.com/uc?export=download&id=1xLMpNMO7GY_3BIiUH_txrgExYqHEPNdi'
wget -O $REF_IN_FR_REF3 'https://drive.google.com/uc?export=download&id=1zxUVBuOVEk2PTnh_ROtZQ4AMyUHPD5wI'

# check files are here
if ! [[ -f "$SRC_F_FR_RAW" ]]; then echo "$SRC_F_FR_RAW is not found!"; exit; fi
if ! [[ -f "$REF_IN_FR_REF0" ]]; then echo "$REF_IN_FR_REF0 is not found!"; exit; fi
if ! [[ -f "$REF_IN_FR_REF1" ]]; then echo "$REF_IN_FR_REF1 is not found!"; exit; fi
if ! [[ -f "$REF_IN_FR_REF2" ]]; then echo "$REF_IN_FR_REF2 is not found!"; exit; fi
if ! [[ -f "$REF_IN_FR_REF3" ]]; then echo "$REF_IN_FR_REF3 is not found!"; exit; fi


echo "Tokenizing source data, the ref files will be used with sacrebleu which does not need preprocessing..."

cat $SRC_F_FR_RAW | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_F_FR

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $SRC_F_FR.$CODES $SRC_F_FR $BPE_CODES $SRC_VOCAB

echo "Binarizing data..."
rm -f $SRC_F_FR.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_F_FR.$CODES



# Entertainment and Music
cd $VALID_EM_PATH
wget -O $SRC_IN_EM_RAW 'https://drive.google.com/uc?export=download&id=1J88TszNWtCBu0ltxpG_w4gwaauJ7rKLf'
wget -O $REF_F_EM_REF0 'https://drive.google.com/uc?export=download&id=1C-5JuuUgOCQvIEQ4VvF6K_-vCFQg_APL'
wget -O $REF_F_EM_REF1 'https://drive.google.com/uc?export=download&id=1_pabYarvAvf-VdWfZtejX52xN0ZdSMMP'
wget -O $REF_F_EM_REF2 'https://drive.google.com/uc?export=download&id=1mWLA_vxxlBcbmXCMdKUu2CDTD4Q6MRlU'
wget -O $REF_F_EM_REF3 'https://drive.google.com/uc?export=download&id=1IFe6ZoriK9c_88JN0NFhz-q1yj2RJSI6'

# check files are here
if ! [[ -f "$SRC_IN_EM_RAW" ]]; then echo "$SRC_IN_EM_RAW is not found!"; exit; fi
if ! [[ -f "$REF_F_EM_REF0" ]]; then echo "$REF_F_EM_REF0 is not found!"; exit; fi
if ! [[ -f "$REF_F_EM_REF1" ]]; then echo "$REF_F_EM_REF1 is not found!"; exit; fi
if ! [[ -f "$REF_F_EM_REF2" ]]; then echo "$REF_F_EM_REF2 is not found!"; exit; fi
if ! [[ -f "$REF_F_EM_REF3" ]]; then echo "$REF_F_EM_REF3 is not found!"; exit; fi


echo "Tokenizing source data, the ref files will be used with sacrebleu which does not need preprocessing..."

cat $SRC_IN_EM_RAW | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_IN_EM

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $SRC_IN_EM.$CODES $SRC_IN_EM $BPE_CODES $SRC_VOCAB

echo "Binarizing data..."
rm -f $SRC_IN_EM.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_IN_EM.$CODES


wget -O $SRC_F_EM_RAW 'https://drive.google.com/uc?export=download&id=1Xv9h2AlvTiHzWBV9ZqKuvoR6kglIkcg8'
wget -O $REF_IN_EM_REF0 'https://drive.google.com/uc?export=download&id=14rKSeNj8tnmBoafmy8d6VijU9WtgjiKx'
wget -O $REF_IN_EM_REF1 'https://drive.google.com/uc?export=download&id=1RsdkVd8ckergnj36SUlytumyZERFOw13'
wget -O $REF_IN_EM_REF2 'https://drive.google.com/uc?export=download&id=1Slvr8mIZLcoHJNUzBHTSw95Z7QuHZJet'
wget -O $REF_IN_EM_REF3 'https://drive.google.com/uc?export=download&id=1W9nVxCg3Fit26JHc2C6I5zfFlpzmPs-W'

# check files are here
if ! [[ -f "$SRC_F_EM_RAW" ]]; then echo "$SRC_F_EM_RAW is not found!"; exit; fi
if ! [[ -f "$REF_IN_EM_REF0" ]]; then echo "$REF_IN_EM_REF0 is not found!"; exit; fi
if ! [[ -f "$REF_IN_EM_REF1" ]]; then echo "$REF_IN_EM_REF1 is not found!"; exit; fi
if ! [[ -f "$REF_IN_EM_REF2" ]]; then echo "$REF_IN_EM_REF2 is not found!"; exit; fi
if ! [[ -f "$REF_IN_EM_REF3" ]]; then echo "$REF_IN_EM_REF3 is not found!"; exit; fi


echo "Tokenizing source data, the ref files will be used with sacrebleu which does not need preprocessing..."

cat $SRC_F_EM_RAW | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_F_EM

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $SRC_F_EM.$CODES $SRC_F_EM $BPE_CODES $SRC_VOCAB

echo "Binarizing data..."
rm -f $SRC_F_EM.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_F_EM.$CODES



#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monolingual training data:"
echo "    Formal: $SRC_TOK.$CODES.pth"
echo "    Informal: $TGT_TOK.$CODES.pth"
echo "Parallel train data:"
echo "    Formal: $PARA_SRC.$CODES.pth"
echo "    Informal: $PARA_TGT.$CODES.pth"
echo "Parallel validation data:"
echo "    Formal: $SRC_VALID.$CODES.pth"
echo "    Informal: $TGT_VALID.$CODES.pth"
echo "Parallel test data:"
echo "    Formal: $SRC_TEST.$CODES.pth"
echo "    Informal: $TGT_TEST.$CODES.pth"
echo ""


#
# Train fastText on concatenated embeddings
#

if ! [[ -f "$CONCAT_BPE" ]]; then
  echo "Concatenating source and target monolingual data..."
  cat $SRC_TOK.$CODES $TGT_TOK.$CODES | shuf > $CONCAT_BPE
fi
echo "Concatenated data in: $CONCAT_BPE"

if ! [[ -f "$CONCAT_BPE.vec" ]]; then
  echo "Training fastText on $CONCAT_BPE..."
  $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 256 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE -output $CONCAT_BPE
fi
echo "Cross-lingual embeddings in: $CONCAT_BPE.vec"
