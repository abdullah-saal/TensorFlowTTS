

CUDA_VISIBLE_DEVICES=9 python extract_duration.py \
  --rootdir ../../../dump_ljspeech_ar/valid/  \
  --outdir ../../../dump_ljspeech_ar/valid/durations/ \
  --checkpoint ./exp/train.ar.tacotron2.v5/checkpoints/model-48000.h5 \
  --use-norm 1 \
  --config ./conf/tacotron2.v1.yaml \
  --batch-size 32 \
  --win-front 3 \
  --win-back 3 || exit

CUDA_VISIBLE_DEVICES=9 python extract_duration.py \
  --rootdir ../../../dump_ljspeech_ar/train/  \
  --outdir ../../../dump_ljspeech_ar/train/durations/ \
  --checkpoint ./exp/train.ar.tacotron2.v5/checkpoints/model-48000.h5 \
  --use-norm 1 \
  --config ./conf/tacotron2.v1.yaml \
  --batch-size 32 \
  --win-front 3 \
  --win-back 3