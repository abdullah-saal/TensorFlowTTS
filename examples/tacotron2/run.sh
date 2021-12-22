CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python ./train_tacotron2.py \
  --train-dir ../../../dump_ljspeech_ar/train/ \
  --dev-dir ../../../dump_ljspeech_ar/valid/ \
  --outdir ./exp/train.ar.tacotron2.v5/ \
  --config ./conf/tacotron2.v1.yaml \
  --use-norm 1 \
  --mixed_precision 0 \
  --resume ""

  # pip install --force-reinstall --no-deps .