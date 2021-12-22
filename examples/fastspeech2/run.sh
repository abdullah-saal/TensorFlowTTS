CUDA_VISIBLE_DEVICES=9 python examples/fastspeech2/train_fastspeech2.py \
  --train-dir ../dump_ljspeech_ar/train/ \
  --dev-dir ../dump_ljspeech_ar/valid/ \
  --outdir ./examples/fastspeech2/exp/train.ar.fastspeech2.v2/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v2.yaml \
  --use-norm 1 \
  --f0-stat ../dump_ljspeech_ar/stats_f0.npy \
  --energy-stat ../dump_ljspeech_ar/stats_energy.npy \
  --mixed_precision 1 \
  --resume ""