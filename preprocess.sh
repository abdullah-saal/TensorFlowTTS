

rootdir=ljspeech_ar
tensorflow-tts-preprocess --rootdir ./$rootdir --outdir ./dump_22_$rootdir --config ./TensorflowTTS/preprocess/ljspeech_preprocess.yaml  --dataset saal || exit

tensorflow-tts-normalize --rootdir ./dump_22_$rootdir --outdir ./dump_22_$rootdir --config ./TensorflowTTS/preprocess/ljspeech_preprocess.yaml --dataset saal
