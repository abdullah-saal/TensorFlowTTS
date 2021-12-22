from typing import List
import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor, AutoConfig
from tensorflow_tts.processor import SaalTTSProcessor
from phonetizer import Phonetize
import anltk
from collections import deque
import os


def disseminate_shadda(line: str) -> str:
    """ Replace shadda with double letters
        eg : "الشَّمس" becomes "الشْ شَمس"
    """
    output = deque()
    SHADDA = chr(1617)
    SUKOON = chr(1618)

    i = 0
    sz = len(line)
    while i < sz:

        if line[i] == SHADDA:
            output.append(SUKOON)
            output.append(' ')
            output.append(line[i - 1])
            if i + 1 < sz and anltk.is_tashkeel(line[i + 1]):
                output.append(line[i + 1])
                i += 1
            else:
                output.append(SUKOON)

        else:
            output.append(line[i])
        i += 1
    return ''.join(output)


class SaalTTS:
    def __init__(self, path: str, saved: bool = False) -> None:
        self.processor = SaalTTSProcessor(
            data_dir=None,
            loaded_mapper_path='dump_ljspeech_ar/saal_mapper.json')
        if saved:
            self.tacotron2 = tf.saved_model.load(
                os.path.join(path, "tacatron2_saved"))
            self.mb_melgan = tf.saved_model.load(
                os.path.join(path, "mb_melgan_saved"))
            self.fastspeech2 = tf.saved_model.load(
                os.path.join(path, "fast_speech2_saved"))
        else:
            self.tacotron2_config = AutoConfig.from_pretrained(
                "./TensorflowTTS/examples/tacotron2/exp/train.ar.tacotron2.v4/config.yml"
            )
            # fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
            self.tacotron2 = TFAutoModel.from_pretrained(
                "./TensorflowTTS/examples/tacotron2/exp/train.ar.tacotron2.v5/checkpoints/model-42000.h5",
                name="tacotron2",
                config=self.tacotron2_config)
            self.mb_melgan_config = AutoConfig.from_pretrained(
                'models/mb_gan/multiband_melgan.v1.yaml')
            self.mb_melgan = TFAutoModel.from_pretrained(
                config=self.mb_melgan_config,
                pretrained_path="models/mb_gan/generator-940000.h5",
                name="mb_melgan")
            self.fastspeech2_config = AutoConfig.from_pretrained(
                './TensorflowTTS/examples/fastspeech2/conf/fastspeech2.v2.yaml'
            )
            self.fastspeech2 = TFAutoModel.from_pretrained(
                './TensorflowTTS/examples/fastspeech2/exp/train.ar.fastspeech2.v2/checkpoints/model-200000.h5',
                name="fastspeech2",
                config=self.fastspeech2_config)

    def _preprocess(self, text: str) -> str:
        return disseminate_shadda(text)

    def __call__(self, lines: List[str], out_audio_path: str = 'audio.wav'):
        audio = []
        for line in lines:
            text = self._preprocess(line)
            phones = Phonetize(text)
            input_ids = self.processor.text_to_sequence(phones)
            print(line)
            print(input_ids)
            _, mel_outputs, stop_token_prediction, alignment_history = self.tacotron2.inference(
                tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                tf.convert_to_tensor([len(input_ids)], tf.int32),
                tf.convert_to_tensor([0], dtype=tf.int32))
            audio_out = self.mb_melgan.inference(mel_outputs)[0, :, 0]
            audio.append(audio_out)
            # save to file
        audio = np.concatenate(audio, axis=None)
        sf.write(out_audio_path, audio, 16000, "PCM_16")

    def infer(self, lines: List[str], out_audio_path: str = 'audiofs'):
        audio = []
        for line in lines:
            text = self._preprocess(line)
            phones = Phonetize(text)
            input_ids = self.processor.text_to_sequence(phones)
            print(line)
            print(input_ids)
            mel_before, mel_after, duration_outputs, _, _ = self.fastspeech2.inference(
                input_ids=tf.expand_dims(
                    tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32))
            audio_before = self.mb_melgan.inference(mel_before)[0, :, 0]
            audio_after = self.mb_melgan.inference(mel_after)[0, :, 0]
            audio.append(audio_after)
        audio = np.concatenate(audio, axis=None)

        # save to file
        # sf.write(out_audio_path + "_before.wav", audio_before, 16000, "PCM_16")
        sf.write(out_audio_path + "_after.wav", audio, 16000, "PCM_16")

    def export(self, path: str):
        self.tacotron2.setup_window(win_front=6, win_back=6)
        self.tacotron2.setup_maximum_iterations(3000)
        tf.saved_model.save(self.tacotron2,
                            os.path.join(path, 'tacatron2_saved'),
                            signatures=self.tacotron2.inference)
        tf.saved_model.save(self.mb_melgan,
                            os.path.join(path, 'mb_melgan_saved'),
                            signatures=self.mb_melgan.inference)


# print(tacotron2)
# # initialize mb_melgan model
# mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")

# # inference
# processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
text = "السَّمَاءُ هِيَ وَصْفٌ لِمَا نَرَاهُ فَوْقَ الأَرْضِ، وَهِيَ أَيْضًا الْكَوْنُ بِمَا يَحْوِيهِ مِنْ مَجَرَّاتٍ وَنُجُومٍ وَكَوَاكِبَ وَمَادَّةٍ مُظْلِمَةٍ تَنْتَشِرُ فِي جَمِيعِ أَرْجَاءِ الْكَوْنِ، وَمَا نَرَاهُ مِنْ لَوْنٍ أَزْرَقَ فَهُوَ انْعِكَاسُ ضَوْءِ الشَّمْسِ عَلَى الْغِلَافِ الْجَوِّيِّ لِلأَرْضِ."
lines = [
    "السَّمَاءُ هِيَ وَصْفٌ لِمَا نَرَاهُ فَوْقَ الأَرْضِ",
    "وَهِيَ أَيْضًا الْكَوْنُ بِمَا يَحْوِيهِ مِنْ مَجَرَّاتٍ وَنُجُومٍ",
    "وَكَوَاكِبَ وَمَادَّةٍ مُظْلِمَةٍ تَنْتَشِرُ فِي جَمِيعِ أَرْجَاءِ الْكَوْنِ",
    "وَمَا نَرَاهُ مِنْ لَوْنٍ أَزْرَقَ فَهُوَ انْعِكَاسُ ضَوْءِ الشَّمْسِ",
    "الشَّمْسِ عَلَى الْغِلَافِ الْجَوِّيِّ لِلأَرْضِ"
]
tts = SaalTTS('./models/saved', saved=False)

# tts(lines, 'out_audio_saved.wav')
tts.infer(lines)
# tts.export('./models/saved')
import IPython

IPython.embed()
exit(1)
# sf.write('./audio_after.wav', audio_after, 16000, "PCM_16")