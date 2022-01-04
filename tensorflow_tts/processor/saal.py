# -*- coding: utf-8 -*-


import os
import re

import numpy as np
import soundfile as sf
from dataclasses import dataclass

from tensorflow_tts.processor.base_processor import BaseProcessor
from tensorflow_tts.utils.utils import PROCESSOR_FILE_NAME

valid_symbols = [
    'IH',
    'AE',
    'UH',
    'IY',
    'AE:',
    'UW',
    'AA',
    'AA:',
    'AH',
    'AH:',
    'UX',
    'IX',
    'AW',
    'AY',
    'W',
    'Y',
    'L',
    'R',
    'N',
    'M',
    'H',
    'HH',
    'AI',
    'B',
    'T',
    'D',
    'TT',
    'DD',
    'K',
    'E',
    'Q',
    'F',
    'TH',
    'DH',
    'DH2',
    'S',
    'Z',
    'SS',
    'SH',
    'ZH',
    'KH',
    'GH',
]
_punctuation = "-"

valid_symbols.append("<eos>")

SAAL_SYMBOLS = valid_symbols + list(_punctuation)


@dataclass
class SaalTTSProcessor(BaseProcessor):
    cleaner_names: str = None

    positions = {
        "wave_file": 0,
        "text": 1,
        "text_norm": 2,
    }
    train_f_name: str = "metadata.csv"

    def create_items(self):
        if self.data_dir:
            with open(os.path.join(self.data_dir, self.train_f_name),
                      encoding="utf-8") as f:
                self.items = [
                    self.split_line(self.data_dir, line, "|") for line in f
                ]


    def split_line(self, data_dir, line, split):
        parts = line.strip().split(split)
        wave_file = parts[self.positions["wave_file"]]
        text_norm = parts[self.positions["text_norm"]]
        wav_path = os.path.join(data_dir, "wavs", f"{wave_file}.wav")
        speaker_name = "ljspeech"
        return text_norm, wav_path, speaker_name

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_path)
        audio = audio.astype(np.float32)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": os.path.split(wav_path)[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def setup_eos_token(self):
        return None  # because we do not use this

    def save_pretrained(self, saved_path):
        os.makedirs(saved_path, exist_ok=True)
        self._save_mapper(os.path.join(saved_path, PROCESSOR_FILE_NAME), {})

    def text_to_sequence(self, text):
        return self.symbols_to_ids(text.split() + ['<eos>'])

    def inference_text_to_seq(self, text: str):
        return self.symbols_to_ids(self.text_to_ph(text))

    def symbols_to_ids(self, symbols_list: list):
        return [self.symbol_to_id[s] for s in symbols_list]

    def text_to_ph(self, text_phones: str):
        return text_phones.split() + ['<eos>']
