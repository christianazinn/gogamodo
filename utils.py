import json
import os
import subprocess
import numpy as np
from configparser import ConfigParser
from collections import Counter
import music21
import symusic
from symusic.types import Score


def get_mtg_tags(embeddings, tag_model, tag_json, max_num_tags=5, tag_threshold=0.01):
    # TODO stop opening me every run
    with open(tag_json, "r") as json_file:
        metadata = json.load(json_file)
    # TODO handle TypeError: Error cannot convert argument LIST_EMPTY to MATRIX_REAL
    predictions = tag_model(embeddings)
    mean_act = np.mean(predictions, 0)
    ind = np.argpartition(mean_act, -max_num_tags)[-max_num_tags:] if max_num_tags is not None else np.arange(len(mean_act))
    tags = []
    confidence_score = []
    for i in ind:
        # print(metadata['classes'][i] + str(mean_act[i]))
        if tag_threshold is None or mean_act[i] > tag_threshold:
            tags.append(metadata["classes"][i])
            confidence_score.append(mean_act[i])
    ind = np.argsort(-np.array(confidence_score))
    tags = [tags[i] for i in ind]
    confidence_score = np.round((np.array(confidence_score)[ind]).tolist(), 4).tolist()

    return tags, confidence_score


def get_final_inst_list(score: Score, instrumentmap: dict):
    try:
        instruments = []
        for track in score.tracks:
            if not track.empty():
                instruments.append(128 if track.is_drum else track.program)

        # Filter to top 5 unique instruments
        unique_instruments = list(dict.fromkeys(instruments))
        fulllist = unique_instruments[: min(5, len(unique_instruments))]
    except Exception:
        fulllist = []

    # instrument mapping and summary
    out_inst_list = []
    for inst in fulllist:
        out_inst_list.append(instrumentmap[inst][3])
    
    # instruments summary - only add one instance of each instrument, then keep top 5
    out_inst_sum_list = []
    for rr in out_inst_list:
        if rr not in out_inst_sum_list:
            out_inst_sum_list.append(rr)
    how_many = np.min((5, len(out_inst_sum_list)))
    out_inst_sum_list = out_inst_sum_list[0:how_many]

    return {
        "mapped_instruments_summary": out_inst_sum_list,
        "mapped_instruments": out_inst_list,
        "sorted_instruments": fulllist,
    }


def get_initial_key(file: str):
    # still use music21 for this because it's more reliable than symusic
    # (and I don't know how to read a symusic KeySignature)
    try:
        midi = music21.converter.parse(file)
        key = str(midi.analyze("key"))
    except Exception:
        key = None

    if key is not None and "-" in key:
        key = key.replace("-", "b")

    return {"key": key}


def get_initial_time_signature(score: Score):
    try:
        time_sig = score.time_signatures[0]
        time_signature = str(time_sig.numerator) + "/" + str(time_sig.denominator)
    except Exception:
        time_signature = None

    return {"time_signature": time_signature}


def get_tempo_duration(score: Score):
    try:
        bpm = score.tempos[0].qpm
        bpm = int(np.round(bpm))
    except Exception:
        bpm = None

    if bpm is not None:
        tempo_marks = np.array((40, 60, 70, 90, 110, 140, 160, 210))
        tempo_caps = [
            "Grave",
            "Largo",
            "Adagio",
            "Andante",
            "Moderato",
            "Allegro",
            "Vivace",
            "Presto",
            "Prestissimo",
        ]
        index = np.sum(bpm > tempo_marks)
        tempo_cap = tempo_caps[index]
    else:
        tempo_cap = None

    # alternative: tempo_marks = np.array((80, 120, 160))
    #              tempo_caps = ["Slow", "Moderate tempo", "Fast", "Very fast"]

    try:
        ticks = score.end() - score.start()
        duration = ticks / score.tempos[0].qpm * 60  # assumes ttype="quarter"
        duration = int(np.round(duration))
    except Exception:
        duration = None

    if duration is not None:
        dur_marks = np.array((30, 120, 300))
        # TODO modify me
        dur_caps = ["Short fragment", "Short song", "Song", "Long piece"]
        index = np.int32(np.sum(duration > dur_marks))
        dur_cap = dur_caps[index]
    else:
        dur_cap = None

    return {
        "tempo": [bpm, tempo_cap],
        "duration": [duration, dur_cap],
    }


def find_most_repeating_sequence(chords_list, sequence_length):
    sequences = [
        tuple(chords_list[i : i + sequence_length])
        for i in range(len(chords_list) - sequence_length + 1)
    ]
    delete_index = []
    for i, seqs in enumerate(sequences):
        if seqs[0] == seqs[-1]:
            delete_index.append(i)
    for i in reversed(delete_index):
        sequences.pop(i)
    sequence_counts = Counter(sequences)
    try:
        most_common_sequence, count = sequence_counts.most_common(1)[0]
        return most_common_sequence, count
    except Exception:
        # print(f"Exception at fmrs: {e}")
        return None, 0


def give_me_final_seq(chords):
    sequence_3, count_3 = find_most_repeating_sequence(chords, 3)
    sequence_4, count_4 = find_most_repeating_sequence(chords, 4)
    sequence_5, count_5 = find_most_repeating_sequence(chords, 5)
    total_count = count_3 + count_4 + count_5
    if count_5 > 0.25 * (total_count):
        if count_5 > 0.79 * count_4:
            return sequence_5, count_5
    if count_4 > 0.3 * (total_count):
        if count_4 > 0.79 * count_3:
            return sequence_4, count_4
    if count_3 == 0:
        if count_4 == 0:
            if count_5 == 0:
                return None, 0  # everything is 0
            else:
                return sequence_5, count_5
        else:
            return sequence_4, count_4
    else:
        return sequence_3, count_3


# -------------------------------------------------


def extract_chords(audio_file, chord_estimator):
    chords = chord_estimator.extract(audio_file)
    chords_out = [(x.chord, x.timestamp) for x in chords[1:-1]]
    # chord summary
    ch_name = []
    ch_time = []
    for ch in chords_out:
        ch_name.append(ch[0])
        ch_time.append(ch[1])
    if len(ch_name) < 3:
        final_seq = ch_name
        final_count = 1
    else:
        final_seq, final_count = give_me_final_seq(ch_name)
    if final_seq is not None:
        if len(final_seq) == 4:
            if final_seq[0] == final_seq[2] and final_seq[1] == final_seq[3]:
                final_seq = final_seq[0:2]
    return {
        "chord_summary": [final_seq, final_count],
        # "chords": chords_out,
        "audio_file": audio_file,
    }


def extract_midi_features(file: str, instrumentmap: dict):
    score = symusic.Score(file, ttype="quarter")
    i = get_final_inst_list(score, instrumentmap)
    k = get_initial_key(file)
    t = get_initial_time_signature(score)
    b = get_tempo_duration(score)
    return i | k | t | b

class DotDict:
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

class DotConfigParser(ConfigParser):
    def parse_args(self):
        result = {}
        for section in self.sections():
            for key, value in self.items(section):
                if value.lower() in ('true', 'false'):
                    result[key] = self.getboolean(section, key)
                else:
                    try:
                        result[key] = self.getint(section, key)
                    except ValueError:
                        try:
                            result[key] = self.getfloat(section, key)
                        except ValueError:
                            result[key] = value
                            
        return DotDict(result)