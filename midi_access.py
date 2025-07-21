

import os
import pretty_midi
import numpy as np
from collections import Counter


def load_midi_files_by_artist(base_dir):
    artist_midi = {}
    for artist in os.listdir(base_dir):
        artist_path = os.path.join(base_dir, artist)
        if os.path.isdir(artist_path):
            midi_files = [f for f in os.listdir(artist_path) if f.endswith('.mid') or f.endswith('.midi')]
            midi_data_list = []
            for midi_file in midi_files:
                file_path = os.path.join(artist_path, midi_file)
                try:
                    midi_data = pretty_midi.PrettyMIDI(file_path)
                    midi_data_list.append(midi_data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            artist_midi[artist] = midi_data_list
    return artist_midi


def get_pitch_class_histogram(midi_data):
    pitch_classes = [note.pitch % 12 for inst in midi_data.instruments for note in inst.notes]
    hist, _ = np.histogram(pitch_classes, bins=np.arange(13), density=True)
    return hist

def get_ioi_histogram(midi_data, bins=20, range_max=2.0):
    onsets = [note.start for inst in midi_data.instruments for note in inst.notes]
    onsets = np.sort(onsets)
    iois = np.diff(onsets)
    return hist

def extract_chords(midi_data, time_window=0.05):
    notes = []
    for inst in midi_data.instruments:
        for note in inst.notes:
            notes.append((note.start, note.pitch))
    notes.sort()
    chords = []
    current_chord = []
    last_onset = None
    for onset, pitch in notes:
        if last_onset is None or abs(onset - last_onset) < time_window:
            current_chord.append(pitch % 12)
        else:
            if len(current_chord) > 1:
                chords.append(tuple(sorted(set(current_chord))))
            current_chord = [pitch % 12]
        last_onset = onset
    if len(current_chord) > 1:
        chords.append(tuple(sorted(set(current_chord))))
    return chords

# Define top chords globally (from all artists or a sample)
def get_top_chords(artist_midi_dict, top_n=20):
    chord_counter = Counter()
    for midi_list in artist_midi_dict.values():
        for midi in midi_list:
            chord_counter.update(extract_chords(midi))
    return [chord for chord, _ in chord_counter.most_common(top_n)]


def get_chord_histogram(midi_data, top_chords):
    chords = extract_chords(midi_data)
    chord_counter = Counter(chords)
    hist = np.array([chord_counter.get(chord, 0) for chord in top_chords], dtype=np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist