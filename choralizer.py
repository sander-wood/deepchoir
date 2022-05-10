import os
import numpy as np
from config import *
from music21 import *
from tqdm import trange
from model import build_model
from loader import chorale_loader
from samplings import gamma_sampling
from tensorflow.python.keras.utils.np_utils import to_categorical

# use cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_chord_notes(chord_vec):
    
    chord_notes = []

    for idx, n in enumerate(chord_vec):
        if n==0:
            continue
        k = 0
        while k*12+idx<128:
            chord_notes.append(k*12+idx)
            k += 1
    
    return chord_notes


def chorale_generator(input_melody, input_beat, input_fermata, input_chord, model, seg_length=SEGMENT_LENGTH, chord_gamma=1-HARMONICITY):
    
    # Padding sequences
    song_melody = seg_length*[0] + input_melody + seg_length*[0]
    song_beat = seg_length*[0] + input_beat + seg_length*[0]
    song_fermata = seg_length*[0] + input_fermata + seg_length*[0]
    song_chord = seg_length*[[0.]*12] + input_chord + seg_length*[[0.]*12]
    song_alto = seg_length*[0]
    song_tenor = seg_length*[0]
    song_bass = seg_length*[0]
    
    for idx in range(seg_length, len(song_melody)-seg_length):

        melody_left = song_melody[idx-seg_length: idx]
        melody_right = song_melody[idx: idx+seg_length][::-1]
        beat_left = song_beat[idx-seg_length: idx]
        beat_right = song_beat[idx: idx+seg_length][::-1]
        fermata_left = song_fermata[idx-seg_length: idx]
        fermata_right = song_fermata[idx: idx+seg_length][::-1]
        chord_left = song_chord[idx-seg_length: idx]
        chord_right = song_chord[idx: idx+seg_length][::-1]
        alto = song_alto[idx-seg_length: idx]
        tenor = song_tenor[idx-seg_length: idx]
        bass = song_bass[idx-seg_length: idx]

        # one-hot
        melody_left = to_categorical(melody_left, num_classes=130)
        melody_right = to_categorical(melody_right, num_classes=130)
        beat_left = to_categorical(beat_left, num_classes=4)
        beat_right = to_categorical(beat_right, num_classes=4)
        fermata_left = to_categorical(fermata_left, num_classes=2)
        fermata_right = to_categorical(fermata_right, num_classes=2)
        alto = to_categorical(alto, num_classes=130)
        tenor = to_categorical(tenor, num_classes=130)
        bass = to_categorical(bass, num_classes=130)
        
        # concat beat, fermata and chord
        condition_left = np.concatenate((beat_left, fermata_left, chord_left), axis=-1)
        condition_right = np.concatenate((beat_right, fermata_right, chord_right), axis=-1)

        # expand one dimension
        melody_left = np.expand_dims(melody_left, axis=0)
        melody_right = np.expand_dims(melody_right, axis=0)
        condition_left = np.expand_dims(condition_left, axis=0)
        condition_right = np.expand_dims(condition_right, axis=0)
        alto = np.expand_dims(alto, axis=0)
        tenor = np.expand_dims(tenor, axis=0)
        bass = np.expand_dims(bass, axis=0)

        # predict
        predictions = model.predict([melody_left, melody_right, condition_left, condition_right, alto, tenor, bass])
        result = []
        pre_notes = [129, 129, 129]
        if song_melody[idx]!=129:
            melody_pre_note = song_melody[idx]

        for p_idx, prediction in enumerate(predictions):
            
            prediction = prediction[0]
            chord_notes = get_chord_notes(song_chord[idx])

            if song_melody[idx]==128 or melody_pre_note==128 or len(chord_notes)==0:
                prediction = gamma_sampling(prediction, [[128, 129]], [0], return_probs=True)
            
            else:
                if pre_notes[p_idx] in chord_notes:
                    prediction = gamma_sampling(prediction, [[128, 129], chord_notes+[129]], [0.5, chord_gamma], return_probs=True)

                else:
                    prediction = gamma_sampling(prediction, [[128, 129], chord_notes], [0.5, chord_gamma], return_probs=True)

                if song_beat[idx]==3:
                    predictions = gamma_sampling(prediction, [[129]], [1], return_probs=True)
            
            result.append(np.argmax(prediction))

        song_alto.append(result[0])
        song_tenor.append(result[1])
        song_bass.append(result[2])
        
        if result[0]!=129:
            pre_notes[0] = result[0]
        if result[1]!=129:
            pre_notes[1] = result[1]
        if result[2]!=129:
            pre_notes[2] = result[2]

    # Remove padding
    song_alto = song_alto[seg_length:]
    song_tenor = song_tenor[seg_length:]
    song_bass = song_bass[seg_length:]

    return [song_alto, song_tenor, song_bass]


def txt2music(txt, fermata_txt, gap, ks_list, ts_list):

    if len(ts_list)==0:
        ts_list = [meter.TimeSignature('c')]
    
    if len(ks_list)==0:
        ks_list = [key.KeySignature(0)]

    # Initialization
    notes = [ts_list[0], ks_list[0]]
    pre_element = None
    duration = 0.0
    offset = 0.0
    corrected_gap = -1*(gap.semitones)

    ks_cnt = ts_cnt = 1

    # Decode text sequences
    for element in txt+[130]:
        
        if element!=129:

            # Create new note
            if pre_element!=None:

                # If is note
                if pre_element<128:
                    new_note = note.Note(pre_element+corrected_gap)

                # If is rest
                elif pre_element==128:
                    new_note = note.Rest()
                
                if fermata_txt[int(offset/0.25)]==1 and last_note_is_fermata==False:
                    new_note.expressions.append(expressions.Fermata())
                    last_note_is_fermata = True
                
                elif fermata_txt[int(offset/0.25)]!=1:
                    last_note_is_fermata = False

                new_note.quarterLength = duration
                new_note.offset = offset
                notes.append(new_note)
            
            # Updata offset, duration and save the element
            offset += duration
            duration = 0.25
            pre_element = element

            if ks_cnt<len(ks_list) and offset>=ks_list[ks_cnt].offset:
                notes.append(ks_list[ks_cnt])
                ks_cnt += 1

            if ts_cnt<len(ts_list) and offset>=ts_list[ts_cnt].offset:
                notes.append(ts_list[ts_cnt])
                ts_cnt += 1
        
        else:
             
            # Updata duration
            duration += 0.25

    return notes


def export_music(melody, chorale_list, fermata_txt, gap, filename):

    ks_list = []
    ts_list = []
    filename = '.'.join(filename.split('.')[:-1])

    # Get meta information
    for element in melody.flat:

        if isinstance(element, meter.TimeSignature):
            ts_list.append(element)
            
        if isinstance(element, key.KeySignature):
            ks_list.append(element)

    # Compose four parts
    new_score = [melody.parts[0]]

    for i in range(3):
        new_part = stream.Part(txt2music(chorale_list[i], fermata_txt, gap, ks_list, ts_list))
        new_part = new_part.transpose(interval.Interval(0))
        new_score.append(new_part)

    # Save as mxl
    new_score = stream.Stream(new_score)

    if WATER_MARK:
        meta = metadata.Metadata()
        meta.title = filename
        meta.composer = "Choralized by DeepChoir"
        new_score.insert(0,meta)
        
    new_score.write('mxl', fp=OUTPUTS_PATH+'/'+filename+'.mxl')


if __name__ == '__main__':
    
    # Load model
    model = build_model(weights_path=WEIGHTS_PATH, training=False)
    melody_data, beat_data, fermata_data, chord_data, melodies, gaps, scores_len_list, filenames = chorale_loader(path=INPUTS_PATH)

    start_idx = 0
    end_idx = 0

    # Process each score
    for idx in trange(len(scores_len_list)):

        end_idx += scores_len_list[idx]
        chorale_list = []
        fermata_txt = []

        for sub_idx, input_melody in enumerate(melody_data[start_idx: end_idx]):
            chorale_list += chorale_generator(input_melody, beat_data[start_idx+sub_idx], fermata_data[start_idx+sub_idx], chord_data[start_idx+sub_idx], model)
            fermata_txt += fermata_data[start_idx+sub_idx]
        
        export_music(melodies[idx], chorale_list, fermata_txt, gaps[idx], filenames[idx])

        start_idx = end_idx
      