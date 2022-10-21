import os
import numpy as np
from config import *
from music21 import *
from tqdm import trange
from model import build_model
from loader import get_filenames, convert_files
from samplings import gamma_sampling
from tensorflow.python.keras.utils.np_utils import to_categorical

# use cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_chord_tones(chord_vec):
    
    chord_tones = []

    for idx, n in enumerate(chord_vec):
        if n==0:
            continue
        k = 0
        while k*12+idx<128:
            chord_tones.append(k*12+idx)
            k += 1
    
    return chord_tones


def chorale_generator(input_melody, input_beat, input_fermata, input_chord, model, gap, seg_length=SEGMENT_LENGTH, chord_gamma=1-HARMONICITY, onset_gamma=HOMOPHONICITY):
    
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
            
            if song_melody[idx]==129:
                prediction = gamma_sampling(prediction, [[129]], [1-onset_gamma], return_probs=True)
            
            else:
                prediction = gamma_sampling(prediction, [[129]], [onset_gamma], return_probs=True)

            chord_tones = get_chord_tones(song_chord[idx])

            if song_melody[idx]==128 or melody_pre_note==128 or len(chord_tones)==0:
                prediction = gamma_sampling(prediction, [[128, 129]], [0], return_probs=True)
            
            else:
                if pre_notes[p_idx] in chord_tones:
                    chord_tones = chord_tones+[129]
                    
                prediction = gamma_sampling(prediction, [[128, 129], chord_tones], [0.5, chord_gamma], return_probs=True)

                if song_beat[idx]==3 and song_melody[idx]!=129:
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
    song_alto = np.array(song_alto[seg_length:])
    song_tenor = np.array(song_tenor[seg_length:])
    song_bass = np.array(song_bass[seg_length:])

    song_alto = np.where(song_alto<128, song_alto-gap, song_alto)
    song_tenor = np.where(song_tenor<128, song_tenor-gap, song_tenor)
    song_bass = np.where(song_bass<128, song_bass-gap, song_bass)

    return [song_alto.tolist(), song_tenor.tolist(), song_bass.tolist()]


def txt2music(txt, fermata_txt, ks_list, ts_list):

    if len(ts_list)==0:
        ts_list = [meter.TimeSignature('c')]
    
    if len(ks_list)==0:
        ks_list = [key.KeySignature(0)]

    # Initialization
    notes = [ts_list[0], ks_list[0]]
    pre_element = None
    duration = 0.0
    offset = 0.0

    ks_cnt = ts_cnt = 1

    # Decode text sequences
    for element in txt+[130]:
        
        if element!=129:

            # Create new note
            if pre_element!=None:

                # If is note
                if pre_element<128:
                    new_note = note.Note(pre_element)

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


def export_music(melody, chorale_list, fermata_txt, filename, keep_chord=KEEP_CHORD):

    ks_list = []
    ts_list = []
    new_melody = []
    filename = os.path.basename(filename)
    filename = '.'.join(filename.split('.')[:-1])

    # Get meta information
    for element in melody.flat:

        if isinstance(element, meter.TimeSignature):
            ts_list.append(element)
            
        if isinstance(element, key.KeySignature):
            ks_list.append(element)

        if not isinstance(element, harmony.ChordSymbol):
            new_melody.append(element)

    # Compose four parts
    if keep_chord:
        new_score = [melody]
    
    else:
        new_score = [stream.Part(new_melody)]

    for i in range(3):
        new_part = stream.Part(txt2music(chorale_list[i], fermata_txt, ks_list, ts_list))
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
    filenames = get_filenames(INPUTS_PATH)
    data_corpus = convert_files(filenames, fromDataset=False)

    # Process each score
    for idx in trange(len(data_corpus)):

        chorale_list = []
        fermata_txt = []
        song_data = data_corpus[idx]
        melody_score = song_data[4]
        filename = song_data[6]

        for part_idx in range(len(song_data[0])):
            input_melody = song_data[0][part_idx]
            input_beat = song_data[1][part_idx]
            input_fermata = song_data[2][part_idx]
            input_chord = song_data[3][part_idx]
            gap = song_data[5][part_idx]
            fermata_txt += input_fermata
            chorale_list.append(chorale_generator(input_melody, input_beat, input_fermata, input_chord, model, gap))
        
        alto_list = [n for chorale in chorale_list for n in chorale[0]]
        tenor_list = [n for chorale in chorale_list for n in chorale[1]]
        bass_list = [n for chorale in chorale_list for n in chorale[2]]

        export_music(melody_score, [alto_list, tenor_list, bass_list], fermata_txt, filename)
