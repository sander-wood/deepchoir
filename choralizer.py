import math
import numpy as np
from config import *
from music21 import *
from model import build_model
from loader import chorale_loader
from tensorflow.python.keras.utils.np_utils import to_categorical

def sample(prediction, token_list=[], density_list=[]):
    
    modified_list = []
    frozen_list = []
    frozen_prob = 0

    if not isinstance(density_list, list):
        density_list = [density_list]
        
    if len(density_list)==1 or not isinstance(token_list, list):
        token_list = [token_list]

    for idx, d in enumerate(density_list):

        for token in modified_list:
            frozen_prob += prediction[token]
            frozen_list.append(token)

        prob_sum = 0
        modified_list = []

        try:
            for token in token_list[idx]:
                prob_sum += prediction[token]
                modified_list.append(token)
        
        except:
            prob_sum += prediction[token_list[idx]]
            modified_list.append(token_list[idx])

        for token in modified_list:
            if token in frozen_list:
                frozen_list.remove(token)
                frozen_prob -= prediction[token]

        rest_prob = 1-prob_sum-frozen_prob
        old_prob = prob_sum

        if d==0:
            prob_sum = (1-frozen_prob)*int(prob_sum!=0)
        elif d==1:
            prob_sum = int(prob_sum==1)
        else:
            prob_sum = (1-frozen_prob)*prob_sum**math.tan(math.pi*d/2)

        for token in modified_list:
            prediction[token] = prob_sum*(prediction[token]/old_prob)

        for p_idx in range(len(prediction)):
            if (p_idx not in modified_list) and (p_idx not in frozen_list):
                prediction[p_idx] += (old_prob-prob_sum)*(prediction[p_idx]/rest_prob)
        
        for p_idx in range(len(prediction)):
            if not (0<=prediction[p_idx] and prediction[p_idx]<=1):
                prediction[p_idx] = 0
        
        pred_sum = prediction.sum()

        if pred_sum!=1:
            for p_idx in range(len(prediction)):
                prediction[p_idx] = prediction[p_idx]/pred_sum

    return np.argmax(prediction)


def chorale_generator(input_melody, input_beat, input_fermata, input_chord, model, seg_length=SEGMENT_LENGTH, polyphonicity=POLYPHONICITY, harmonicity=HARMONICITY):
    
    # Padding sequences
    missed_num = seg_length-len(input_melody)%seg_length

    if missed_num!=seg_length:

        song_melody = input_melody+[0]*missed_num
        song_beat = input_beat+[0]*missed_num
        song_beat = to_categorical(song_beat, num_classes=4).tolist()
        song_fermata = input_fermata+[0]*missed_num
        song_chord = input_chord+[[0]*12]*missed_num
    
    else:

        song_melody = input_melody
        song_beat = input_beat
        song_beat = to_categorical(song_beat, num_classes=4).tolist()
        song_fermata = input_fermata
        song_chord = input_chord
    
    song_condition = [[float(song_fermata[n_idx])]+song_beat[n_idx]+song_chord[n_idx] for n_idx in range(len(song_melody))]
    song_condition = np.array(song_condition).reshape(int(len(song_condition)/seg_length), seg_length, 17)
    song_melody = to_categorical(song_melody, num_classes=131).reshape(int(len(song_melody)/seg_length), seg_length, 131)
        
    # Predict the rest three parts
    net_output = np.array(model.predict(x=[song_condition, song_melody]))
    net_output = net_output.reshape(3, net_output.shape[1]*net_output.shape[2], 131)
    new_output = []
    
    # Sample each part
    for part_output in net_output:

        new_part_output = []

        # Sample each time step
        for idx, prob in enumerate(part_output):

            # Find all chord tones
            note_list = []

            for n_idx, n in enumerate(song_chord[idx]):
                if n==1:
                    for group_num in range(11):
                        n_pitch = 1+n_idx+(group_num*12)
                        if n_pitch<=128:
                            note_list.append(n_pitch)
                        else:
                            break
            
            # Rest
            if len(note_list)==0 or input_melody[min(idx, len(input_melody)-1)]==129:
                new_part_output.append(sample(prob, [129, 130], 0))
                continue

            # Set density based on polyphonicity
            if input_beat[min(idx, len(input_beat)-1)]==3:
                d = 1
            
            elif input_melody[min(idx, len(input_melody)-1)]!=130:
                d = 1-polyphonicity

            else:
                d = polyphonicity

            new_part_output.append(sample(prob, [130, note_list], [d, 1-harmonicity]))
        
        new_output.append(new_part_output)

    # Remove padding
    if missed_num!=seg_length:
        for part_idx in range(3):
            new_output[part_idx] = new_output[part_idx][:-missed_num]
    
    return new_output


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
    for element in txt+[131]:
        
        if element!=130:

            # Create new note
            if pre_element!=None:

                # If is note
                if pre_element<129:

                    new_note = note.Note(pre_element-1+corrected_gap)

                # If is rest
                elif pre_element==129:

                    new_note = note.Rest()
                
                if fermata_txt[int(offset/0.25)]==1:
                    new_note.expressions.append(expressions.Fermata())

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
        meta.title = filename.split('.')[-2]
        meta.composer = "Choralized by DeepChoir"
        new_score.insert(0,meta)
        
    new_score.write('mxl', fp=OUTPUTS_PATH+'/'+filename.split('.')[-2]+'.mxl')


if __name__ == '__main__':
    
    # Load model
    model = build_model(weights_path=WEIGHTS_PATH)
    melody_data, beat_data, fermata_data, chord_data, melodies, gaps, scores_len_list, filenames = chorale_loader(path=INPUTS_PATH)

    start_idx = 0
    end_idx = 0

    # Process each score
    for idx, scores_len in enumerate(scores_len_list):

        end_idx += scores_len
        chorale_list = []
        fermata_txt = []

        for sub_idx, input_melody in enumerate(melody_data[start_idx: end_idx]):
            chorale_list += chorale_generator(input_melody, beat_data[start_idx+sub_idx], fermata_data[start_idx+sub_idx], chord_data[start_idx+sub_idx], model)
            fermata_txt += fermata_data[start_idx+sub_idx]
        
        export_music(melodies[idx], chorale_list, fermata_txt, gaps[idx], filenames[idx])

        start_idx = end_idx
      
