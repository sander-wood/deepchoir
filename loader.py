from copy import deepcopy
import os
import pickle
import numpy as np
from music21 import *
from config import *
from tqdm import trange

def ks2gap(ks):
    
    if isinstance(ks, key.KeySignature):
        ks = ks.asKey()
        
    try:
        # Identify the tonic
        if ks.mode == 'major':
            tonic = ks.tonic

        else:
            tonic = ks.parallel.tonic
    
    except:
        return interval.Interval(0)

    # Transpose score
    gap = interval.Interval(tonic, pitch.Pitch('C'))

    return gap.semitones


def split_by_key(score):

    scores = []
    score_part = []
    ks_list = []
    ks = None
    ts = meter.TimeSignature('c')
    pre_offset = 0

    for element in score.flat:

        # If is key signature
        if isinstance(element, key.KeySignature) or isinstance(element, key.Key):

            # If is not the first key signature
            if ks!=None:

                scores.append(stream.Stream(score_part))
                ks = element
                ks_list.append(ks)
                pre_offset = ks.offset
                ks.offset = 0
                new_ts = meter.TimeSignature(ts.ratioString)
                score_part = [ks, new_ts]
            
            else:

                ks = element
                ks_list.append(ks)
                score_part.append(ks)

        # If is time signature
        elif isinstance(element, meter.TimeSignature):

            element.offset -= pre_offset
            ts = element
            score_part.append(element)
        
        else:

            element.offset -= pre_offset
            score_part.append(element)

    scores.append(stream.Stream(score_part))
    if ks_list==[]:
        ks_list = [key.KeySignature(0)]
        
    gap_list = [ks2gap(ks) for ks in ks_list]

    return scores, gap_list


def quant_score(score):
    
    for element in score.flat:
        onset = np.ceil(element.offset/0.25)*0.25

        if isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord):
            offset = np.ceil((element.offset+element.quarterLength)/0.25)*0.25
            element.quarterLength = offset - onset

        element.offset = onset

    return score


def get_filenames(input_dir):
    
    filenames = []

    # Traverse the path
    for dirpath, dirlist, filelist in os.walk(input_dir):
        # Traverse the list of files
        for this_file in filelist:
            # Ensure that suffixes in the training set are valid
            if input_dir==DATASET_PATH and os.path.splitext(this_file)[-1] not in EXTENSION:
                continue
            filename = os.path.join(dirpath, this_file)
            filenames.append(filename)
    
    return filenames


def beat_seq(ts):

    # Read time signature
    beatCount = ts.numerator
    beatDuration = 4/ts.denominator

    # Create beat sequence
    beat_sequence = [0]*beatCount*int(beatDuration/0.25)
    beat_sequence[0] += 1

    # Check if the numerator is divisible by 3 or 2
    medium = 0 

    if (ts.numerator%3)==0:
        medium = 3

    elif (ts.numerator%2)==0:
        medium = 2

    for idx in range(len(beat_sequence)):

        # Add 1 to each beat
        if idx%((beatDuration/0.25))==0:

            beat_sequence[idx] += 1
        
        # Mark medium-weight beat (at every second or third beat)
        if (medium==3 and idx%((3*beatDuration/0.25))==0) or \
            (medium==2 and idx%((2*beatDuration/0.25))==0):

            beat_sequence[idx] += 1
            
    return beat_sequence


def melody_reader(melody_part, gap):

    # Initialization
    melody_txt = []
    ts_seq = []
    beat_txt = []
    fermata_txt = []
    chord_txt = []
    chord_token = [0.]*12
    fermata_flag = False

    # Read note and meta information from melody part
    for element in melody_part.flat:
        
        if isinstance(element, note.Note):
            # midi pitch as note onset
            token = element.transpose(gap).pitch.midi

            for f in element.expressions:
                if isinstance(f, expressions.Fermata):
                    fermata_flag = True
                    break

        elif isinstance(element, note.Rest):
            # 128 as rest onset
            token = 128
            
        elif isinstance(element, chord.Chord) and not isinstance(element, harmony.ChordSymbol):
            notes = [n.transpose(gap).pitch.midi for n in element.notes]
            notes.sort()
            token = notes[-1]
            
        elif isinstance(element, harmony.ChordSymbol):
            element = element.transpose(gap)
            chord_token = [0.]*12
            for n in element.pitches:
                chord_token[n.midi%12] = 1.
            continue

        # Read the current time signature
        elif isinstance(element, meter.TimeSignature):

            ts_seq.append(element)
            continue

        else:
            continue
        
        melody_txt += [token]+[129]*(int(element.quarterLength*4)-1)
        fermata_txt += [int(fermata_flag)]*int(element.quarterLength*4)
        chord_txt += [chord_token]*int(element.quarterLength*4)
        fermata_flag = False
        
    # Initialization
    cur_cnt = 0
    pre_cnt = 0
    beat_sequence = beat_seq(meter.TimeSignature('c'))

    # create beat sequence
    if len(ts_seq)!=0:

        # Traverse time signartue sequence
        for ts in ts_seq:
            
            # Calculate current time step
            cur_cnt = ts.offset/0.25

            if cur_cnt!=0:
                
                # Fill in the previous beat sequence
                beat_txt += beat_sequence*int((cur_cnt-pre_cnt)/len(beat_sequence))

                # Complete the beat sequence
                missed_beat = int((cur_cnt-pre_cnt)%len(beat_sequence))

                if missed_beat!=0:
                    beat_txt += beat_sequence[:missed_beat]

            # Update variables
            beat_sequence = beat_seq(ts)
            pre_cnt = cur_cnt

    # Handle the last time signature
    cur_cnt = len(melody_txt)
    beat_txt += beat_sequence*int((cur_cnt-pre_cnt)/len(beat_sequence))

    # Complete the beat sequence
    missed_beat = int((cur_cnt-pre_cnt)%len(beat_sequence))

    if missed_beat!=0:
        beat_txt += beat_sequence[:missed_beat]

    return melody_txt, beat_txt, fermata_txt, chord_txt


def convert_files(filenames, fromDataset=True):

    print('\nConverting %d files...' %(len(filenames)))
    soprano_melody = []
    soprano_beat = []
    soprano_fermata = []
    soprano_chord = []
    data_corpus = []

    alto= []
    tenor = []
    bass = []
    gaps = []

    for filename_idx in trange(len(filenames)):

        # Read this music file
        filename = filenames[filename_idx]

        # Ensure that suffixes are valid
        if os.path.splitext(filename)[-1] not in EXTENSION:
            continue

        # try:
        # Read this music file
        score = converter.parse(filename)

        # Read each part
        for idx, part in enumerate(score.parts):
            
            if not fromDataset:
                original_score = deepcopy(part)
                soprano_melody = []
                soprano_beat = []
                soprano_fermata = []
                soprano_chord = []
                
            part = quant_score(part)
            splited_score, gap_list = split_by_key(part)
            gaps.append(gap_list)

            if idx==0:

                # Convert soprano
                for s_idx in range(len(splited_score)):
                    melody_part = splited_score[s_idx]
                    melody_txt, beat_txt, fermata_txt, chord_txt = melody_reader(melody_part, gap_list[s_idx])

                    soprano_melody.append(melody_txt)
                    soprano_beat.append(beat_txt)
                    soprano_fermata.append(fermata_txt)
                    soprano_chord.append(chord_txt)

            else:

                # Convert alto, tenor and bass
                for s_idx in range(len(splited_score)):
                    melody_part = splited_score[s_idx]
                    melody_txt, beat_txt, fermata_txt, chord_txt = melody_reader(melody_part, gap_list[s_idx])

                if idx==1:
                    alto.append(melody_txt)

                elif idx==2:
                    tenor.append(melody_txt)

                elif idx==3:
                    bass.append(melody_txt)

        # except:
        #     # Unable to read this music file
        #     print("Warning: Failed to read \"%s\"" %filename)
        #     continue

        if not fromDataset:
            data_corpus.append([soprano_melody, soprano_beat, soprano_fermata, soprano_chord, original_score, gaps, filename])  

    if fromDataset:
        data_corpus =  [[soprano_melody, soprano_beat, soprano_fermata, soprano_chord], alto, tenor, bass, filenames]

        with open(CORPUS_PATH, "wb") as filepath:
            pickle.dump(data_corpus, filepath)
            
    else:
        return data_corpus


if __name__ == "__main__":

    # Read encoded music information and file names
    filenames = get_filenames(input_dir=DATASET_PATH)
    convert_files(filenames)