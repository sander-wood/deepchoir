import os
import pickle
import numpy as np
from music21 import *
from config import *
from tqdm import trange

def part(score):

    try:
        score = score.parts[0]

    except:
        score = score
    
    return score


def leadsheet_converter(score):

    # Initialization
    melody_part = []
    chord_part = [] 
    chord_list = []
    
    # Read lead sheet
    for element in score.flat:
        
        # If is ChordSymbol
        if isinstance(element, harmony.ChordSymbol):
            chord_list.append(element)

        else:
            melody_part.append(element)

    # If no chord at the beginning
    if chord_list[0].offset!=0:

        first_rest = note.Rest()
        first_rest.quarterLength = chord_list[0].offset
        chord_part.append(first_rest)

    # Instantiated chords
    for idx in range(1, len(chord_list)):

        new_chord = chord.Chord(chord_list[idx-1].notes)
        new_chord.offset = chord_list[idx-1].offset
        new_chord.quarterLength = chord_list[idx].offset-chord_list[idx-1].offset
        chord_part.append(new_chord)
    
    # Add the last chord
    new_chord = chord.Chord(chord_list[-1].notes)
    new_chord.offset = chord_list[-1].offset
    new_chord.quarterLength = melody_part[-1].offset-chord_list[idx].offset
    chord_part.append(new_chord)

    return stream.Part(melody_part).flat, stream.Part(chord_part).flat


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


def norm_pos(pos):

    # Calculate extra position
    extra_pos = pos%0.25

    # If greater than 0
    if extra_pos>0:
        pos = pos-extra_pos+0.25
    
    return pos


def norm_duration(element):

    # Read the duration
    note_duration = element.quarterLength
    
    # Calculate positions of note
    note_start = element.offset
    note_end = note_start + note_duration

    # Regularized position and duration
    note_start = norm_pos(note_start)
    note_end = norm_pos(note_end)
    note_duration = note_end-note_start

    return note_duration


def melody2txt(melody_part):

    # Initialization
    pre_ele = None
    melody_txt = []
    ts_seq = []
    beat_txt = []
    fermata_txt = []

    # Read note and meta information from melody part
    for element in melody_part.flat:
        
        if isinstance(element, note.Note) or isinstance(element, note.Rest):
            
            # Read the regularized duration
            note_duration = norm_duration(element)
            
            # Skip if the duration is equal to 0 after regularization
            if note_duration==0:
                continue
            
            fermata_flag = False

            # '129' for holding
            note_steps = int(note_duration/0.25)

            # Reads the MIDI pitch of a note (0~127)
            if isinstance(element, note.Note):
                melody_txt.append(element.pitch.midi)

                for f in element.expressions:
                    if isinstance(f, expressions.Fermata):
                        fermata_txt += [1]*(note_steps)
                        fermata_flag = True
                        break

            # '128' for rest
            elif isinstance(element, note.Rest):

                # Merge adjacent rests
                if isinstance(pre_ele, note.Rest):
                    melody_txt.append(129)
                
                else:
                    melody_txt.append(128)
            
            melody_txt += [129]*(note_steps-1)

            if fermata_flag==False:
                fermata_txt += [0]*(note_steps)

            # Save current note
            pre_ele = element

        # Read the current time signature
        elif isinstance(element, meter.TimeSignature):

            ts_seq.append(element)
    
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

    return melody_txt, beat_txt, fermata_txt


def chord2txt(chord_part):
    
    chord_txt = []

    for element in chord_part.flat:
        chord_vec = [0.]*12

        if isinstance(element, chord.Chord):

            # Read the regularized duration
            note_duration = norm_duration(element)
            
            # Skip if the duration is equal to 0 after regularization
            if note_duration==0:
                continue

            # '130' for holding
            note_steps = int(note_duration/0.25)

            for n in element.notes:
                chord_vec[n.pitch.midi%12] += 1
        
            chord_txt += [chord_vec]*int(note_steps)
        
        elif isinstance(element, note.Rest):

            # Read the regularized duration
            note_duration = norm_duration(element)
            
            # Skip if the duration is equal to 0 after regularization
            if note_duration==0:
                continue

            # '130' for holding
            note_steps = int(note_duration/0.25)
            chord_txt += [chord_vec]*int(note_steps)

    return chord_txt


def transpose(score):

    # Set default interval, key signature and tempo
    gap = interval.Interval(0)
    ks = key.KeySignature(0)

    for element in score.flat:
        
        # Found key signature
        if isinstance(element, key.KeySignature) or isinstance(element, key.Key):

            if isinstance(element, key.KeySignature):
                ks = element.asKey()
            
            else:
                ks = element

            # Identify the tonic
            if ks.mode == 'major':
                tonic = ks.tonic

            else:
                tonic = ks.parallel.tonic

            # Transpose score
            gap = interval.Interval(tonic, pitch.Pitch('C'))
            score = score.transpose(gap)

            break
        
        # No key signature found
        elif isinstance(element, note.Note) or \
             isinstance(element, note.Rest) or \
             isinstance(element, chord.Chord):
            break

        else:

            continue

    return score, gap


def key_split(score):

    scores = []
    score_part = []
    ks = None
    ts = None
    pre_offset = 0

    for element in part(score).flat:

        # If is key signature
        if isinstance(element, key.KeySignature) or isinstance(element, key.Key):

            # If is not the first key signature
            if ks!=None:

                scores.append(stream.Stream(score_part))
                ks = element
                pre_offset = ks.offset
                ks.offset = 0
                new_ts = meter.TimeSignature(ts.ratioString)
                score_part = [ks, new_ts]
            
            else:

                ks = element
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

    return scores


def chorale_loader(path=DATASET_PATH):

    soprano_melody = []
    soprano_beat = []
    soprano_fermata = []
    soprano_chord = []

    alto= []
    tenor = []
    bass = []

    melodies = []
    gaps = []
    filenames = []
    scores_len_list = []
    np.random.seed(0)

    # Traverse the path
    for dirpath, dirlist, filelist in os.walk(path):
        
        # Traverse the list of files
        for file_idx in trange(len(filelist)):

            this_file = filelist[file_idx]
            filename = os.path.join(dirpath, this_file)

            # Ensure that suffixes are valid
            if os.path.splitext(filename)[-1] not in EXTENSION:
                continue

            try:
                # Read this music file
                score = converter.parse(filename)
                melodies.append(score)
                filenames.append(this_file)

                if path!='dataset':
                    scores = key_split(score)
                    scores_len_list.append(len(scores))

                    for sub_score in scores:

                        # Converte music to text data
                        sub_score, gap= transpose(sub_score)
                        gaps.append(gap)
                        melody_part, chord_part = leadsheet_converter(sub_score)
                        melody_txt, beat_txt, fermata_txt = melody2txt(melody_part)
                        chord_txt = chord2txt(chord_part)

                        soprano_melody.append(melody_txt)
                        soprano_beat.append(beat_txt)
                        soprano_fermata.append(fermata_txt)
                        soprano_chord.append(chord_txt)
                
                else:
                
                    score, gap= transpose(score)

                    # Read each part
                    for idx, part in enumerate(score.parts):
                        
                        if idx==0:

                            # Convert soprano
                            melody_part, chord_part = leadsheet_converter(part)
                            melody_txt, beat_txt, fermata_txt = melody2txt(melody_part)
                            chord_txt = chord2txt(chord_part)

                            soprano_melody.append(melody_txt)
                            soprano_beat.append(beat_txt)
                            soprano_fermata.append(fermata_txt)
                            soprano_chord.append(chord_txt)

                        else:

                            # Convert alto, tenor and bass
                            melody_txt, beat_txt, fermata_txt = melody2txt(part)

                            if idx==1:
                                alto.append(melody_txt)

                            elif idx==2:
                                tenor.append(melody_txt)

                            elif idx==3:
                                bass.append(melody_txt)
            
            except:
                # Unable to read this music file
                print("Warning: Failed to read \"%s\"" %filename)
                continue
            

    if path=='dataset':
        return [[soprano_melody, soprano_beat, soprano_fermata, soprano_chord], alto, tenor, bass, filenames]

    else:
        return soprano_melody, soprano_beat, soprano_fermata, soprano_chord, melodies, gaps, scores_len_list, filenames


if __name__ == "__main__":

    # Read encoded music information and file names
    chorale_corpus = chorale_loader()

    # Save as corpus
    with open(CORPUS_PATH, "wb") as filepath:
        pickle.dump(chorale_corpus,filepath)