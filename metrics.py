import math
import os
import loader
import numpy as np
from tqdm import trange
from music21 import *

def chord2vec(element):

    if isinstance(element, chord.Chord):

        # Extracts the MIDI pitch of each note in a chord
        pitch_list = [sub_ele.pitch.midi for sub_ele in element.notes]
        pitch_list = sorted(pitch_list)
    
    elif isinstance(element, note.Rest):

        # Four '13' to indicate that it is a 'rest'
        return [13]*4

    # Reduced MIDI pitch range
    first_note = pitch_list[0]
    pitch_list = [num-first_note for num in pitch_list]
    pitch_list = [first_note%12]+pitch_list[1:]

    vec = []

    # All notes within one octave (range 1 to 12)
    for i, element in enumerate(pitch_list):

        if element<12 and i<4:

            vec.append(element+1)

    # Padding
    vec = vec + [0]*(4-len(vec))

    return vec


def get_melody_list(melody_part):

    melody_list = []

    for element in melody_part.flat:

        if isinstance(element, note.Note):

            melody_list.append(element)

    return melody_list


def get_chord_list(chord_part):

    chord_list = []

    for element in chord_part.flat:

        if isinstance(element, chord.Chord):

            chord_list.append(element)

    return chord_list


def get_chord_vec_list(chord_list):

    chord_vec_list = []

    for chord in chord_list:
        
        chord_vec = chord2vec(chord)
        
        new_chord_vec = [chord_vec[0]]

        for n in chord_vec[1:]:
            if n!=0:
                new_chord_vec.append(n)
        
        chord_vec = new_chord_vec

        for i in range(len(chord_vec)):

            if i != 0 and chord_vec[i] == 0:

                del chord_vec[i]
                continue

            chord_vec[i] -= 1

            if i != 0: chord_vec[i] = (chord_vec[i] + chord_vec[0]) % 12

        chord_vec_list.append(chord_vec)

    return chord_vec_list


def tonal_centroid(notes):

    fifths_lookup = {9:[1.0, 0.0], 2:[math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)], 7:[math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
                     0:[0.0, 1.0], 5:[math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)], 10:[math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
                     3:[-1.0, 0.0], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 1:[math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
                     6:[0.0, -1.0], 11:[math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}
    minor_thirds_lookup = {3:[1.0, 0.0], 7:[1.0, 0.0], 11:[1.0, 0.0],
                           0:[0.0, 1.0], 4:[0.0, 1.0], 8:[0.0, 1.0],
                           1:[-1.0, 0.0], 5:[-1.0, 0.0], 9:[-1.0, 0.0],
                           2:[0.0, -1.0], 6:[0.0, -1.0], 10:[0.0, -1.0]}
    major_thirds_lookup = {0:[0.0, 1.0], 3:[0.0, 1.0], 6:[0.0, 1.0], 9:[0.0, 1.0],
                           2:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 5:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 11:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           1:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 7:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 10:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}

    fifths = [0.0, 0.0]
    minor = [0.0, 0.0]
    major = [0.0, 0.0]
    r1 =1
    r2 =1
    r3 = 0.5

    if notes:

        for note in notes:

            for i in range(2):

                fifths[i] += r1 * fifths_lookup[note][i]
                minor[i] += r2 * minor_thirds_lookup[note][i]
                major[i] += r3 * major_thirds_lookup[note][i]

        for i in range(2):

            fifths[i] /= len(notes)
            minor[i] /= len(notes)
            major[i] /= len(notes)

    return fifths + minor + major


def get_CTnCTR(melody_list, chord_list):

    chord_vec_list = get_chord_vec_list(chord_list)
    note_index = 0
    c = 0
    p = 0
    n = 0

    for chord_index in range(len(chord_list)):

        if (note_index >= len(melody_list)): break

        while melody_list[note_index].offset < chord_list[chord_index].offset:

            note_index += 1
            if (note_index >= len(melody_list)): 
                if c + n == 0: return 0
                return (c+p) / (c+n)

        while ((melody_list[note_index].offset + melody_list[note_index].quarterLength <= chord_list[chord_index].offset + chord_list[chord_index].quarterLength)
            and (melody_list[note_index].offset >= chord_list[chord_index].offset)):

            if melody_list[note_index].pitch.pitchClass in chord_vec_list[chord_index]:

                c += melody_list[note_index].quarterLength
                
            else:

                n += melody_list[note_index].quarterLength
                j = 1

                if (note_index + j >= len(melody_list)): break

                while ((melody_list[note_index + j].offset < chord_list[chord_index].offset + chord_list[chord_index].quarterLength)
                    and (melody_list[note_index + j].offset >= chord_list[chord_index].offset)):

                    if melody_list[note_index + j].pitch.pitchClass != melody_list[note_index].pitch.pitchClass:

                        if ((melody_list[note_index + j].pitch.pitchClass in chord_vec_list[chord_index])
                            and (abs(melody_list[note_index + j].pitch.pitchClass - melody_list[note_index].pitch.pitchClass <= 2))):

                            p += melody_list[note_index].quarterLength
                            break
                    j += 1

                    if (note_index + j >= len(melody_list)): break

            note_index += 1

            if (note_index >= len(melody_list)): break
    
    if c + n == 0: return 0
    return (c+p) / (c+n)


def get_PCS(melody_list, chord_list):

    chord_vec_list = get_chord_vec_list(chord_list)
    note_index = 0
    score = 0
    cnt = 0

    for chord_index in range(len(chord_list)):

        if (note_index >= len(melody_list)): break
        
        while melody_list[note_index].offset < chord_list[chord_index].offset:

            note_index += 1
            if (note_index >= len(melody_list)): 
                if cnt == 0: return 0
                return score / cnt

        while ((melody_list[note_index].offset < chord_list[chord_index].offset + chord_list[chord_index].quarterLength)
            and (melody_list[note_index].offset >= chord_list[chord_index].offset)):

            m = melody_list[note_index].pitch.pitchClass
            dur = melody_list[note_index].quarterLength

            for c in chord_vec_list[chord_index]:

                if abs(m - c) == 0 or abs(m - c) == 3 or abs(m - c) == 4 or abs(m - c) == 7 or abs(m - c) == 8 or abs(m - c) == 9 or abs(m - c) == 5:

                    if abs(m - c) == 5:

                        cnt += dur

                    else:

                        cnt += dur
                        score += dur

                else:

                    cnt += dur
                    score += -dur

            note_index += 1
            if (note_index >= len(melody_list)): break

    if cnt == 0: return 0
    return score / cnt


def get_MCTD(melody_list, chord_list):

    chord_vec_list = get_chord_vec_list(chord_list)
    note_index = 0
    score = 0
    cnt = 0

    for chord_index in range(len(chord_list)):

        if (note_index >= len(melody_list)): break
        
        while melody_list[note_index].offset < chord_list[chord_index].offset:

            note_index += 1
            
        while ((melody_list[note_index].offset < chord_list[chord_index].offset + chord_list[chord_index].quarterLength)
            and (melody_list[note_index].offset >= chord_list[chord_index].offset)):

            m = melody_list[note_index].pitch.pitchClass
            dur = melody_list[note_index].quarterLength
            
            score += np.sqrt(np.sum((np.asarray(tonal_centroid([m])) - np.asarray(tonal_centroid(chord_vec_list[chord_index])))) ** 2) * dur
            cnt += dur
            
            note_index += 1
            if (note_index >= len(melody_list)): break

    if cnt == 0: return 0
    return score / cnt


if __name__ == "__main__":

    # Initialization
    all_CTnCTR = all_PCS = all_MCTD = 0
    cnt = 0
    
    # Traverse the path
    for dirpath, dirlist, filelist in os.walk('outputs-1.0'):
        
        # Traverse the list of files
        for file_idx in trange(len(filelist)):
            
            # Ensure that suffixes in the training set are valid
            this_file = filelist[file_idx]
    
            # Read the this music file
            filename = os.path.join(dirpath, this_file)
            score = converter.parse(filename)

            soprano_part, alto_part, tenor_part, bass_part = score.parts
            melody_part, chord_part = loader.leadsheet_converter(soprano_part)

            # Read melody part and chord part as list
            chord_list = get_chord_list(chord_part)
            alto_list = get_melody_list(alto_part)
            tenor_list = get_melody_list(tenor_part)
            bass_list = get_melody_list(bass_part)

            melody_txt, beat_txt, fermata_txt = loader.melody2txt(melody_part)
            alto_txt, beat_txt, fermata_txt = loader.melody2txt(alto_part)
            tenor_txt, beat_txt, fermata_txt = loader.melody2txt(tenor_part)
            bass_txt, beat_txt, fermata_txt = loader.melody2txt(bass_part)

            # Calculate each metric
            all_CTnCTR += get_CTnCTR(alto_list, chord_list)
            all_CTnCTR += get_CTnCTR(tenor_list, chord_list)
            all_CTnCTR += get_CTnCTR(bass_list, chord_list)

            all_PCS += get_PCS(alto_list, chord_list)
            all_PCS += get_PCS(tenor_list, chord_list)
            all_PCS += get_PCS(bass_list, chord_list)

            all_MCTD += get_MCTD(alto_list, chord_list)
            all_MCTD += get_MCTD(tenor_list, chord_list)
            all_MCTD += get_MCTD(bass_list, chord_list)
            cnt += 1

    # Print the average of each metric
    print('CTnCTR = ', all_CTnCTR/(cnt*3))
    print('PCS = ', all_PCS/(cnt*3))
    print('MCTD = ', all_MCTD/(cnt*3))