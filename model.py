import pickle
import os
import numpy as np
from config import *
from keras.layers import Input
from keras.layers  import concatenate
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical


def create_training_data(corpus_path=CORPUS_PATH, seg_length=SEGMENT_LENGTH, val_ratio=VAL_RATIO):
    
    # Load corpus
    with open(corpus_path, "rb") as filepath:
        corpus = pickle.load(filepath)

    # Inputs and targets for the training set
    input_condition = []
    input_melody = []
    output_alto = []
    output_tenor = []
    output_bass = []

    # Inputs and targets for the validation set
    val_input_condition = []
    val_input_melody = []
    val_output_alto = []
    val_output_tenor = []
    val_output_bass = []

    # Load txt data
    soprano_melody = corpus[0][0]
    soprano_beat = corpus[0][1]
    soprano_fermata = corpus[0][2]
    soprano_chord = corpus[0][3]

    alto_melody = corpus[1]
    tenor_melody= corpus[2]
    bass_melody = corpus[3]

    beat_data = soprano_beat
    fermata_data = soprano_fermata
    chord_data = soprano_chord

    cnt = 0
    np.random.seed(0)

    # Process each melody sequence in the corpus
    
    for song_idx in range(len(soprano_melody)):

        # Randomly assigned to the training or validation set with the probability
        if np.random.rand()>val_ratio:
            train_or_val = 'train'
        
        else:
            train_or_val = 'val'

        # Padding sequences
        missed_num = seg_length-len(soprano_melody[song_idx])%seg_length

        if missed_num!=seg_length:

            song_melody = soprano_melody[song_idx]+[0]*missed_num
            song_beat = beat_data[song_idx]+[0]*missed_num
            song_beat = to_categorical(song_beat, num_classes=4).tolist()
            song_fermata = fermata_data[song_idx]+[0]*missed_num
            song_chord = chord_data[song_idx]+[[0]*12]*missed_num

            song_alto = alto_melody[song_idx]+[0]*missed_num
            song_tenor = tenor_melody[song_idx]+[0]*missed_num
            song_bass = bass_melody[song_idx]+[0]*missed_num
        
        else:

            song_melody = soprano_melody[song_idx]
            song_beat = beat_data[song_idx]
            song_beat = to_categorical(song_beat, num_classes=4).tolist()
            song_fermata = fermata_data[song_idx]
            song_chord = chord_data[song_idx]

            song_alto = alto_melody[song_idx]
            song_tenor = tenor_melody[song_idx]
            song_bass = bass_melody[song_idx]
        
        # Create condition sequence
        song_condition = [[float(song_fermata[n_idx])]+song_beat[n_idx]+song_chord[n_idx] for n_idx in range(len(song_melody))]

        # Create pairs
        for idx in range(len(song_melody)-seg_length+1):

            melody = song_melody[idx: idx+seg_length]
            condition = song_condition[idx: idx+seg_length]

            alto = song_alto[idx: idx+seg_length]
            tenor = song_tenor[idx: idx+seg_length]
            bass = song_bass[idx: idx+seg_length]

            if train_or_val=='train':
                input_condition.append(condition)
                input_melody.append(melody)
                output_alto.append(alto)
                output_tenor.append(tenor)
                output_bass.append(bass)

            else:
                val_input_condition.append(condition)
                val_input_melody.append(melody)
                val_output_alto.append(alto)
                val_output_tenor.append(tenor)
                val_output_bass.append(bass)

            cnt += 1

    print("Successfully read %d samples" %(cnt))
    input_condition = np.array(input_condition).reshape(len(input_condition), seg_length, 17)
    input_melody = to_categorical(input_melody, num_classes=131)
    output_alto = to_categorical(output_alto, num_classes=131)
    output_tenor = to_categorical(output_tenor, num_classes=131)
    output_bass = to_categorical(output_bass, num_classes=131)
    
    val_input_condition = np.array(val_input_condition).reshape(len(val_input_condition), seg_length, 17)
    val_input_melody = to_categorical(val_input_melody, num_classes=131)
    val_output_alto = to_categorical(val_output_alto, num_classes=131)
    val_output_tenor = to_categorical(val_output_tenor, num_classes=131)
    val_output_bass = to_categorical(val_output_bass, num_classes=131)
    
    return (input_condition, input_melody, output_alto, output_tenor, output_bass), \
           (val_input_condition, val_input_melody, val_output_alto, val_output_tenor, val_output_bass)


def build_model(rnn_size=RNN_SIZE, num_layers=NUM_LAYERS, seg_length=SEGMENT_LENGTH, dropout=DROPOUT, weights_path=None):

    # Create input layer of melody encoder
    input_note = Input(shape=(seg_length, 131), 
                        name='input_note')
    note = TimeDistributed(Dense(131, activation='tanh'))(input_note)

    # Create input layer of condition encoder
    input_condition = Input(shape=(seg_length, 17), 
                        name='input_condition')
    condition = TimeDistributed(Dense(17, activation='tanh'))(input_condition)

    # Create the hidden layer of melody encoder
    for idx in range(num_layers):
        
        note = Bidirectional(LSTM(units=rnn_size, 
                                    return_sequences=True,
                                    name='note_'+str(idx+1)))(note)
        note = TimeDistributed(Dense(units=rnn_size, activation='tanh'))(note)
        note = Dropout(dropout)(note)

    # Create the hidden layer of condition encoder
    for idx in range(num_layers):
        
        condition = Bidirectional(LSTM(units=rnn_size, 
                                    return_sequences=True,
                                    name='condition_'+str(idx+1)))(condition)
        condition = TimeDistributed(Dense(units=rnn_size, activation='tanh'))(condition)
        condition = Dropout(dropout)(condition)

    merge = concatenate(
        [
            note,
            condition
        ]
    )                    
    
    # Create chorale decoder
    merge = TimeDistributed(Dense(rnn_size, activation='tanh'),name='merge')(merge)

    # Create three output layers of chorale decoder
    alto = TimeDistributed(Dense(131, activation='softmax'),name='alto')(merge)
    tenor = TimeDistributed(Dense(131, activation='softmax'),name='tenor')(merge)
    bass = TimeDistributed(Dense(131, activation='softmax'),name='bass')(merge)

    model = Model(
                  inputs=[input_condition, input_note],
                  outputs=[alto, tenor, bass]
                 )

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    if weights_path==None:
        model.summary()

    else:
        model.load_weights(weights_path)

    return model


def train_model(data,
                data_val, 
                batch_size=BATCH_SIZE, 
                epochs=EPOCHS, 
                verbose=2,
                weights_path=WEIGHTS_PATH):

    model = build_model()

    # Load or remove existing weights
    if os.path.exists(weights_path):
        
        try:

            model.load_weights(weights_path)
            print("checkpoint loaded")
        
        except:

            os.remove(weights_path)
            print("checkpoint deleted")

    # Set monitoring indicator
    if len(data_val[0])!=0:
        monitor = 'val_loss'

    else:
        monitor = 'loss'

    # Save weights
    checkpoint = ModelCheckpoint(filepath=weights_path,
                                 monitor=monitor,
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    if len(data_val[0])!=0:

        # With validation set
        history = model.fit(x={'input_condition': np.array(data[0]),
                               'input_note': np.array(data[1])},
                            y={'alto': np.array(data[2]), 
                                'tenor': np.array(data[3]), 
                                'bass': np.array(data[4])},
                            validation_data=({'input_condition': np.array(data_val[0]),
                                            'input_note': np.array(data_val[1])},
                                            {'alto': np.array(data_val[2]), 
                                            'tenor': np.array(data_val[3]), 
                                            'bass': np.array(data_val[4])}),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[checkpoint])
    else:

        # Without validation set
        history = model.fit(x={'input_condition': np.array(data[0]),
                               'input_note': np.array(data[1])},
                            y={'alto': np.array(data[2]), 
                                'tenor': np.array(data[3]), 
                                'bass': np.array(data[4])},
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[checkpoint])
    
    return history
    

if __name__ == "__main__":

    # Load the training and validation sets
    data, data_val = create_training_data()
    
    # Train model
    history = train_model(data, data_val)