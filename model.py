import os
import pickle
import numpy as np
from config import *
from keras import Model
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers  import concatenate
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical

def create_training_data(corpus_path=CORPUS_PATH, seg_length=SEGMENT_LENGTH, val_ratio=VAL_RATIO):
    
    # Load corpus
    with open(corpus_path, "rb") as filepath:
        corpus = pickle.load(filepath)

    # Inputs and targets for the training set
    input_melody_left = []
    input_melody_right = []
    input_beat_left = []
    input_beat_right = []
    input_fermata_left = []
    input_fermata_right = []
    input_chord_left = []
    input_chord_right = []
    input_alto = []
    input_tenor = []
    input_bass = []
    output_alto = []
    output_tenor = []
    output_bass = []

    # Inputs and targets for the validation set
    val_input_melody_left = []
    val_input_melody_right = []
    val_input_beat_left = []
    val_input_beat_right = []
    val_input_fermata_left = [] 
    val_input_fermata_right = []
    val_input_chord_left = []
    val_input_chord_right = []
    val_input_alto = []
    val_input_tenor = []
    val_input_bass = []
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
    filenames = corpus[4]

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
            # print(filenames[song_idx])

        song_melody = seg_length*[0] + soprano_melody[song_idx] + seg_length*[0]
        song_beat = seg_length*[0] + beat_data[song_idx] + seg_length*[0]
        song_fermata = seg_length*[0] + fermata_data[song_idx] + seg_length*[0]
        song_chord = seg_length*[[0.]*12] + chord_data[song_idx] + seg_length*[[0.]*12]
        song_alto = seg_length*[0] + alto_melody[song_idx] + seg_length*[0]
        song_tenor = seg_length*[0] + tenor_melody[song_idx] + seg_length*[0]
        song_bass = seg_length*[0] + bass_melody[song_idx] + seg_length*[0]

        # Create pairs
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
            
            target_alto = song_alto[idx]
            target_tenor = song_tenor[idx]
            target_bass = song_bass[idx]

            if train_or_val=='train':
                input_melody_left.append(melody_left)
                input_melody_right.append(melody_right)
                input_beat_left.append(beat_left)
                input_beat_right.append(beat_right)
                input_fermata_left.append(fermata_left)
                input_fermata_right.append(fermata_right)
                input_chord_left.append(chord_left)
                input_chord_right.append(chord_right)
                input_alto.append(alto)
                input_tenor.append(tenor)
                input_bass.append(bass)
                output_alto.append(target_alto)
                output_tenor.append(target_tenor)
                output_bass.append(target_bass)

            else:
                val_input_melody_left.append(melody_left)
                val_input_melody_right.append(melody_right)
                val_input_beat_left.append(beat_left)
                val_input_beat_right.append(beat_right)
                val_input_fermata_left.append(fermata_left)
                val_input_fermata_right.append(fermata_right)
                val_input_chord_left.append(chord_left)
                val_input_chord_right.append(chord_right)
                val_input_alto.append(alto)
                val_input_tenor.append(tenor)
                val_input_bass.append(bass)
                val_output_alto.append(target_alto)
                val_output_tenor.append(target_tenor)
                val_output_bass.append(target_bass)

            cnt += 1

    print("Successfully read %d samples" %(cnt))

    # One-hot vectorization
    input_melody_left = to_categorical(input_melody_left, num_classes=130)
    input_melody_right = to_categorical(input_melody_right, num_classes=130)
    input_beat_left = to_categorical(input_beat_left, num_classes=4)
    input_beat_right = to_categorical(input_beat_right, num_classes=4)
    input_fermata_left = to_categorical(input_fermata_left, num_classes=2)
    input_fermata_right = to_categorical(input_fermata_right, num_classes=2)
    input_alto = to_categorical(input_alto, num_classes=130)
    input_tenor = to_categorical(input_tenor, num_classes=130)
    input_bass = to_categorical(input_bass, num_classes=130)
    output_alto = to_categorical(output_alto, num_classes=130)
    output_tenor = to_categorical(output_tenor, num_classes=130)
    output_bass = to_categorical(output_bass, num_classes=130)
    
    # concat beat, fermata and chord
    input_condition_left = np.concatenate((input_beat_left, input_fermata_left, input_chord_left), axis=-1)
    input_condition_right = np.concatenate((input_beat_right, input_fermata_right, input_chord_right), axis=-1)

    if len(val_input_melody_left)!=0:
        val_input_melody_left = to_categorical(val_input_melody_left, num_classes=130)
        val_input_melody_right = to_categorical(val_input_melody_right, num_classes=130)
        val_input_beat_left = to_categorical(val_input_beat_left, num_classes=4)
        val_input_beat_right = to_categorical(val_input_beat_right, num_classes=4)
        val_input_fermata_left = to_categorical(val_input_fermata_left, num_classes=2)
        val_input_fermata_right = to_categorical(val_input_fermata_right, num_classes=2)
        val_input_alto = to_categorical(val_input_alto, num_classes=130)
        val_input_tenor = to_categorical(val_input_tenor, num_classes=130)
        val_input_bass = to_categorical(val_input_bass, num_classes=130)
        val_output_alto = to_categorical(val_output_alto, num_classes=130)
        val_output_tenor = to_categorical(val_output_tenor, num_classes=130)
        val_output_bass = to_categorical(val_output_bass, num_classes=130)

        val_input_condition_left = np.concatenate((val_input_beat_left, val_input_fermata_left, val_input_chord_left), axis=-1)
        val_input_condition_right = np.concatenate((val_input_beat_right, val_input_fermata_right, val_input_chord_right), axis=-1)
    
    return (input_melody_left, input_melody_right, input_condition_left, input_condition_right, input_alto, input_tenor, input_bass, output_alto, output_tenor, output_bass), \
           (val_input_melody_left, val_input_melody_right, val_input_condition_left, val_input_condition_right, val_input_alto, val_input_tenor, val_input_bass, val_output_alto, val_output_tenor, val_output_bass)


def build_model(rnn_size=RNN_SIZE, num_layers=NUM_LAYERS, seg_length=SEGMENT_LENGTH, dropout=DROPOUT, weights_path=None, training=False):

    # Soprano embedding
    input_melody_left = Input(shape=(seg_length, 130), name='input_melody_left')
    melody_left = TimeDistributed(Dense(rnn_size, activation='relu'), name='melody_left_embedding')(input_melody_left)

    input_melody_right = Input(shape=(seg_length, 130), name='input_melody_right')
    melody_right = TimeDistributed(Dense(rnn_size, activation='relu'), name='melody_right_embedding')(input_melody_right)

    # Conditions embedding
    input_condition_left = Input(shape=(seg_length, 18), name='input_condition_left')
    condition_left = TimeDistributed(Dense(rnn_size, activation='relu'), name='condition_left_embedding')(input_condition_left)

    input_condition_right = Input(shape=(seg_length, 18), name='input_condition_right')
    condition_right = TimeDistributed(Dense(rnn_size, activation='relu'), name='condition_right_embedding')(input_condition_right)
    
    # Output shift embedding
    input_alto = Input(shape=(seg_length, 130), name='input_alto')
    alto = TimeDistributed(Dense(rnn_size, activation='relu'), name='alto_embedding')(input_alto)

    input_tenor = Input(shape=(seg_length, 130), name='input_tenor')
    tenor = TimeDistributed(Dense(rnn_size, activation='relu'), name='tenor_embedding')(input_tenor)

    input_bass = Input(shape=(seg_length, 130), name='input_bass')
    bass = TimeDistributed(Dense(rnn_size, activation='relu'), name='bass_embedding')(input_bass)
    
    return_sequences = True

    # Create encoders
    for idx in range(num_layers):
     
        if idx==num_layers-1:
            return_sequences = False

        melody_left = LSTM(rnn_size, 
                           name='melody_left_'+str(idx+1),
                           return_sequences=return_sequences)(melody_left)
        if idx!=num_layers-1:
            melody_left = TimeDistributed(Dense(rnn_size, activation='relu'))(melody_left)
        melody_left = BatchNormalization()(melody_left)
        melody_left = Dropout(dropout)(melody_left, training=training)

        melody_right = LSTM(rnn_size,
                            name='melody_right_'+str(idx+1),
                            return_sequences=return_sequences)(melody_right)
        if idx!=num_layers-1:
            melody_right = TimeDistributed(Dense(rnn_size, activation='relu'))(melody_right)
        melody_right = BatchNormalization()(melody_right)
        melody_right = Dropout(dropout)(melody_right, training=training)

        condition_left = LSTM(rnn_size,
                                name='condition_left_'+str(idx+1),
                                return_sequences=return_sequences)(condition_left)
        if idx!=num_layers-1:
            condition_left = TimeDistributed(Dense(rnn_size, activation='relu'))(condition_left)
        condition_left = BatchNormalization()(condition_left)
        condition_left = Dropout(dropout)(condition_left, training=training)

        condition_right = LSTM(rnn_size,
                                name='condition_right_'+str(idx+1),
                                return_sequences=return_sequences)(condition_right)
        if idx!=num_layers-1:
            condition_right = TimeDistributed(Dense(rnn_size, activation='relu'))(condition_right)
        condition_right = BatchNormalization()(condition_right)
        condition_right = Dropout(dropout)(condition_right, training=training)

        alto = LSTM(rnn_size,
                    name='alto_'+str(idx+1),
                    return_sequences=return_sequences)(alto)
        if idx!=num_layers-1:
            alto = TimeDistributed(Dense(rnn_size, activation='relu'))(alto)
        alto = BatchNormalization()(alto)
        alto = Dropout(dropout)(alto, training=training)

        tenor = LSTM(rnn_size,
                    name='tenor_'+str(idx+1),
                    return_sequences=return_sequences)(tenor)
        if idx!=num_layers-1:
            tenor = TimeDistributed(Dense(rnn_size, activation='relu'))(tenor)
        tenor = BatchNormalization()(tenor)
        tenor = Dropout(dropout)(tenor, training=training)

        bass = LSTM(rnn_size,
                    name='bass_'+str(idx+1),
                    return_sequences=return_sequences)(bass)
        if idx!=num_layers-1:
            bass = TimeDistributed(Dense(rnn_size, activation='relu'))(bass)
        bass = BatchNormalization()(bass)
        bass = Dropout(dropout)(bass, training=training)

    # Merge hidden layer output
    merge = concatenate(
        [melody_left, melody_right, condition_left, condition_right, alto, tenor, bass],
    )                    

    # Create decoder
    for idx in range(num_layers):
        merge= Dense(units=rnn_size,
                    activation='relu',
                    name='merge_'+str(idx))(merge)
        merge = BatchNormalization()(merge)
        merge = Dropout(dropout)(merge, training=training)

    # Output alto, tenor and bass
    target_alto = Dense(130, activation='softmax',name='target_alto')(merge)
    target_tenor = Dense(130, activation='softmax',name='target_tenor')(merge)
    target_bass = Dense(130, activation='softmax',name='target_bass')(merge)

    model = Model(
                  inputs=[input_melody_left, input_melody_right, input_condition_left, input_condition_right, input_alto, input_tenor, input_bass],
                  outputs=[target_alto, target_tenor, target_bass]
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
        history = model.fit(x={'input_melody_left': data[0],
                               'input_melody_right': data[1],
                               'input_condition_left': data[2],
                               'input_condition_right': data[3],
                               'input_alto': data[4],
                               'input_tenor': data[5],
                               'input_bass': data[6]},
                            y={'target_alto': data[7],
                                'target_tenor': data[8],
                                'target_bass': data[9]},
                            validation_data=({'input_melody_left': data_val[0],
                                                'input_melody_right': data_val[1],
                                                'input_condition_left': data_val[2],
                                                'input_condition_right': data_val[3],
                                                'input_alto': data_val[4],
                                                'input_tenor': data_val[5],
                                                'input_bass': data_val[6]},
                                                {'target_alto': data_val[7],
                                                'target_tenor': data_val[8],
                                                'target_bass': data_val[9]}),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[checkpoint])
    else:

        # Without validation set
        history = model.fit(x={'input_melody_left': data[0],
                               'input_melody_right': data[1],
                               'input_condition_left': data[2],
                               'input_condition_right': data[3],
                               'input_alto': data[4],
                               'input_tenor': data[5],
                               'input_bass': data[6]},
                            y={'target_alto': data[7],
                                'target_tenor': data[8],
                                'target_bass': data[9]},
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