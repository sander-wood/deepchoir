# Path setting
DATASET_PATH = "dataset"
CORPUS_PATH = "corpus.bin"
WEIGHTS_PATH = 'weights.hdf5'
INPUTS_PATH = "inputs"
OUTPUTS_PATH = "outputs"

# 'loader.py'
EXTENSION = ['.musicxml', '.xml', '.mxl']

# 'model.py'
VAL_RATIO = 0.1
DROPOUT = 0.2
SEGMENT_LENGTH = 128
RNN_SIZE = 128
NUM_LAYERS = 3
BATCH_SIZE = 256
EPOCHS = 20

# 'choralizer.py'
HARMONICITY = 0.5
WATER_MARK = True