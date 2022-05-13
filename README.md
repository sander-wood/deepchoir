# DeepChoir

## Chord-Conditioned Melody Choralization with Controllable Harmonicity and Polyphonicity

This is the source code of DeepChoir, a melody choralization system, which can generate a four-part chorale for a given melody conditioned on a chord progression, trained/validated on Chordified JSB Chorales Dataset.  
  
The evaluation data we used in our experiments in the `outputs` folder, and the musical discrimination test is available at https://sander-wood.github.io/deepchoir/test.  

The generated samples (chorales, folk songs and a symphony) are in the `samples` folder, you can also listening them at https://sander-wood.github.io/deepchoir/samples.  

For more information, see our paper: [arXiv paper](https://arxiv.org/abs/2202.08423).  
  
We have updated the code for DeepChoir and the current version has a higher quality of generation overall than the original. The paper will be updated around mid-2022.  
  
## Chordified JSB Chorales Dataset

Since the original JSB Chorales Dataset has no chord progressions and the workload of carrying out harmonic analysis manually is too large, we perform the following automated pre-processing to add chord symbols.  
  
1.　**Flattening**: all repeat barlines are removed by flattening each score to make them more machine-readable.  
2.　**Chordify**: a tool in [music21](https://web.mit.edu/music21/doc/usersGuide/usersGuide_09_chordify.html?highlight=chordify) for simplifying a complex score with multiple parts into a succession of chords in one part.  
3.　**Labelling**: we first move all the chords to the closed position, and then label the chordified chords as chord symbols. Finally, all chord symbols on beats of the soprano part are kept.  

After removing a few scores that cannot be properly chordified, we ended up with a total of 366 chorales for training (90\%) and validation (10\%).  

You can find this chordified version of JSB Chorales dataset in the `dataset` folder. 

<div align="center">
  <img src=https://github.com/sander-wood/deepchoir/blob/homepage/figs/070.png width=100% />
    
  Chordified BWV 322 exported in MuseScore3
</div>

## Install Dependencies
Python: 3.7.9  
Keras: 2.3.0  
tensorflow-gpu: 2.2.0  
music21: 6.7.1  
tqdm: 4.62.3  
samplings: 0.1.6
  
PS: Third party libraries can be installed using the `pip install` command.

## Melody Choralization
1.　Put the melodies with chord symbols (in the format of .musicxml, .xml or .mxl) in the `inputs` folder;  
2.　Simply run `choralizer.py`;  
3.　Wait a while and the choralized melodies will be saved in the `outputs` folder.  
  
You can set two parameters HARMONICITY∈[0, 1] in `config.py` to adjust two attributes of the generated chorales.  
  
The higher the value of HARMONICITY, the generated three parts are more in harmony with the melody.  

## Use Your Own Dataset
1.　Store all the chorales with chord symbols in the `dataset` folder;  
2.　Run `loader.py`, which will generate `corpus.bin`;  
3.　Run `model.py`, which will generate `weights.hdf5`.  
  
After that, you can use `choralizer.py` to generate chorales that fit the musical style of the new dataset.   
  
If you need to finetune the parameters, you can do so in `config.py`. It is not recommended to change the parameters in other files.
