# DeepChoir

## Chord-Conditioned Melody Harmonization with Controllable Harmonicity

This is the source code of DeepChoir, a chord-conditioned melody harmonization system with controllable harmonicity, trained/validated on Chordified JSB Chorales Dataset.  
  
The evaluation data we used in our experiments in the `outputs` folder, and the online discrimination test is available at https://sander-wood.github.io/deepchoir/test.  

You can listening generated samples (chorale, folk, pop, and symphony) at https://sander-wood.github.io/deepchoir/samples.  

For more information, see our paper: [arXiv paper](https://arxiv.org/abs/2202.08423).  
  
## Chordified JSB Chorales Dataset

As we did not find any full JSB Chorales dataset with human-annotated chord symbols, and as performing harmonic analysis manually is too time-consuming, we carried out the following automated preprocessing.
  
1.　**Chordifying**: simplify a complex score with multiple voices into a succession of chords in one voice via a tool in [music21](https://web.mit.edu/music21/doc/usersGuide/usersGuide_09_chordify.html?highlight=chordify).  
2.　**Labelling**: rank all chords based on beat strength, then only keep those with the highest scores and add them to sopranos.

We ended up with a total of 366 chorales for training (90\%) and validation (10\%). Admittedly, these automatically labelled chord symbols are not authoritative from the perspective of harmonic analysis, but they are sufficient for our task.

You can find this chordified version of JSB Chorales dataset in the `dataset` folder. 

<div align="center">
  <img src=https://github.com/sander-wood/deepchoir/blob/homepage/figs/070.png width=100% />
    
  Chordified BWV 322 rendered by MuseScore3
</div>

## Install Dependencies
Python: 3.7.9  
Keras: 2.3.0  
tensorflow-gpu: 2.2.0  
music21: 6.7.1  
tqdm: 4.62.3  
samplings: 0.1.6
  
PS: Third party libraries can be installed using the `pip install` command.

## Melody Harmonization
1.　Put the melodies with chord symbols (in the format of .musicxml, .xml or .mxl) in the `inputs` folder;  
2.　Simply run `choralizer.py`;  
3.　Wait a while and the harmonized melodies will be exported in the `outputs` folder.  
  
You can set two parameters HARMONICITY∈[0, 1] in `config.py` to adjust two attributes of the generated chorales.  
  
The higher the value of HARMONICITY, the generated three parts are more in harmony with the melody.  

## Use Your Own Dataset
1.　Store all the chorales with chord symbols in the `dataset` folder;  
2.　Run `loader.py`, which will generate `corpus.bin`;  
3.　Run `model.py`, which will generate `weights.hdf5`.  
  
After that, you can use `choralizer.py` to generate chorales that fit the musical style of the new dataset.   
  
If you need to finetune the parameters, you can do so in `config.py`. It is not recommended to change the parameters in other files.
