# TREDNet
### Regulatory region detection using two-phase deep neural networks

TREDNet is a modular deep neural network architecture which allows building models to predict regulatory regions.
It comprises two phases:

#### Phase one:
This phase is similar to the standard in the field of genomics, multitask classification setup based 
convolutional neural networks (CNN) such as DeepSEA, Beluga, DanQ etc. It predicts 1924 genomic features of 
transcription factors, histone modifications and DNase-I Hypersensitivity Sites in 123 different tissues and cell-lines. 

#### Phase two:

This phase is a smaller CNN, which is trained on enhancer regions. Let's say we have a set of ehnancers defined as regions
of chromatin accessibility which overlap H3K27ac and H3K4me1 histone modification regions. Then you train the phase two 
model on these enhancer regions, however each region is first passed through the first phase and converted to a vector 
of length 1924. 

The model can be built as a single output classifier (enhancers of a single tissue) or multitask classifier (enhancers 
of multiple tissues)

---------------------------------------------------------------------------------------------------

### Requirements

TREDNet is developed using Python and uses keras/tensorflow for training DL models.

It requires a CUDA-capable linux computer with a GPU node and >150GB memory.  

Dependencies of TREDNet can be installed with pip as follows.
```
    pip install requirements.txt
```

Data preparation step requires:
  - GRCh37/hg19 assembly of the human genome. The fasta file can be downloaded from:
https://hgdownload.soe.ucsc.edu/downloads.html
  - Pre-trained model of the phase-one. Download the archive file:  https://ftp.ncbi.nlm.nih.gov/pub/Dcode/TREDNet/TREDNet_models_bundle.tar.gz
    Uncompress the files to the `data/` directory.


---------------------------------------------------------------------------------------------------

### How to run

Once all the requirements are met, create a dataset to be used for training the model.
```
python create_dataset.py
```

It needs a set of enhancers and control regions. By default, it will use the example sets of 
`data/E118.H3K27ac.enhancers.bed` and `data/E118.H3K27ac.controls.bed`. Supply your own enhancer and control sets' bed files if necessary.

Once the dataset is created, train the model with:
```
python train_model.py

```

To score the variants using the phase-two model:
```
python score_variant.py

```


---------------------------------------------------------------------------------------------------
### How to cite:

TODO

---------------------------------------------------------------------------------------------------
### Contact

First author of the manuscript.

---------------------------------------------------------------------------------------------------
### TODO

- Add an interface for choosing the phase-two model (islets, HepG2, K562 etc.)
  (Where to deposit the trained models?)
- Add a module for delta/IEP score generation.  
- (maybe) Add a multi-task version of phase-two 
