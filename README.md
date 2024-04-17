# TREDNet
TREDNet is now published at PNAS: https://www.pnas.org/doi/10.1073/pnas.2206612120

---------------------------------------------------------------------------------------------------

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

It requires a CUDA-capable linux computer with a GPU node and >150GB memory for training the models from scratch. 
For running precidtions having a GPU machine is recommended but not required. 

Data preparation step requires:
  - GRCh37/hg19 assembly of the human genome. The fasta file can be downloaded from:


---------------------------------------------------------------------------------------------------

### How to run

There are two ways to run predictions on input sequences:

#### Kipoi database
TREDNet models are deposited in the Kipoi database:
http://kipoi.org/models/TREDNet/

Easiest way is to run the model through Kipoi interface. To do so, install the `kipoi` package first:
```
pip install kipoi
```

Then use the `-use-kipoi` flag when running the `score_variant.py` command. When you run the command for the first time, Kipoi package will download the models of two phases. 

#### Local model
If Kipoi proves to be too much of an overhead, run: `bash data_download.sh` script to download the models. The models will be downloaded to `./data` directory. 

Dependencies of TREDNet can be installed with pip as follows.
```
    pip install requirements.txt
```

Then run the `score_variant.py` command with `-use-kipoi` flag omitted, since by default it is set to `False`. 

#### Run the command"

`./score_variant.py [options]` 

Options:

- `-vcf-file` : "VCF file containing variants for calculating the mutational scores."
- `-phase-one-file` : "Phase one model weights (hdf5) file. Skip if using `-use-kipoi` flag"
- `-phase-two-name` : "Phase two model's cell name. Allowed options: islet, HepG2, K562"
- `-save-file` : "File to save the scores"
- `-hg19-fasta` : "Fasta file to hg19 assembly"
- `-score-delta` : "Generate delta scores"
- `-score-iep` : "Generate IEP scores"
- `-use-kipoi` : "Use Kipoi models for running the models"

---------------------------------------------------------------------------------------------------

### Run PAS/DAS detection

There are two ways to run predictions on input sequences:

PAS/DAS detection requires pre-generated delta scores. Input file required in h5 format. 

Then run the `peak_detection/annotate.py`  

#### Run the command"

`peak_detection/annotate.py delta_file model_dir save_file` 

---------------------------------------------------------------------------------------------------
### How to cite:

#### Modeling islet enhancers using deep learning identifies candidate causal variants at loci associated with T2D and glycemic traits
<span style="font-size:1em;"> Sanjarbek Hudaiberdiev, D. Leland Taylor, Wei Song, Narisu Narisu, Redwan M. Bhuiyan, Henry J. Taylor, Xuming Tang, Tingfen Yan, Amy J. Swift, Lori L. Bonnycastle, DIAMANTE Consortium, Shuibing Chen, Michael L. Stitzel, Michael R. Erdos, Ivan Ovcharenko, and Francis S. Collins </span>

Proceedings of National Academy of Science (PNAS) 2023 120 (35) e2206612120. DOI: 10.1073/pnas.2206612120

---------------------------------------------------------------------------------------------------
### Contact

Open an issue on this GitHub repo or email to the first author of the publication.  

