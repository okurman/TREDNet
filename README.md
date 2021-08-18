# TREDNet
### Regulatory region detection using two-phase deep neural networks

TREDNet is a modular deep neural network architecture which allows building models to predict regulatory regions.
It comprises two phases:

##### Phase one:
This phase is very similar to the now-standard in the field of genomics, multitask classification setup based 
convolutional neural networks (CNN) such as DeepSEA, Beluga, DanQ etc. It predicts 1924 genomic features of 
transcription factors, histone modifications and DNase-I Hypersensitivity Sites in 123 different tissues and cell-lines. 

The phase one is trained on the entire genome. 

##### Phase two:

This phase is a smaller CNN, which is trained on enhancer regions. Let's say we have a set of ehnancers defined as regions
of chromatin accessibility which overlap H3K27ac and H3K4me1 histone modification regions. Then you train the phase two 
model on these enhancer regions, however each region is first passed through the first phase and converted to a vector 
of length 1924. 

The model can be built as a single output classifier (enhancers of a single tissue) or multitask classifier (enhancers 
of multiple tissues)

### How to cite:

(tentative)
Hudaiberdiev S, Taylor L, Song W, Narisu N, Stitzel M, Ovcharenko I and Collins F, "Modeling islet enhancers using deep learning identifies candidate causal variants at loci associated with T2D."



