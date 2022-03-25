# DeepBind

* This repository is for DeepBind reproduction using Pytorch.<br>

* Please note that this reproduction is designed for only Chip-seq datasets.<br>

* You can check the DeepBind Papaer on <a href="https://www.nature.com/articles/nbt.3300">here</a>, the corresponding supplementary nots on <a href="https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.3300/MediaObjects/41587_2015_BFnbt3300_MOESM51_ESM.pdf">here</a>, and the original code with tensorflow 1.x on <a href="https://github.com/jisraeli/DeepBind">here</a>.<br>

# Preparations

1. Dependencies<br>
Create virtual environment DeepBind using following commands<br>
`conda env create --file environment.yml`

2. Download datasets<br>
You can get the required datsets on<br>
https://github.com/jisraeli/DeepBind/tree/master/data/

# How to run the code

1. ~~TF_Binding_Predcition.ipynb~~<br>
*no longer supported

2. ~~TF_Binding_Prediction.py<br>
This code is the same as ipynb format file, but you can experiment multiple datasets using the following commands<br>
`python TF_Binding_Prdiction.py --TF ARID3A`<br>
You can choose datasets among <br>
[ARID3A / CTCFL / ELK1 / FOXA1 / GABPA / MYC / REST / SP1 / USF1 / ZBTB7A]~~
*no longer supported

3. Logo/seq_logo_from_model.ipynb
using this code, you can create sequence logos for specific TF model you trained

4. TF_Binding_Prediction_hyperparameter_experiments.py<br>
This code is designed for hyperparameter tuning experiments.<br>
You can execute this code using the command shwon below<br>
`python TF_Binding_Prediction_hyperparameter_experiments.py 
–-TF {TF Name} –-id {experiments id}`<br>


# TF Binding Prediction AUC Results

* You can check the trainig and testing results on <a href="results/">here</a>.<br>

# Sequence Logo

* You can check the sequence logos created by using the trained models on <a href="Logo/Image">here</a>.<br>

<hr>

* Reference Code : https://github.com/MedChaabane/DeepBind-with-PyTorch