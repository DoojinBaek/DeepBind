# Preparations

1. Dependencies<br>
Create virtual environment DeepBind using following commands<br>
`conda env create --file environment.yml`

2. Download datasets<br>
You can get the required datsets on<br>
https://github.com/jisraeli/DeepBind/tree/master/data/

<hr>

# How to run the code
1. TF_Binding_Predcition.ipynb<br>

2. TF_Binding_Prediction.py<br>
This code is the same as ipynb format file, but you can experiment multiple datasets using the following commands<br>
`python TF_Binding_Prdiction.py --TF ARID3A`<br>
You can choose datasets among <br>
[ARID3A / CTCFL / ELK1 / FOXA1 / GABPA / MYC / REST / SP1 / USF1 / ZBTB7A]

3. Logo/seq_logo_from_model.ipynb
using this code, you can create sequence logos for specific TF model you trained

<hr>

# TF Binding Prediction AUC Results
You can check the trainig and testing results on <a href="results/">here</a>.<br>

<hr>

# Sequence Logo
You can check the sequence logos created by using the trained models on <a href="Logo/Image">here</a>.<br>

<hr>
*Reference Code : https://github.com/MedChaabane/DeepBind-with-PyTorch