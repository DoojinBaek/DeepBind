import os
import numpy as np
import random
import math 

bases = 'ACGT' # DNA bases

# only supporting DNA bases
def seq2pad(sequence, motiflen):
    rows = len(sequence) + 2*motiflen - 2
    S = np.empty([rows, 4])
    base = bases
    for i in range(rows):
        for j in range(4):
            if i-motiflen+1<len(sequence) and sequence[i-motiflen+1] == 'N' or i<motiflen-1 or i>len(sequence)+motiflen-2:
                S[i,j] = np.float32(0.25)
            elif sequence[i-motiflen+1] == base[j]:
                S[i,j] = np.float32(1)
            else:
                S[i,j] = np.float32(0)
    return np.transpose(S)

def dinuc_shuffle(sequence):
    b = [sequence[i:i+2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d = ''.join([str(x) for x in b])
    return d

def complement(sequence):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    complement_sequence = [complement[base] for base in sequence]
    return complement_sequence

def reverse_complement(sequence):
    sequence = list(sequence)
    sequence.reverse()
    return ''.join(complement(sequence))


# datasets

def datasets(path):
    path = './data/encode/'
    files = os.listdir(path)

    dataset_names = [[], []]

    for file in files:
        if file.endswith("B.seq.gz"):
            dataset_names[1].append(path+file)
        elif file.endswith("AC.seq.gz"):
            dataset_names[0].append(path+file)

    dataset_names[0].sort()
    dataset_names[1].sort()

    if(len(dataset_names[0]) != len(dataset_names[1])):
        raise Exception("Dataset Corrputed. Please Download The Dataset Again")

    return dataset_names

def logsampler(a,b):
    x = np.random.uniform(low = 0, high = 1)
    y = 10**((math.log10(b)-math.log10(a))*x + math.log10(a))
    return y

def sqrtsampler(a,b):
    x = np.random.uniform(low = 0, high = 1)
    y = (b-a)*math.sqrt(x) + a
    return y