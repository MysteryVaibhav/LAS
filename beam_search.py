import torch
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict
import sys
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
 
class beamsearch_v2:
    def __init__(self, probs): 
           # Probs is a tensor
           #self.probs = probs
           self.probs = probs.data.cpu().numpy()     
          
    def decode(self):        
        probs = self.probs
        sequences = [[list(), 0]]
        #print "Shape I got is ", probs.shape
        for row in probs:
            #print "Row I got is ", row
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score + row[j]]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            # select k best
            sequences = ordered[len(ordered)-18:]
        #print "I am returning", sequences
        seq, score = sequences[17]
        return seq, score