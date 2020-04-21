import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
filepath = os.path.join(__location__, 'ex2data1.txt')

data_frame = pd.read_csv(filepath, header = None)
print(data_frame.corr())