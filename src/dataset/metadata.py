import torch
import numpy as np
import pandas as pd
from torchvision.datasets import Cityscapes

meta = []
for item in Cityscapes.classes:
    meta.append([item.id,item.name])
data = pd.DataFrame(meta,columns=['id','name'])
data.to_csv('assets/meta.csv',index=False)