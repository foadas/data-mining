import pandas as pd
import numpy as np
max_age = 73
min_age= 14
range_age = max_age - min_age
width = int(np.round(range_age/10))
intervals = [age for age in range(min_age,max_age+width,width)]
print(intervals)