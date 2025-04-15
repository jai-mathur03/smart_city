import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

df = pd.read_csv('jaipur_waste_dataset.csv')
print(len(df))  # Should be 730
print(df.head())
print(df.tail())