import pandas as pd
import numpy as np
my_data=pd.read_csv("forestfires.csv")
print(my_data)
# check for nan values(replace with 0) and convert categorical data into numerical data if any numerical data is not present
temp1=my_data["temp"]
#print(temp1)
print(temp1.isna().any())
if temp1.isna().any():
    print("no nan ")
else:
    print("yes nan")