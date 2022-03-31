import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import statistics as stats
import numpy as np

df = pd.read_csv('new_curry.csv', delimiter=',')


#ORDERED
opps = dict()

for i, row in df.iterrows():
    if row['OPP'] in opps:
        #Append points for an existing team
        opps[row['OPP']].append(row['PTS'])
    else:
        #Create a new array for a team
        opps[row['OPP']] = [row['PTS']]

#Compute averages for points against a team
opps.update((key, stats.mean(value)) for key, value in opps.items())

#Sort by the average points
sorted_opps = sorted(opps.items(), key=lambda x: x[1])

#Replace avg_points with a value from 1-30
opp_list = list(range(0,29))

for i in range(1,30):
    opp_list[i-1] = [sorted_opps[i-1][0], i]

orders = []

for i, row in df.iterrows():
    for opp in opp_list:
        if row['OPP'] == opp[0]:
            orders.append(opp[1])

df['ORDER'] = orders

#df.to_csv('curry_ordered.csv', index=False)

''''
#ORDINAL
new_df = pd.get_dummies(df)
new_df.to_csv('curry_onehot.csv', index=False)
'''