import numpy as np
import pandas as pd

methods = ['lgb', 'nn', 'ridge', 'rf', 'log_reg']

submits = []

for method in methods:
    sub = pd.read_csv(method + '_all_table.csv')
    submits.append(sub)

submit = submits[0].copy()
submit['TARGET'] = 0.0

for sub in submits:
    submit['TARGET'] += sub['TARGET'] / len(methods)

submit.to_csv('average.csv', index=False)
# Private Score: 0.78021, Public Score: 0.77820

submit = submits[0].copy()
submit['TARGET'] = 0.0

for i, sub in enumerate(submits):
    weight = (len(methods) - i) * 2/ (len(methods) + 1) / len(methods)
    #print(weight)
    submit['TARGET'] += sub['TARGET'] * weight

submit.to_csv('weighted_average.csv', index=False)
# Private Score: 0.78191, Public Score: 0.78028
