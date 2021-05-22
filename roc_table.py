# Description: Calculate AUC-ROC for each model on test set to make table

import os
import pandas as pd
from sklearn import metrics

# Define files with test ROC curves
data_dir = 'results'
files = ['b_roc.csv', 'mcnn_roc.csv', 'mrcnn_roc.csv', 'dcnn_roc.csv']
models = ['Baseline CNN', 'Multilayer CNN', 'Multilayer CNN + Dropout', 'Multilayer CNN + Dropout + Deep FCN']

aucs = []

# Compute AUC-ROCs
for f in files:
    df = pd.read_csv(os.path.join(data_dir, f))
    cstat = metrics.auc(df['fpr'], df['tpr'])
    aucs.append(cstat)

# Show results
res = pd.DataFrame.from_dict({'model' : models, 'AUC-ROC' : aucs})
print(res)
