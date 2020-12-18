import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, percentileofscore, norm

def LoadPickle(infile):
    with open(infile, 'rb') as f:
        outfile = pickle.load(f)
        return outfile

def pearsonr_ci(x,y,alpha=0.05):
    r, p = pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def stats(weighted, analysis, control_personality):
    true_model_output = LoadPickle(f'../pred_model_{analysis}_weighted-{weighted}_control-personality-{control_personality}.pkl')
    actual_vals = np.concatenate(true_model_output['actual_vals'])
    pred_vals = np.concatenate(true_model_output['pred_vals'])
    mode_pls_components = max(set(true_model_output['selected_n_components']), key=true_model_output['selected_n_components'].count)
    true_rval, raw_pval, lo, hi = pearsonr_ci(actual_vals, pred_vals.ravel())
    permuted_models_output = pd.read_csv(f'../pred_model_{analysis}_weighted-{weighted}_control-personality-{control_personality}_permutations.csv')
    permuted_rvals = np.asarray(p5000_df['rvalue'])
    ptest_pval = float((100-percentileofscore(permuted_rvals, true_rval))/100)
    return ptest_pval

stats(weighted = True, analysis = 'meet', control_personality = False)
