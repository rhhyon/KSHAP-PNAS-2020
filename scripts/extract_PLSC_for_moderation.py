import pandas as pd
import igraph
import pickle
import pandas as pd
import numpy as np
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

data_path = '../'

def LoadPickle(infile):
    with open(infile, 'rb') as f:
        outfile = pickle.load(f)
        return outfile

def exclude_cohabitants(df):
    df = df[~(df['phys_dist'] == 0)]
    df = df.reset_index()
    return df

def get_modal_plsr_components_val(analysis, weighted, control_personality):
    true_model_output = LoadPickle(f'../pred_model_{analysis}_weighted-{weighted}_control-personality-{control_personality}.pkl')
    n_components = max(set(true_model_output['selected_n_components']), key=true_model_output['selected_n_components'].count)
    return n_components

def extract_PLSC(df, n_components):
    df['soc_proximity_cleaned'] = -df['soc_dist_cleaned']

    roi_col = [i for i in list(df.columns) if 'edge' in i]
    X_data = df[roi_col]
    y_data = df['soc_proximity_cleaned']
    iterator = 0
    svr_plots = {}
    actual_vals, pred_vals = [], []
    selected_n_components, selected_edges = [], []
    pls_pipe = Pipeline(steps = [('scale1', RobustScaler()),
                                ('filter', SelectKBest(f_regression, k = 1000)),
                               ('plsr', PLSRegression(n_components = n_components))])
    pls_pipe.fit(X_data, y_data)
    X_data_components = pd.DataFrame(pls_pipe.transform(X_data), columns = [f'neural_similarity_c{i}' for i in range(1, n_components+1)])
    df.rename(columns={'level_0': 'sub1', 'level_1': 'sub2'}, inplace=True)
    df_combined = pd.concat([df[['sub1', 'sub2', 'soc_proximity_cleaned', 'soc_dist_cleaned', 'phys_dist']], X_data_components], axis = 1)

    return df_combined

def main(analysis, weighted, control_personality):
    df = pd.read_csv(f'{data_path}/{analysis}_weighted-{weighted}_control-personality-{control_personality}.csv')
    n_components = get_modal_plsr_components_val(analysis, weighted, control_personality)
    df_combined = extract_PLSC(df, n_components)
    df_combined.to_csv(f'../PLS_components_{analysis}_weighted-{weighted}_control-personality-{control_personality}.csv', index = False)

main(analysis = 'meet', weighted = True, control_personality = False)
