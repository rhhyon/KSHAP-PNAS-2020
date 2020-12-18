import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cross_decomposition import PLSRegression
import os
from scipy.stats import linregress

data_path = '../data'


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

def run_plsr(df, analysis, weighted, control_personality, n_components):
    roi_cols = [i for i in list(df.columns) if 'edge' in i]

    X_data = df[roi_cols]
    y_data = df[outcome_var]

    actual_vals = []
    pred_vals = []

    pipe = Pipeline(steps = [('scaler', RobustScaler()),
                        ('filter', SelectKBest(f_regression, k = 1000)),
                        ('plsr', PLSRegression(n_components = n_components))])
    kf = KFold(n_splits = 10, shuffle = True)
    cvsplit = kf.split(X_data)

    for train, test in cvsplit:
        train_X = X_data.values[train]
        train_y = y_data[train]

        test_X = X_data.values[test]
        test_y = y_data[test]

        pipe.fit(train_X, train_y)
        predicted_y = grid.best_estimator_.predict(test_X)

        pred_vals.append(np.concatenate(pipe.predict(test_X).reshape(1,-1)))
        actual_vals.append(test_y)

    results = list(linregress(np.concatenate(actual_vals), np.concatenate(pred_vals, axis = 0)))
    output = []
    output.append(results)
    output_df = pd.DataFrame(output, columns = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr'])

    if not os.path.exists(f'../pred_model_{analysis}_weighted-{weighted}_control-personality-{control_personality}_permutations.csv'):
        output_df.to_csv(f'../pred_model_{analysis}_weighted-{weighted}_control-personality-{control_personality}_permutations.csv', index = False)
    else:
        output_df.to_csv(f'../pred_model_{analysis}_weighted-{weighted}_control-personality-{control_personality}_permutations.csv', mode = 'a', index = False, header = False)

def main(analysis, weighted, control_personality):
    df = pd.read_csv(f'{data_path}/{analysis}_weighted-{weighted}_control-personality-{control_personality}_permuted_{i}.csv')
    # df = exclude_cohabitants(df) # optional

    n_components = get_modal_plsr_components_val(analysis, weighted, control_personality)

    run_plsr(df, analysis, weighted, control_personality, n_components)

# run permute_predictive_models.py {insert number here} in command line
# ideally, submit as a batch script to a cluster so models are run in parallel,
# especially if permuting 5,000 times

main(weighted = True, analysis = 'meet', control_personality = False)
