import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, GridSearchCV


def SavePickle(infile, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(infile, f, protocol = 2)

def exclude_cohabitants(df):
    df = df[~(df['phys_dist'] == 0)]
    df = df.reset_index()
    return df

def run_plsr(df, analysis, weighted, control_personality):
    roi_cols = [i for i in list(df.columns) if 'edge' in i]

    X_data = df[roi_cols]
    y_data = df['soc_dist_cleaned']

    output = {}
    coefs = []
    actual_vals = []
    pred_vals = []
    selected_n_components = []
    selected_edges = []
    pls_components = []

    param_grid = {'plsr__n_components': [i for i in range(1, 111)]}
    grid_pipe = Pipeline(steps = [('scaler', RobustScaler()),
                        ('filter', SelectKBest(f_regression, k = 1000)),
                        ('plsr', PLSRegression())])
    kf = KFold(n_splits = 10, shuffle = True)
    cvsplit = kf.split(X_data)
    grid = GridSearchCV(grid_pipe, param_grid = param_grid, cv = kf, n_jobs = 1)

    for train, test in cvsplit:
        train_X = X_data.values[train]
        train_y = y_data[train]

        test_X = X_data.values[test]
        test_y = y_data[test]

        grid.fit(train_X, train_y)
        predicted_y = grid.best_estimator_.predict(test_X)
        X_data_components = pd.DataFrame(grid.best_estimator_.transform(X_data), columns = [f'c{i}' for i in range(1, grid.best_params_['plsr__n_components'] + 1)])

        pls_components.append(X_data_components)
        selected_n_components.append(grid.best_params_['plsr__n_components'])
        coefs.append(grid.best_estimator_.named_steps['plsr'].coef_[:,0])
        selected_edges.append(X_data.columns[grid.best_estimator_.named_steps['filter'].get_support(indices=True)].tolist())
        pred_vals.append(predicted_y)
        actual_vals.append(test_y)

    output['actual_vals'] = actual_vals
    output['pred_vals'] = pred_vals
    output['coefs'] = coefs
    output['selected_n_components'] = selected_n_components
    output['selected_edges'] = selected_edges
    output['pls_components'] = pls_components

    SavePickle(output, f'../pred_model_{analysis}_weighted-{weighted}_control-personality-{control_personality}.pkl')


def main(analysis, weighted, control_personality):
    df = pd.read_csv(f'../{analysis}_weighted-{weighted}_control-personality-{control_personality}.csv')
    # df = exclude_cohabitants(df) # optional

    run_plsr(df, analysis, weighted, control_personality)


main(weighted = True, analysis = 'meet', control_personality = False)
