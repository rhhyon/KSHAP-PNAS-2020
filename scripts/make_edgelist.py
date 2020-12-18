import pandas as pd
import igraph
import numpy as np

data_path = '../data'
EGO_COL = ['nid']
ALTER_COLS = ['n_nid_' + str(i) for i in range(1, 7)]
IN_VILLAGE_COLS = ['n_myeon_' + str(i) for i in range(1, 7)]

def preproc(df):
    # Include all spouses even if outside of village
    df['n_myeon_1'] = df['n_myeon_1'].map({1: 1, 2: 1})

    for col in IN_VILLAGE_COLS[1:]:
        df[col] = df[col].map({1: 1, 2: 0})

    df = df.fillna(0)

    df['nid'] = df['nid'].astype(int)#.astype(str)
    for alter_col in ALTER_COLS:
        df[alter_col] = df[alter_col].astype(int)#.astype(str)

    for in_vil_col in IN_VILLAGE_COLS:
        df[in_vil_col] = df[in_vil_col].astype(int)#.astype(str)
    return df

def get_network_subs(df):
    egos = df['nid'].tolist()
    df_subset = df.loc[:, ['nid', 'n_nid_1','n_nid_2', 'n_nid_3', 'n_nid_4', 'n_nid_5', 'n_nid_6', 'n_myeon_1', 'n_myeon_2', 'n_myeon_3', 'n_myeon_4', 'n_myeon_5', 'n_myeon_6']]

    # Multiply in_village values (1 or 0) with alter IDs to exclude out-of-village alters
    alters_df = df.loc[:, ALTER_COLS]
    in_village_df = df.loc[:, IN_VILLAGE_COLS]
    df_subset = pd.DataFrame(alters_df.values*in_village_df.values, columns = alters_df.columns, index = alters_df.index)
    df_subset['nid'] = df['nid']
    df_subset = df_subset[EGO_COL + ALTER_COLS]
    edgelist_tmp = df_subset.set_index('nid').transpose().to_dict(orient = 'list')

    # Delete NaNs (0's). alters that were not nominated (like if someone had <7 friends) or alters that were out-of-village
    for i in range(7):
        for v in edgelist_tmp.values():
            if 0 in v:
                v.remove(0)

    # Identify egos that did not nominate any alters
    egos_without_alters_list = [k for k, v in edgelist_tmp.items() if v == []]

    # Create dictionary without egos_without_alters
    egos_without_alters = {k: v for k, v in edgelist_tmp.items()}# if v != []}

    # Create graph
    g = igraph.Graph.TupleList([(k, v) for k, vs in egos_without_alters.items() for v in vs], directed = True)
    g = igraph.Graph.simplify(g, loops=True)

    # Create list of network IDs (N = 802)
    network_subs = [g.vs[i]['name'] for i in range(len(g.vs))]
    network_subs = [str(i) for i in network_subs]

    return network_subs

def create_weighted_edgelist1(df, network_subs, weight):
    # Create dictionary of ego+alters and their edge weights
    weight_cols = [f'n_{weight}_' + str(i) for i in range(1, 7) ]
    weight_dict = {}
    network_subs = [int(i) for i in network_subs]
    for alter_col, weight_col in zip(ALTER_COLS, weight_cols):
        for idx, (alter, weight) in enumerate(zip(df[alter_col].values, df[weight_col].values)):
            if (alter in network_subs) and (weight != 0):
                ego = str(df['nid'].iloc[idx])
                weight_dict[ego + ',' + str(alter)] = weight

    return weight_dict

def create_weighted_edgelist2(weight, weight_dict):
    # Convert dictionary to a dataframe
    edgelist_df = pd.DataFrame(list(weight_dict.items()), columns = ['dyads', 'weight'])
    alter_ego_df = edgelist_df.dyads.str.split(',', expand=True)
    edgelist_df_cleaned = pd.concat([alter_ego_df, edgelist_df.weight], axis = 1)
    edgelist_df_cleaned.columns = ['source', 'target', 'weight']
    edgelist_df_cleaned.source = 's' + edgelist_df_cleaned.source.astype(str); edgelist_df_cleaned.target = 's' + edgelist_df_cleaned.target.astype(str)

    if not weight == 'feel':
        reverse_code = {1:9, 2:8, 3:7, 4:6, 6:4, 7:3, 8:2, 9:1}
        edgelist_df_cleaned['weight'] = edgelist_df_cleaned['weight'].replace(reverse_code)

    edgelist_df_cleaned.to_csv(f'{data_path}/edgelist_weighted_{weight}.csv', sep=' ', index = False, header = False)

def create_unweighted_edgelist(weight_dict):
    edgelist_df = pd.DataFrame(list(weight_dict.items()), columns = ['dyads', 'weight'])
    alter_ego_df = edgelist_df.dyads.str.split(',', expand=True)
    edgelist_df_cleaned = pd.concat([alter_ego_df, edgelist_df.weight], axis = 1)
    edgelist_df_cleaned.columns = ['source', 'target', 'weight']
    edgelist_df_cleaned.source = 's' + edgelist_df_cleaned.source.astype(str); edgelist_df_cleaned.target = 's' + edgelist_df_cleaned.target.astype(str)
    unweighted = edgelist_df_cleaned[['source', 'target']]
    unweighted.to_csv(f'{data_path}/edgelist_unweighted.csv', sep=' ', index = False, header = False)


def main(weight):
    df = pd.read_csv(f'{data_path}/social_network_data.csv')
    df = preproc(df)
    network_subs = get_network_subs(df)
    weight_dict = create_weighted_edgelist1(df, network_subs, weight)
    create_weighted_edgelist2(weight, weight_dict)

    # create_unweighted_edgelist(weight_dict) # optional

# weight = feel (closeness), comm (communication freq), or meet (meeting freq)
main(weight = 'meet')
