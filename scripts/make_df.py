import igraph
import pickle
import pandas as pd
import numpy as np
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data_path = '../data'

def LoadPickle(infile):
    with open(infile, 'rb') as f:
        outfile = pickle.load(f)
        return outfile

def exclude_distant_dyads(g, subjects):
    dyads = list(itertools.product(subjects, subjects))
    unique_dyads = []
    for dyad in dyads:
        dyad_o = tuple(sorted(dyad))
        if dyad_o not in unique_dyads and dyad_o[0] != dyad_o[1]:
            unique_dyads.append(dyad_o)

    distant_dyads = []
    for dyad in unique_dyads:
        subj1 = dyad[0]
        subj2 = dyad[1]

        # add geodesic distance between these participants in the social network
        soc_dist = igraph.Graph.shortest_paths(g, source = subj1, target = subj2, weights = None)[0][0]

        if soc_dist == float('inf'):
            distant_dyads.append(dyad)

        if soc_dist > 4:
            distant_dyads.append(dyad)

    dyads = [i for i in unique_dyads if not i in distant_dyads]

    return dyads

def exclude_kin(g, dyads, subjects, kd_df):
    kinship_values = [3, 5, 7, 8, 9]
    n_relations = ['n_relation_' + str(i) for i in range(1,7)]
    n_alters = ['n_nid_' + str(i) for i in range(1,7)]
    related_dyads = []
    for dyad in dyads:
        subj1 = dyad[0]
        subj2 = dyad[1]

        soc_dist = igraph.Graph.shortest_paths(g, source = subj1, target = subj2)[0][0]

        # Kinship data only available for people at social distance 1
        if soc_dist == 1:
            for n_alter, n_relation in zip(n_alters, n_relations):
                if kd_df[((kd_df['nid'] == int(subj1[1:])) & (kd_df[n_alter] == int(subj2[1:])))][n_relation].values in kinship_values:
                    print('Subjects %s and %s are related. Excluding' % (subj1, subj2))
                    related_dyads.append(dyad)

                elif kd_df[(kd_df['nid'] == int(subj2[1:])) & (kd_df[n_alter] == int(subj1[1:]))][n_relation].values in kinship_values:
                    print('Subjects %s and %s are related. Excluding' % (subj1, subj2))
                    related_dyads.append(dyad)

    dyads_kin_excluded = [i for i in dyads if not i in related_dyads]

    return dyads_kin_excluded

def make_df(g, weighted, dyads, personality_df, kd_df, geographic_distance_df):
    dist_dict = {}
    for dyad in dyads:

        subj1 = dyad[0]
        subj2 = dyad[1]

        if weighted == True:
            soc_dist = igraph.Graph.shortest_paths(g, source = subj1, target = subj2, weights = 'weight')[0][0]
        else:
            soc_dist = igraph.Graph.shortest_paths(g, source = subj1, target = subj2, weights = None)[0][0]

        dist_dict[dyad] = {}
        dist_dict[dyad]['soc_dist'] = soc_dist
        connectomes_df = LoadPickle(f'{data_path}/subjects_connectomes.pkl')
        roi_list = [i for i in list(connectomes_df.columns) if 'edge' in i]
        subj1_vec = np.asarray(connectomes_df[connectomes_df['subject'] == subj1]).ravel()[1:]
        subj2_vec = np.asarray(connectomes_df[connectomes_df['subject'] == subj2]).ravel()[1:]

        this_dist = np.abs(subj1_vec - subj2_vec)

        for dist, edge in zip(this_dist, roi_list):
            dist_dict[dyad][edge] = dist

        # Age similarity
        subj1_age = kd_df[kd_df['nid'] == int(subj1[1:])]['age'].item()
        subj2_age = kd_df[kd_df['nid'] == int(subj2[1:])]['age'].item()
        age_dist = np.abs(subj1_age - subj2_age)
        dist_dict[dyad]['age_dist'] = float(age_dist)

        # Gender similarity
        subj1_gender = kd_df[kd_df['nid'] == int(subj1[1:])]['sex'].item()
        subj2_gender = kd_df[kd_df['nid'] == int(subj2[1:])]['sex'].item()
        if subj1_gender == subj2_gender:
            gender_similarity = 1
        else:
            gender_similarity = 0
        dist_dict[dyad]['same_gender'] = gender_similarity

        # Personality similarity
        # Fill in missing values
        personality_df['Personality_N'] = personality_df['Personality_N'].fillna(personality_df['Personality_N'].mean())
        # personality_df
        big5_traits = ['N', 'E', 'O', 'A', 'C']
        for trait in big5_traits:
            subj1_trait = personality_df[personality_df['nid'] == subj1]['Personality_%s' % trait].values[0]
            subj2_trait = personality_df[personality_df['nid'] == subj2]['Personality_%s' % trait].values[0]

            trait_dist = np.abs(subj1_trait - subj2_trait)
            dist_dict[dyad]['Personality_%s' % trait] = float(trait_dist)

        # Geographic distance
        phys_dist = geographic_distance_df[((geographic_distance_df['nid1'] == subj1) & (geographic_distance_df['nid2'] == subj2)) | ((geographic_distance_df['nid1'] == subj2) & (geographic_distance_df['nid2'] == subj1))]['walk distance (m)'].values[0]

        dist_dict[dyad]['phys_dist'] = phys_dist


    dist_df = pd.DataFrame(dist_dict).transpose()
    dist_df[roi_list] = -dist_df[roi_list] # Convert to neural similarity

    return dist_df

def regress_out_covariates(df, control_personality):
    if control_personality == True:
        regressor_cols = ['Personality_N', 'Personality_E', 'Personality_O', 'Personality_A', 'Personality_C', 'age_dist', 'same_gender']
    else:
        regressor_cols = ['age_dist', 'same_gender']

    regressors = df[regressor_cols]
    outcome_var = df['soc_dist']

    pipe = Pipeline(steps=[('scale', StandardScaler()),
                            ('lm', LinearRegression())])
    pipe.fit(regressors, outcome_var)
    predicted = pipe.predict(regressors)
    actual = outcome_var
    resid = actual - predicted
    df['soc_dist_cleaned'] = resid

    return df

def main(weighted, analysis, control_personality):
    kd_df = pd.read_csv(f'{data_path}/kinship_and_demographics_df.csv')
    personality_df = pd.read_csv(f'{data_path}/personality_df.csv')
    geographic_distance_df = pd.read_csv(f'{data_path}/geographic_distance_df.csv')
    connectomes_df = LoadPickle(f'{data_path}/subjects_connectomes.pkl')
    g = LoadPickle(f'../g_weighted-{weighted}_{analysis}.pkl')
    fmri_subjects = LoadPickle(f'{data_path}/fmri_subjects.pkl')
    dyads = exclude_distant_dyads(g, fmri_subjects)
    # dyads = exclude_kin(g, dyads, fmri_subjects, kd_df) # Optional

    df = make_df(g, weighted, dyads, personality_df, kd_df, geographic_distance_df)

    df_cleaned = regress_out_covariates(df, control_personality)
    df_cleaned.reset_index(inplace = True)
    df_cleaned.to_csv(f'../{analysis}_weighted-{weighted}_control-personality-{control_personality}.csv', index = False)

main(weighted = True, analysis = 'meet', control_personality = False)
