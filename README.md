# KSHAP-PNAS-2020

This repository hosts all of the data and code for "Similarity in functional brain connectivity at rest predicts interpersonal closeness in the social network of an entire village" by Ryan Hyon, Yoosik Youm, Junsol Kim, Jeanyung Chey, Seyul Kwak, and Carolyn Parkinson (https://doi.org/10.1073/pnas.2013606117). Please email me at rhyon at ucla d0t edu if you have any questions!

## Data files
edges_system_dictionary.pkl - Dictionary containing brain network names and coordinates associated with each functional connectivity edge.  
fmri_subjects.pkl - List of fMRI subjects.  
geographic_distance_df.csv - Dataframe containing geographic proximity data.  
kinship_and_demographics_df.csv - Dataframe containing kinship and demographics data.  
personality_df.csv - Dataframe containing personality data.  
social_network_data.csv - Data to be used to create edgelists.  
subjects_connectomes.pkl - Functional connectomes for each subject.  
survey_participants.pkl - List of individuals that participated in the social network survey.  
systems_sym_flat.pkl - Array containing brain network names in the order of the functional connectivity edges.  



## Scripts files
extract_PLSC_for_moderation.py - Extracts the primary PLS component to be used as the moderating variable in the moderation analysis.  
make_df.py - Creates master dataframe containing neural similarity predictors and social network proximity data.  
make_df_systems_rois.py - Creates master dataframe containing within/between brain network predictors.  
make_edgelist.py - Creates edgelist.  
make_graph_object.py - Creates igraph social network object.  
mlm.R - Runs statistics for MLM analyses.  
moderation.R - Runs statistics for moderation analyses.  
permute_dfs.py - Permutes data to be used for permutation testing.  
permute_predictive_models.py - Runs predictive modeling using permuted data.  
predict.py - Runs predictive modeling.  
stats.py - Calculates statistics from permutation testing.  
