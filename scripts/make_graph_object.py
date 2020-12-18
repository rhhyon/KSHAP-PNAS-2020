import igraph
import pickle

data_path = '../data'

def LoadPickle(infile):
    with open(infile, 'rb') as f:
        outfile = pickle.load(f)
        return outfile

def SavePickle(infile, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(infile, f, protocol = 2)

def fill_missing_edges(g):
    egos = LoadPickle(f'{data_path}/survey_participants.pkl')
    non_mutual = []
    for e in range(0, len(g.es)):
        target = str(g.vs(g.es[e].target)['name'][0])
        target_id = g.es[e].target
        source = str(g.vs(g.es[e].source)['name'][0])
        source_id = g.es[e].source

        if igraph.Graph.is_mutual(g, edges = e) == False:
            non_mutual.append((source, target))
            existing_weight = g.es[e]['weight']

            if target not in egos: # if target didn't participate in the survey
                g.add_edge(target, source, weight = existing_weight)
            else: # if the target did participate in the survey
                g.add_edge(target, source, weight = 0)
    return g

def make_graph_object(weighted, analysis):
    if weighted == True:
        if 'communication' in analysis:
            g = igraph.Graph.Read_Ncol(f'{data_path}/edgelist_weighted_comm.csv', weights = True, directed = True)
        elif 'meeting' in analysis:
            g = igraph.Graph.Read_Ncol(f'{data_path}/edgelist_weighted_meet.csv', weights = True, directed = True)
        else:
            g = igraph.Graph.Read_Ncol(f'{data_path}/edgelist_weighted_feel.csv', weights = True, directed = True)

        g = fill_missing_edges(g)
        g = igraph.Graph.as_undirected(g, mode = 'collapse', combine_edges = 'mean')
        w = g.es['weight']
        w = [1/i for i in w]
        g.es['weight'] = w

    else:
        g = igraph.Graph.Read_Ncol(f'{data_path}/edgelist_unweighted.csv', weights = False, directed = True)
        g = igraph.Graph.as_undirected(g, mode = edge_type, combine_edges = 'mean')

    g.add_vertex(name = 's304065') #  add fMRI isolate

    return g


def main(weighted, analysis):
    g = make_graph_object(weighted, analysis)
    SavePickle(g, f'../g_weighted-{weighted}_{analysis}.pkl')

main(weighted = True, analysis = 'meet')
