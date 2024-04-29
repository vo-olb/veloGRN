import pandas as pd
import numpy as np
from itertools import product, permutations
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector

def EarlyPrec(trueEdgesDF: pd.DataFrame, predEdgesDF: pd.DataFrame,
              weight_key: str = 'EdgeWeight', TFEdges: bool = True):
    """
        Computes early precision for a given set of predictions in the form of a DataFrame.
        The early precision is defined as the fraction of true positives in the top-k edges,
        where k is the number of edges in the ground truth network (excluding self loops).

    :param trueEdgesDF:   A pandas dataframe containing the true edges.
    :type trueEdgesDF: DataFrame

    :param predEdgesDF:   A pandas dataframe containing the edges and their weights from
        the predicted network. Use param `weight_key` to assign the column name of edge weights.
        Higher the weight, higher the edge confidence.
    :type predEdgesDF: DataFrame

    :param weight_key:   A str represents the column name containing weights in predEdgeDF.
    :type weight_key: str

    :param TFEdges:   A flag to indicate whether to consider only edges going out of TFs (TFEdges = True)
        or not (TFEdges = False) from evaluation.
    :type TFEdges: bool

    :returns:
        - Eprec: Early precision value
        - Erec: Early recall value
        - EPR: Early precision ratio
    """

    print("Calculating the EPR(early prediction rate)...")

    # Remove self-loops
    trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
    if 'Score' in trueEdgesDF.columns:
        trueEdgesDF = trueEdgesDF.sort_values('Score', ascending=False)
    trueEdgesDF = trueEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    trueEdgesDF.reset_index(drop=True, inplace=True)

    predEdgesDF = predEdgesDF.loc[(predEdgesDF['Gene1'] != predEdgesDF['Gene2'])]
    if weight_key in predEdgesDF.columns:
        predEdgesDF = predEdgesDF.sort_values(weight_key, ascending=False)
    predEdgesDF = predEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    predEdgesDF.reset_index(drop=True, inplace=True)

    uniqueNodes = np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']])
    if TFEdges:
        # Consider only edges going out of source genes
        print("  Consider only edges going out of source genes")

        # Get a list of all possible TF to gene interactions
        possibleEdges_TF = set(product(set(trueEdgesDF.Gene1), set(uniqueNodes)))   #笛卡尔积

        # Get a list of all possible interactions 
        possibleEdges_noSelf = set(permutations(uniqueNodes, r=2))  #两两边的排列组合

        # Find intersection of above lists to ignore self edges
        # TODO: is there a better way of doing this?
        possibleEdges = possibleEdges_TF.intersection(possibleEdges_noSelf)
        possibleEdges = pd.DataFrame(possibleEdges, columns=['Gene1', 'Gene2'], dtype=str)

        # possibleEdgesDict = {'|'.join(p): 0 for p in possibleEdges}
        possibleEdgesDict = possibleEdges['Gene1'] + "|" + possibleEdges['Gene2']

        trueEdges = trueEdgesDF['Gene1'].astype(str) + "|" + trueEdgesDF['Gene2'].astype(str)
        trueEdges = trueEdges[trueEdges.isin(possibleEdgesDict)]
        print("  {} TF Edges in ground-truth".format(len(trueEdges)))
        numEdges = len(trueEdges)

        predEdgesDF['Edges'] = predEdgesDF['Gene1'].astype(str) + "|" + predEdgesDF['Gene2'].astype(str)
        # limit the predicted edges to the genes that are in the ground truth
        predEdgesDF = predEdgesDF[predEdgesDF['Edges'].isin(possibleEdgesDict)]
        print("  {} Predicted TF edges are considered".format(len(predEdgesDF)))

        M = len(set(trueEdgesDF.Gene1)) * (len(uniqueNodes) - 1)        #笛卡尔积可能的边的数量，减去1是因为不考虑self_loop

    else:
        trueEdges = trueEdgesDF['Gene1'].astype(str) + "|" + trueEdgesDF['Gene2'].astype(str)
        trueEdges = set(trueEdges.values)
        numEdges = len(trueEdges)
        print("  {} edges in ground-truth".format(len(trueEdges)))

        M = len(uniqueNodes) * (len(uniqueNodes) - 1)

    if not predEdgesDF.shape[0] == 0:
        # Use num True edges or the number of
        # edges in the dataframe, which ever is lower
        maxk = min(predEdgesDF.shape[0], numEdges)
        edgeWeightTopk = predEdgesDF.iloc[maxk - 1][weight_key]

        nonZeroMin = np.nanmin(predEdgesDF[weight_key].values)
        bestVal = max(nonZeroMin, edgeWeightTopk)

        newDF = predEdgesDF.loc[(predEdgesDF[weight_key] >= bestVal)]
        predEdges = set(newDF['Gene1'].astype(str) + "|" + newDF['Gene2'].astype(str))
        print("  {} Top-k edges selected".format(len(predEdges)))
    else:
        predEdges = set([])

    if len(predEdges) != 0:
        intersectionSet = predEdges.intersection(trueEdges)
        print("  {} true-positive edges".format(len(intersectionSet)))
        Eprec = len(intersectionSet) / len(predEdges)
        Erec = len(intersectionSet) / len(trueEdges)
    else:
        Eprec = 0
        Erec = 0

    random_EP = len(trueEdges) / M
    EPR = Erec / random_EP
    return Eprec, Erec, EPR

def computeScores(trueEdgesDF: pd.DataFrame, predEdgesDF: pd.DataFrame,
                  weight_key: str = 'weight_abs', selfEdges: bool = True):
    """
        Computes precision-recall and ROC curves using scikit-learn
        for a given set of predictions in the form of a DataFrame.

    :param trueEdgesDF:   A pandas dataframe containing the true edges.
    :type trueEdgesDF: DataFrame

    :param predEdgesDF:   A pandas dataframe containing the edges and their weights from
        the predicted network. Use param `weight_key` to assign the column name of edge weights.
        Higher the weight, higher the edge confidence.
    :type predEdgesDF: DataFrame

    :param weight_key:   A str represents the column name containing weights in predEdgeDF.
    :type weight_key: str

    :param selfEdges:   A flag to indicate whether to include self-edges (selfEdges = True)
        or exclude self-edges (selfEdges = False) from evaluation.
    :type selfEdges: bool

    :returns:
        - prec: A list of precision values (for PR plot)
        - recall: A list of precision values (for PR plot)
        - fpr: A list of false positive rates (for ROC plot)
        - tpr: A list of true positive rates (for ROC plot)
        - AUPRC: Area under the precision-recall curve
        - AUROC: Area under the ROC curve
    """

    print("Calculating the AUPRC and AUROC...")

    if 'Score' in trueEdgesDF.columns:
        trueEdgesDF = trueEdgesDF.sort_values('Score', ascending=False)
    trueEdgesDF = trueEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    trueEdgesDF.reset_index(drop=True, inplace=True)

    if weight_key in predEdgesDF.columns:
        predEdgesDF = predEdgesDF.sort_values(weight_key, ascending=False)
    predEdgesDF = predEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    predEdgesDF.reset_index(drop=True, inplace=True)

    # Initialize dictionaries with all
    # possible edges
    if selfEdges:
        possibleEdges = list(product(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]),
                                     repeat=2))
    else:
        possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]),
                                          r=2))
        trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
        predEdgesDF = predEdgesDF.loc[(predEdgesDF['Gene1'] != predEdgesDF['Gene2'])]
        # constrain pred edges in possibleEdges
        s = set(possibleEdges)
        predEdgesDF = predEdgesDF[predEdgesDF.apply(lambda row: (row['Gene1'], row['Gene2']) in s, axis=1)]

    TrueEdgeDict = pd.DataFrame({'|'.join(p): 0 for p in possibleEdges}, index=['label']).T
    PredEdgeDict = pd.DataFrame({'|'.join(p): 0 for p in possibleEdges}, index=['label']).T

    # Compute TrueEdgeDict Dictionary
    # 1 if edge is present in the ground-truth
    # 0 if edge is not present in the ground-truth
    TrueEdgeDict.loc[np.array(trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']), 'label'] = 1
    PredEdgeDict.loc[np.array(predEdgesDF['Gene1'] + "|" + predEdgesDF['Gene2']), 'label'] = np.abs(
        predEdgesDF[weight_key].values)

    # Combine into one dataframe
    # to pass it to sklearn
    outDF = pd.DataFrame([TrueEdgeDict['label'].values, PredEdgeDict['label'].values]).T
    outDF.columns = ['TrueEdges', 'PredEdges']
    prroc = importr('PRROC')
    prCurve = prroc.pr_curve(scores_class0=FloatVector(list(outDF['PredEdges'].values)),
                             weights_class0=FloatVector(list(outDF['TrueEdges'].values)))

    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'],
                                     drop_intermediate=True, pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)

    return prec, recall, fpr, tpr, prCurve[2][0], auc(fpr, tpr)