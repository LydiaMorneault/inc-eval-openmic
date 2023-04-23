import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def compare(X, models):
    """
    Prioritizes tracks by highest to lowest uncertainty using algorithmic disagreement. 

    In this function, uncertainty is calculated as follows: 
        1. Check if the models disagree on the existence of the instrument in the track. 
            Rounding the prediction down to 0 means no instrument, and up to 1 means the instrument is present.
        2. On tracks where the models disagree, get the difference between the predictions for that instrument
        3. The uncertainty score = the average of the differences for each track across instruments
    
    Parameters
    ----------
    X : numpy.ndarray
        The data to be prioritzed
    models : dict
        The trained instrument classifiers organized by instrument. The format is "instrument": [rfc,knn]

    Returns
    ----------
    uncertaintyScores : dict
        A dictionary of all tracks and their average algorithmic difference
    allInstProbs : dict
        A dictionary of all the instruments and their predictions

    
    """
    uncertaintyScores = {}  # dictionary of each track's uncertainty score by instrument
    allInstProbs = {}       # dictionary of the predictions for every instrument for each track

    for instrument in models:
        print(instrument, "starting")

        rfc = models[instrument][0]
        knn = models[instrument][1]

        instrPreds = {}   # a dict containing the predictions by each model
        trkUncertainties = {}


        for trk in range(len(X)):
        # for trk in range(100):
            feature_mean = np.mean(X[trk], axis=0, keepdims=True)

            # Each model makes a prediction
            rfcPred = rfc.predict_proba(feature_mean)[0,1]
            knnPred = knn.predict_proba(feature_mean)[0,1]

            instrPreds[trk] = [rfcPred, knnPred]

            # If this track already has an uncertainty score, add to it
            trkUncertainties[trk] = abs(rfcPred - knnPred)
        
        
        # Sort the dictionary to get the highest uncertainty score    
        sortedTrx = dict(sorted(trkUncertainties.items(), key=lambda item:item[1], reverse=True))
        
        uncertaintyScores[instrument] = sortedTrx
        allInstProbs[instrument] = instrPreds

    return uncertaintyScores, allInstProbs



# def addRandomTracks(numRandom, numTrx, indexList, instrumentProbs):
#     """
#     Adds random track indices to an existing list.
    
#     Parameters
#     ----------
#     numRandom : int
#         Number of random indices to be added
#     numTrx : int
#         Total number of tracks
#     indexList : list
#         List of previously selected track indices

#     Returns
#     ----------
#     indexList : list
#         List of indices with random indices included.
#     """
#     i = 0
#     rand_idx = np.random.randint(0, numTrx)

#     while i < numRandom:
#         if (rand_idx not in indexList) and (rand_idx in instrumentProbs):
#             indexList.append(rand_idx)
#             i += 1
#         rand_idx = np.random.randint(0, numTrx)

#     return indexList


def trainModel(modelType, inst_num, X_train, Y_true_train, Y_mask_train, Y_true_labeled=None, X_labeled=-1):
    """
    Trains either a Random Forest or K-Nearest Neighbors scikitlearn model. 
    
    Parameters
    ----------
    modelType : str
        'rfc' for Random Forest Classifier 
        'knn' for K-Nearest Neighbors
    inst_num : int
        Instrument class number, based on the classmap 
    X_train : numpy.ndarray
        Training data
    Y_true_train : numpy.ndarray
        Actual values for training data
    Y_mask_train : numpy.ndarray
        Indicates whether a label exists for each instrument in the track
    Y_true_labeled : numpy.ndarray, optional
        Actual values for labeled data
    X_labeled : numpy.ndarray, optional
        Labeled data
        
    Returns
    ----------
    model : scikitlearn model
        Trained model

    """
    NUM_NEIGHBS = 20
    props = {}

    ###########################################################################
    # SUB-SAMPLE DATA - isolate the data for which we have annotations

    # isolate data that has been labeled as this instrument
        # Use the Y_mask_train array to slice out only the training examples
        # for which we have annotations for the given class
    train_inst = Y_mask_train[:, inst_num] 

    X_train_inst = X_train[train_inst]
    
    ###########################################################################
    # SIMPLIFY DATA - average over time

    if X_labeled != -1:
        # combine the training and labeled sets
        X_train_new = np.append(X_train_inst, X_labeled, axis=0)

        # averages features over time
        X_train_inst_sklearn = np.mean(X_train_new, axis=1)
    else:
        X_train_inst_sklearn = np.mean(X_train_inst, axis=1)

    ###########################################################################
    # INITIALIZE CLASSIFIER 
    if modelType == "rfc":
        model = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=0)
    else:
        model = KNeighborsClassifier(n_neighbors=NUM_NEIGHBS, weights="distance")

    # labels instrument as present if value over 0.5
    Y_true_train_inst = Y_true_train[train_inst, inst_num] >= 0.5
    props["train"] = [len(Y_true_train), len(Y_true_train_inst)]


    # Again, we slice the labels to the annotated examples
    # We thresold the label likelihoods at 0.5 to get binary labels
    if Y_true_labeled is not None:
        Y_true_train_labeled = Y_true_labeled[:, inst_num] >= 0.5
        Y_true_train_combined = np.append(Y_true_train_inst, Y_true_train_labeled, axis=0)

        # print(inst_num, "Y true train new length", Y_true_train_combined.shape, Y_true_train_inst.shape, len(Y_true_train_labeled))

        props["labeled"] = [len(Y_true_labeled), len(Y_true_train_labeled)]

        # Fit model
        model.fit(X_train_inst_sklearn, Y_true_train_combined)
    else:
        model.fit(X_train_inst_sklearn, Y_true_train_inst)

    return model, props

    

