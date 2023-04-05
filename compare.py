import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def compare(X, models, skipIndices):
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
    instrumentDiffs : dict
        A dictionary of all the instruments disagreed upon by the models
    allInstProbs : dict
        A dictionary of all the instruments and their predictions

    
    """
    uncertaintyScores = {}  # dictionary of each track's uncertainty score by instrument
    allInstProbs = {}       # dictionary of the predictions for every instrument for each track
    # print("Called")

    for instrument in models:
        rfc = models[instrument][0]
        knn = models[instrument][1]

        instrPreds = {}   # a dict containing the predictions by each model
        trkUncertainties = {}

        # print(instrument, ": \t skip indices:", len(skipIndices[instrument]))

        for trk in range(len(X)):
            if trk not in skipIndices[instrument]:
                # print("New track")
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



def addRandomTracks(numRandom, numTrx, indexList):
    """
    Adds random track indices to an existing list.
    
    Parameters
    ----------
    numRandom : int
        Number of random indices to be added
    numTrx : int
        Total number of tracks
    indexList : list
        List of previously selected track indices

    Returns
    ----------
    indexList : list
        List of indices with random indices included.
    """
    i = 0
    rand_idx = np.random.randint(0, numTrx)

    while i < numRandom:
        if rand_idx not in indexList:
            indexList.append(rand_idx)
            i += 1
        rand_idx = np.random.randint(0, numTrx)

    return indexList


def trainModel(modelType, inst_num, X_train, X_test, Y_true_train, Y_true_test, Y_mask_train, Y_mask_test, Y_true_labeled=None, X_labeled=None):
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
    X_test : numpy.ndarray
        Testing data
    Y_true_train : numpy.ndarray
        Training data true values
    Y_true_test : numpy.ndarray
        Testing data true values
    Y_mask_train : numpy.ndarray
        Training data labels
    Y_mask_test : numpy.ndarray
        Testing data labels
    Y_true_labeled : numpy.ndarray, optional
        Values for labeled data
    X_labeled : numpy.ndarray, optional
        Labeled data
        
    Returns
    ----------
    model : scikitlearn model
        Trained model
    X_test_inst_sklearn : numpy.ndarray
        Data used for testing predictions
    Y_true_test_inst : numpy.ndarray
        Labels for test data

    """
    NUM_NEIGHBS = 10

    # isolate data that has been labeled as this instrument
    train_inst = Y_mask_train[:, inst_num] 
    test_inst = Y_mask_test[:, inst_num]

    # gets training data with labels for this instrument
    X_train_inst = X_train[train_inst]

    if X_labeled != None:

        X_train_new = np.append(X_train_inst, X_labeled, axis=0)
    
        # averages features over time
        X_train_inst_sklearn = np.mean(X_train_new, axis=1)
    else:
        X_train_inst_sklearn = np.mean(X_train_inst, axis=1)


    # labels instrument as present if value over 0.5
    Y_true_train_inst = Y_true_train[train_inst, inst_num] >= 0.5

    # Repeat slicing for test
    X_test_inst = X_test[test_inst]
    X_test_inst_sklearn = np.mean(X_test_inst, axis=1)
    Y_true_test_inst = Y_true_test[test_inst, inst_num] >= 0.5

    # Initialize a new classifier
    if modelType == "rfc":
        model = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=0)
    else:
        model = KNeighborsClassifier(n_neighbors=NUM_NEIGHBS, weights="distance")


    if Y_true_labeled is not None:
        Y_true_train_labeled = Y_true_labeled[:, inst_num] >= 0.5
        Y_true_train_combined = np.append(Y_true_train_inst, Y_true_train_labeled, axis=0)

        # Fit model
        model.fit(X_train_inst_sklearn, Y_true_train_combined)
    else:
        model.fit(X_train_inst_sklearn, Y_true_train_inst)

    return model, X_test_inst_sklearn, Y_true_test_inst

    

