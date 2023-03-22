import numpy as np

def compare(X, models, batch=50):
    """
    Prioritizes tracks by highest to lowest uncertainty using algorithmic disagreement.
    
    Parameters
    ----------
    X : numpy.ndarray
        The data to be prioritzed
    models : dict
        The trained instrument classifiers organized by instrument. The format is "instrument": [rfc,knn]
    batch : int, optional
        The number of tracks to be prioritized at once

    Returns
    ----------
    trackUncertScores : dict
        A dictionary of all tracks and their average algorithmic difference
    instrumentDiffs : dict
        A dictionary of all the instruments disagreed upon by the models
    allInstProbs : dict
        A dictionary of all the instruments and their predictions

    
    """
    trackUncertScores = {}  # dictionary of all tracks which will contain the average differences
    instrumentDiffs = {}    # this keeps record of all the instruments disagreed upon
    allInstProbs = {}

    for i in range(batch):    # TODO: Change to full unlabeled set when ready
        track = X[i]
        feature_mean = np.mean(track, axis=0, keepdims=True)

        instrDict = {}
        allInsts = {}

        # Predict for each instrument
        for instrument in models:
            rfc = models[instrument][0]
            knn = models[instrument][1]

            rfcPred = rfc.predict_proba(feature_mean)[0,1]
            knnPred = knn.predict_proba(feature_mean)[0,1]

            allInsts[instrument] = [rfcPred, knnPred]

            # Check if the models agree that the instrument is present or not. 
            # A score of over 0.5 indicates the instrument is present
            if round(rfcPred) != round(knnPred):
                instrDict[instrument] = [rfcPred, knnPred]

                if trackUncertScores.get(i):
                    trackUncertScores[i] = trackUncertScores[i] + abs(rfcPred - knnPred)
                else: 
                    trackUncertScores[i] = abs(rfcPred - knnPred)

                # print('P[{:18s}=1] = RF: {:.3f}, KNN: {:.3f}'.format(instrument, rfc.predict_proba(feature_mean)[0,1], knn.predict_proba(feature_mean)[0,1]))

        # save the instrument differences 
        instrumentDiffs[i] = instrDict
        allInstProbs[i] = allInsts

        # average the differences to get uncertainty score
        if trackUncertScores.get(i):
            trackUncertScores[i] = trackUncertScores[i] / len(instrDict)
            # print(trackUncertScores[i], "/", len(instrDict))

    # # Sort the dictionary to get the highest uncertainty score    
    # sorted_dict = dict(sorted(trackUncertScores.items(), key=lambda item:item[1], reverse=True))

    return trackUncertScores, instrumentDiffs, allInstProbs

