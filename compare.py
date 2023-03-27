import numpy as np

def compare(X, models, batch=50):
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
    batch : int, optional
        The number of tracks to be prioritized at once

    Returns
    ----------
    uncertaintyScores : dict
        A dictionary of all tracks and their average algorithmic difference
    instrumentDiffs : dict
        A dictionary of all the instruments disagreed upon by the models
    allInstProbs : dict
        A dictionary of all the instruments and their predictions

    
    """
    uncertaintyScores = {}  # dictionary of each track's uncertainty score
    instrumentDiffs = {}    # dictionary of the instruments disagreed upon for each track and their predicted values
    allInstProbs = {}       # dictionary of the predictions for every instrument for each track

    for trk in range(batch):  

        # Calculate the mean features for the track at X[trk]
        feature_mean = np.mean(X[trk], axis=0, keepdims=True)

        trkInstruPreds = {}     # a dict containing the evaluations for each instrument by each model
        trkDisagreements = {}   # a dict containing only the instruments that were disagreed upon

        # Evaluate this track by each instrument class
        for instrument in models:

            # Get the models
            rfc = models[instrument][0]
            knn = models[instrument][1]

            # Each model makes a prediction
            rfcPred = rfc.predict_proba(feature_mean)[0,1]
            knnPred = knn.predict_proba(feature_mean)[0,1]

            # Save each of the models' predictions
            trkInstruPreds[instrument] = [rfcPred, knnPred]

            # Check if the models agree that the instrument is present or not. 
            # A score of over 0.5 indicates the instrument is present
            if round(rfcPred) != round(knnPred):
                trkDisagreements[instrument] = [rfcPred, knnPred]

                # If this track already has an uncertainty score, add to it
                if uncertaintyScores.get(trk):
                    uncertaintyScores[trk] = uncertaintyScores[trk] + abs(rfcPred - knnPred)
                else: 
                    uncertaintyScores[trk] = abs(rfcPred - knnPred)
        
        # average the uncertainties across instruments
        if uncertaintyScores.get(trk):
            uncertaintyScores[trk] = uncertaintyScores[trk] / len(trkDisagreements)

        # save the instrument differences 
        instrumentDiffs[trk] = trkDisagreements
        allInstProbs[trk] = trkInstruPreds



    # Sort the dictionary to get the highest uncertainty score    
    sortedTrx = dict(sorted(uncertaintyScores.items(), key=lambda item:item[1], reverse=True))

    return sortedTrx, instrumentDiffs, allInstProbs


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