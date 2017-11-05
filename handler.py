

def handleData(data, seqLength, normalise, close):
    import numpy as np
    seqLength = seqLength + 1
    batchedData = []
    height = np.shape(data)[0]
    for i in range(height - seqLength + 1):
        batchedData.append(data[i:i + seqLength])
        
    origData = np.array(batchedData)
    norms = origData[:,0]
    
     
    
       
    if normalise:
        batchedData = normalise_sequence(batchedData)
    
    
    batchedData = np.array(batchedData)

    batchHeight = np.shape(batchedData)[0]
    
    trainHeight = int(batchHeight * 0.7)
    validHeight = int(batchHeight * 0.9)
    
    
    norms = norms[validHeight:]
    
  
    trainArray = batchedData[:trainHeight, :]
    validArray = batchedData[trainHeight:validHeight, :]
    testArray = batchedData[validHeight:, :]
    testOrigArray = origData[validHeight:,:]
    

    
    x_train = trainArray[:,:-1]
    y_train = trainArray[:,-1]
    
    x_valid = validArray[:,:-1]
    y_valid = validArray[:,-1]
    
    x_test = testArray[:,:-1]
    
    x_orig_test = testOrigArray[:,:-1]
    y_test = testArray[:,-1]
    
    

    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    x_orig_test = np.reshape(x_orig_test, (x_orig_test.shape[0], x_orig_test.shape[1], 1))
    
    if close:
        
        return x_orig_test, norms, x_train, y_train, x_valid, y_valid, x_test, y_test
        

    else:
        return x_train, y_train, x_valid, y_valid, x_test, y_test


def normalise_sequence(data):
    normalisedData = []
    for seq in data:
        normalisedSeq = [((float(v)/float(seq[0]))-1) for v in seq]
        normalisedData.append(normalisedSeq)
    
    return normalisedData
    

def denormalise_sequence(data, norms):
    denormalisedData = [((l+1) * (norms[i])) for i,l in enumerate(data)]
    
    return denormalisedData
   
def proportional_change(pred, x_close_test, x_orig_test):
    import numpy as np
    propPreds = []
    for i,item in enumerate(pred):
        avPrevVals = np.mean(x_close_test[i,:])
        propChange = item / avPrevVals
        avScaledPrevVals = np.mean(x_orig_test[i,:])
        predScaled = propChange * avScaledPrevVals
       
        propPreds.append(predScaled)
        
    return propPreds
    

def adjustedPreds(model, origData, testData):
    p = model.predict(testData)
    adjPreds = []
    for i, item in enumerate(p):
        percChange = (item - testData[i, -1, :]) / float(testData[i, -1, :])
        adjVal = (percChange * float(origData[i, -1, :])) + float(origData[i, -1, :])
        adjPreds.append(adjVal)
        
    return adjPreds
      
        
                            