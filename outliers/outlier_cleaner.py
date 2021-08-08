#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    ### your code goes here
    error = (net_worths - predictions)**2
    data_with_error = np.transpose([ages, net_worths, error])[0]
    
    colIndexForError = 2
    data_with_error_sorted = data_with_error[data_with_error[:,colIndexForError].argsort()]
    
    len_cleaned_data = int(0.9*len(data_with_error))
    cleaned_data = data_with_error_sorted[:len_cleaned_data,:]
    
    return list(map(tuple, cleaned_data))

