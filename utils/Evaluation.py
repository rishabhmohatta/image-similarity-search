import numpy as np
import pandas as pd
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
def get_labels(query_image,query_GT,trained_GT,distance,indices):
    actual_label = []
    predictions = []
    # predicted_label
    num_queries = query_image.shape[0]
    for i in range(num_queries):
        # print([query_GT[i]]*len(indices[i]))
        # break
        actual_label.extend([query_GT[i]]*len(indices[i]))
        # print(len(actual_label),actual_label)
        # print(indices[i])
        predicted_labels = [trained_GT[idx] for idx in indices[i]]
        # print(predicted_labels)
        predictions.extend(predicted_labels)
        # print(actual_label,predictions)
    
    return np.array(actual_label),np.array(predictions)