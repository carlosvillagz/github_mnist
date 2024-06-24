import numpy as np
train_data = [[3,5],[1,7,2,8],[5,9,6]]
def vectorize_sequences(sequences, dimension=10): # sequences is equal to train_data a sequece of reviews in imdb
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1 # Assignment of Elements in the Matrix
    return results
x_train = vectorize_sequences(train_data)
x_train
