import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.datasets import cifar10, imdb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def preprocess_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Select classes 3 and 5
    mask_train = np.logical_or(y_train == 3, y_train == 5).ravel()
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    mask_test = np.logical_or(y_test == 3, y_test == 5).ravel()
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]
    
    # Limit to first 1000 training samples
    X_train = X_train[:10]
    y_train = y_train[:10]
    X_test = X_test[:10]
    y_test = y_test[:10]

    # Change labels to binary
    y_train = np.where(y_train == 3, 0, 1)
    y_test = np.where(y_test == 3, 0, 1)

    # Convert to grayscale
    X_train = tf.image.rgb_to_grayscale(X_train).numpy()
    X_test = tf.image.rgb_to_grayscale(X_test).numpy()

    # Normalize by dividing by 255
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Further normalize by subtracting mean and dividing by standard deviation
    mean_image = np.mean(X_train, axis=0)
    std_image = np.std(X_train, axis=0)
    X_train = (X_train - mean_image) / std_image
    X_test = (X_test - mean_image) / std_image

    return X_train, y_train, X_test, y_test

def preprocess_imdb():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz",num_words=None,skip_top=0, maxlen=None,seed=1,start_char=1,oov_char=2,index_from=3)
    # Reverse the IMDB dictionary index to get the mapping from index to word
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    # Decode reviews into texts
    decoded_train = [" ".join([reverse_word_index.get(i - 3, "?") for i in review]) for review in X_train] #Why i - 3? - the first three indices are reserved for special values: sequence start, unknown word and sequence end, so the indexes of the actual words in the dictionary are shifted by 3.
    decoded_test = [" ".join([reverse_word_index.get(i - 3, "?") for i in review]) for review in X_test]

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(decoded_train)
    X_test = vectorizer.transform(decoded_test)
    
    # Limit to first 1000 training samples
    X_train = X_train[:20]
    y_train = y_train[:20]
    X_test = X_test[:10]
    y_test = y_test[:10]

    return X_train, y_train, X_test, y_test

def preprocess_breast_cancer():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit on training set only and transform both train and test sets
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test