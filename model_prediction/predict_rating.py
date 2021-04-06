from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import sys


def main(args):
    model = load_model('lstm_model.h5')

    DATA_FILE_PATH = args[1]
    with open(DATA_FILE_PATH, 'r') as csv_file:
        df = pd.read_csv(
            csv_file, 
            usecols=['reviews.title', 'reviews.text'])
    df.rename(
        columns = {
            'reviews.rating': 'rating',
            'reviews.title': 'title',
            'reviews.text': 'text'
        }, 
        inplace = True)
    df = df.drop_duplicates()
     
    df['text'] = df['text'].apply(
        (lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    tokenizer = Tokenizer(num_words=2500, lower=True,split=' ')
    tokenizer.fit_on_texts(df['text'].values)
    X = tokenizer.texts_to_sequences(df['text'].values)
    X = pad_sequences(X)
    
    y_predicted = model.predict(X)
    y_predicted = y_predicted.argmax(axis=1)
    readable_y_predicted = []
    for prediction in y_predicted:
        readable_y_predicted.append(prediction + 1)
    df_pred = pd.DataFrame({'reviews.rating': readable_y_predicted})
    
    output_csv = df_pred.to_csv(args[2], index=False)


if __name__ == '__main__':
    main(sys.argv)

