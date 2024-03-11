import sys
import pandas as pd
import os
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath: str):# -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """
    Load data from a SQLite database file.

    Parameters
    ----------
    database_filepath : str
        The filepath of the SQLite database file.

    Returns
    -------
    X : pd.Series
        The input messages.
    Y : pd.DataFrame
        The target categories.
    category_names : List[str]
        The names of the target categories.
    """
    # Create a database engine using the provided filepath
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # Generate a table name by removing the ".db" extension from the database filename
    table_name = os.path.basename(database_filepath).replace(".db", "") + "_table"

    # Read the table with the generated name from the database into a DataFrame
    df = pd.read_sql_table(table_name, engine)

    # Extract the 'message' column as the input X
    X = df['message']

    # Extract all columns starting from the 5th column as the target Y
    Y = df.iloc[:,4:]

    # Get the names of the target categories
    category_names = Y.columns

    return X, Y, category_names
    


def tokenize(text: str):# -> list[str]:
    """
    Tokenizes the input text into words and applies lemmatization.

    Parameters:
    text (str): The input text to be tokenized.

    Returns:
    List[str]: A list of cleaned tokens after lemmatization.
    """
    # Tokenize the text into words
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for tok in tokens:
        # Lemmatize each token, convert to lowercase, and remove leading/trailing whitespaces
        cleaned_tok = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(cleaned_tok)

    return cleaned_tokens


def build_model(clf = AdaBoostClassifier()) -> GridSearchCV:
    """
    Build a machine learning model using a pipeline and grid search.

    Parameters
    ----------
    clf : BaseEstimator, optional
        The classifier to be used in the model, by default AdaBoostClassifier()

    Returns
    -------
    GridSearchCV
        The trained model with the best parameters found by grid search.
    """
    # Define the pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])

    # Define the parameters for grid search
    parameters = {
        'clf__estimator__learning_rate': [0.5, 1.0],
        'clf__estimator__n_estimators': [10, 20]
    }

    # Perform grid search with cross-validation
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 

    return cv
    


def evaluate_model(model: object, X_test: object, Y_test: object, category_names: list) -> None:
    """
    Evaluate the performance of a model by printing the classification report.

    Parameters
    ----------
    model : object
        The trained model to evaluate.
    X_test : object
        The input features for testing.
    Y_test : object
        The true labels for testing.
    category_names : list
        The names of the categories for classification.

    Returns
    -------
    None
    """
    # Predict the labels for the test data
    Y_pred_test = model.predict(X_test)

    # Print the classification report
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))
    

def save_model(model: object, model_filepath: str) -> None:
    """
    Save a machine learning model to a file using pickle.

    Parameters
    ----------
    model : object
        The machine learning model to be saved.
    model_filepath : str
        The file path where the model will be saved.

    Returns
    -------
    None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()