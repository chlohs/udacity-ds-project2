# Disaster Response Pipeline Project

## Project Overview
This project is part of the Data Science Nanodegree Program by Udacity in collaboration with Appen (formerly Figure 8). The dataset contains pre-labeled tweets and messages from real-life disaster events. The aim is to build a Natural Language Processing (NLP) model that categorizes messages to aid in disaster response efforts.

The project includes a web application where an emergency worker can input a new message and receive classification results across multiple categories. This application also provides visualizations of the training dataset.

## Project Components
The project is divided into the following key components:

### 1. ETL Pipeline
A Python script, `process_data.py`, performs the following:

- Loads the messages and categories datasets.
- Merges the two datasets.
- Cleans the data.
- Stores it in a SQLite database.

### 2. ML Pipeline
A Python script, `train_classifier.py`, executes the following:

- Loads data from the SQLite database.
- Splits the dataset into training and test sets.
- Builds a text processing and machine learning pipeline.
- Trains and tunes a model using GridSearchCV.
- Outputs results on the test set.
- Exports the final model as a pickle file.

### 3. Flask Web App
The Flask web app enables users to enter disaster messages and view the classification results in various categories. The app also contains data visualizations that provide insights into the training dataset.

## Installation
To run the web application, you'll need to install the necessary libraries. This project was written in Python 3, and the following libraries are required:

- Flask
- Pandas
- NumPy
- Scikit-Learn
- SQLAlchemy
- Jupyter Notebook
- Plotly
- NLTK

You can install these libraries using `pip`:

```bash
pip install flask pandas numpy scikit-learn sqlalchemy jupyter plotly nltk
```
## Instructions
Run the following commands in the project's root directory to set up your database and model.

- To run the ETL pipeline that cleans data and stores it in a database: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- To run the ML pipeline that trains the classifier and saves it: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
- Run the following command in the app's directory to run your web app: python app/run.py

Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements
This project is licensed under the terms of the MIT license. Acknowledgements to Udacity for the project design and to Appen for providing the dataset.


