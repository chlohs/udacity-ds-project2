import sys
import pandas as pd
import os
from sqlalchemy import create_engine

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load messages and categories datasets and merge them on 'id' column.

    Parameters
    ----------
    messages_filepath : str
        Filepath of the messages dataset.
    categories_filepath : str
        Filepath of the categories dataset.

    Returns
    -------
    pd.DataFrame
        Merged dataframe of messages and categories datasets.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)  
    categories = pd.read_csv(categories_filepath)

    # merge messages and categories datasets on 'id' column
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by splitting the 'categories' column into separate columns,
    renaming the columns, converting the values to numeric, dropping unnecessary columns,
    and dropping duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the 'categories' column.

    Returns
    -------
    pd.DataFrame
        The cleaned dataframe.

    """
    # Split the 'categories' column into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # Extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # Rename the columns of 'categories'
    categories.columns = category_colnames

    # Convert the values in each column to the last character of the string
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]

        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop the original 'categories' column from the dataframe
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Drop rows and columns with no additional value
    df = df.drop('child_alone', axis=1)
    df = df[df['related'] != 2]

    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
    Save a pandas DataFrame to a SQLite database.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be saved.
    database_filename : str
        The filename of the SQLite database.

    Returns
    -------
    None
    """
    # Create a SQLAlchemy engine with the specified database filename
    engine = create_engine('sqlite:///{}'.format(database_filename))

    # Generate a table name by removing the ".db" extension from the database filename
    table_name = os.path.basename(database_filename).replace(".db", "") + "_table"

    # Save the DataFrame to the SQLite database with the specified table name
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()