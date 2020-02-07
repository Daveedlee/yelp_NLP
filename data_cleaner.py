import pandas as pd


def overall_cleaner(df, list_of_columns):
    """
    takes a dataframe and list of columns and returns a new dataframe only with those columns
    df = pandas data frame
    list_of_columns = a list of columns
    """
    new_df = pd.DataFrame(None)
    for i in list_of_columns:
        new_df[i] = df.loc[:,i]
    return new_df