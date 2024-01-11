from sklearn.datasets import load_boston
import pandas as pd
display=pd.options.display
display.max_columns=None
display.max_rows=None
display.width=None
display.max_colwidth=None

def get_data():
    """
    Get scikit-learn's boston data
    :return: boston data as DataFrame
    """
    data = load_boston()
    df = pd.DataFrame(data=data.data,
                      columns=data.feature_names)
    df.insert(loc=0, column="PRICE", value=data.target)
    return df