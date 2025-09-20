from sklearn.base import BaseEstimator, TransformerMixin

# 1. Feature Extraction
class FeatureExtraction(BaseEstimator, TransformerMixin):
    """
    Extracts new features from PassengerId, Cabin and Name columns.
    Transforms them into GroupNumber, NumberWithinGroup, CabinDeck,
    CabinNum, CabinSide and LastName.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # 'PassengerId'
        split = df['PassengerId'].str.split('_').str
        df['GroupNumber'] = split[0].astype(int)
        df['NumberWithinGroup'] = split[1].astype(int)
        df = df.drop(columns='PassengerId')

        # 'Cabin'
        split = df['Cabin'].str.split('/').str
        df['CabinDeck'] = split[0]
        df['CabinNum'] = split[1] # as string because of NaN values
        df['CabinSide'] = split[2]
        df = df.drop(columns='Cabin')

        # 'Name'
        df['LastName'] = df['Name'].str.split().str[1]
        df = df.drop(columns='Name')

        return df
    