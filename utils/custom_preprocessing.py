from sklearn.base import BaseEstimator, TransformerMixin

# 1. Feature Extraction
class FeatureExtraction(BaseEstimator, TransformerMixin):
    """
    Extracts new features from 'PassengerId', 'Cabin' and 'Name'.
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

# 2. Handle Missing Values
## CryoSleep
class CryoSleep(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in 'CryoSleep' based on amenity usage.
    - Passengers who spent on at least one amenity are awake -> CryoSleep = False
    - Remaining passengers are considered to be asleep -> CryoSleep = True
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Spent on amenities
        indexes = df.query("(RoomService > 0) | (FoodCourt > 0) | (ShoppingMall > 0) | (Spa > 0) | (VRDeck > 0)").index
        df.loc[indexes, 'CryoSleep'] = df.loc[indexes, 'CryoSleep'].fillna('False')
        
        # Didn't spend and weren't children
        indexes = df.query("~((RoomService > 0) | (FoodCourt > 0) | (ShoppingMall > 0) | (Spa > 0) | (VRDeck > 0)) and (Age >= 13)").index
        df.loc[indexes, 'CryoSleep'] = df.loc[indexes, 'CryoSleep'].fillna('True')
        
        # Children
        df = df.dropna(subset='CryoSleep')
        
        return df
    