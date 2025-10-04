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
    - Passengers who spent on at least one amenity are awake → CryoSleep = False
    - Remaining passengers are considered to be asleep → CryoSleep = True
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Convert column to boolean
        df['CryoSleep'] = df['CryoSleep'].astype(bool)
        
        # Spent on amenities
        indexes = df.query("(RoomService > 0) | (FoodCourt > 0) | (ShoppingMall > 0) | (Spa > 0) | (VRDeck > 0)").index
        df.loc[indexes, 'CryoSleep'] = df.loc[indexes, 'CryoSleep'].fillna('False')
        
        # Didn't spend and weren't children
        indexes = df.query("~((RoomService > 0) | (FoodCourt > 0) | (ShoppingMall > 0) | (Spa > 0) | (VRDeck > 0)) and (Age >= 13)").index
        df.loc[indexes, 'CryoSleep'] = df.loc[indexes, 'CryoSleep'].fillna('True')
        
        # Children
        df = df.dropna(subset='CryoSleep')
        
        return df
    
## HomePlanet
class HomePlanet(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in 'HomePlanet' based on CabinDeck.
    - Passengers in CabinDeck A, B, or C → HomePlanet = Europa
    - Passengers in CabinDeck G → HomePlanet = Earth
    - Remaining passengers with missing HomePlanet → "Unknown"
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Impute with HomePlanet = Europa
        indexes = df.query("(CabinDeck == 'A') | (CabinDeck == 'B') | (CabinDeck == 'C')").index
        df.loc[indexes, 'HomePlanet'] = df.loc[indexes, 'HomePlanet'].fillna('Europa')
        
        # Impute with HomePlanet = Earth
        indexes = df.query("CabinDeck == 'G'").index
        df.loc[indexes, 'HomePlanet'] = df.loc[indexes, 'HomePlanet'].fillna('Earth')
        
        # Fill the remaining with "Unknown"
        df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')
        
        return df

## Amenities
class Amenities(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in amenity columns based on CryoSleep status, Age, and HomePlanet.
    - Passengers who are awake and Age >= 13 → median of the amenity for their HomePlanet
    - Passengers who are asleep or Age < 13 → 0
    - Remaining missing values → median of each amenity
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        home_planets = ['Earth', 'Europa', 'Mars'] # exclude "Unknown" because they are uncertain
        
        for amenity in amenities:
            # Awake and Age >= 13 → impute with median by HomePlanet
            for home_planet in home_planets:
                mask = df.query(f"(HomePlanet == '{home_planet}') & (CryoSleep == False) & (Age >= 13)")
                amenity_median = mask[amenity].median()
                indexes = mask[amenity].isna().index
                df.loc[indexes, amenity] = df.loc[indexes, amenity].fillna(amenity_median)
                
            # Asleep or Age < 13 → 0
            indexes = df.query(f"(CryoSleep == True) | (Age < 13)")[amenity].isna().index
            df.loc[indexes, amenity] = df.loc[indexes, amenity].fillna(0)
        
        # Fill the remaining with the median of each amenity
        for amenity in amenities:
            amenity_median = df[amenity].median()
            df[amenity] = df[amenity].fillna(amenity_median)
        
        return df
    