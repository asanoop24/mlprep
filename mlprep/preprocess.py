import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class CategoryEncoder:
    '''
    Transform method returns a pandas dataframe object with original categorical columns converted to encoded columns

    labelEncoding : boolean  -> set to True If the categorical columns are to be label encoded
    oneHotEncoding : boolean -> set to True If the categorical columns are to be one hot encoded (using pandas.get_dummies method)
    dropFirst : boolean      -> set to True if first column is to be dropped (usually to avoid multi-collinearity) post one hot encoding. Doesn't matter if oneHotEncoding = False

    df : pandas.DataFrame() -> dataframe object that needs to be encoded
    catCols : list          -> list of the categorical columns that need to be encoded
    '''
    def __init__(self,labelEncoding=True,oneHotEncoding=False,dropFirst=False):
        self.labelEncoding = labelEncoding
        self.oneHotEncoding = oneHotEncoding
        self.dropFirst = dropFirst
        self.labelEncoder = {}
        self.oneHotEncoder = {}
        
    def fit(self,df,catCols=[]):
        df1 = df.copy()
        if self.labelEncoding:
            for col in catCols:
                labelEncoder = LabelEncoder()
                labelEncoder.fit(df1.loc[:,col].astype(str))
                df1.loc[:,col] = labelEncoder.transform(df1.loc[:,col])
                self.labelEncoder[col] = labelEncoder.classes_
                
        if self.oneHotEncoding:
            for col in catCols:
                cats = sorted(df1.loc[:,col].value_counts(dropna=True).index)
                self.oneHotEncoder[col] = cats
        
    def transform(self,df,catCols=[]):
        df1 = df.copy()
        if self.labelEncoding:
            for col in catCols:
                labelEncoder = self.labelEncoder[col]
                labelEncoder = {v:i for i,v in enumerate(labelEncoder.tolist())}
                df1.loc[:,col] = df1.loc[:,col].map(labelEncoder)
                
        if self.oneHotEncoding:
            for col in catCols:
                oneHotEncoder = self.oneHotEncoder[col]
                df1.loc[:,col] = df1.loc[:,col].astype(pandas.CategoricalDtype(categories=oneHotEncoder))
            df1 = pandas.get_dummies(df1,columns=catCols,drop_first=self.dropFirst)
        
        return df1


class Normalizer:
    
    def __init__(self,method='mms'):
        self.method = method
        self.methods = {
            'mms':MinMaxScaler,
            'ss': StandardScaler
            }
        self.normalizers = {}
        
    def fit(self,df,cols):
        df1 = df.copy()
        for col in cols:
            if df1[col].dtype == 'object': continue
            self.normalizers[col] = self.methods[self.method]()
            self.normalizers[col].fit(df1.loc[:,[col]])
    
    def transform(self,df,cols):
        df1 = df.copy()
        for col in cols:
            if df1[col].dtype == 'object': continue
            df1.loc[:,[col]] = self.normalizers[col].transform(df1.loc[:,[col]])
        return df1
