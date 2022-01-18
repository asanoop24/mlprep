# 'Droid Sans Mono', 'monospace', monospace, 'Droid Sans Fallback'

import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Bucketing categorical values
def bucketValues(col,df,byCount=False,byList=False,countThreshold=0,inList=None,outList=None,bucketValue='OTHERS'):
    df1 = df.copy()
    if byCount:
        allValues = df1[col].value_counts()
        inList = df1[col].value_counts()[df1[col].value_counts() < countThreshold].index.tolist()
        df1[col] = df1[col].map(lambda x: bucketValue if x in inList else x)
        return df1
    elif byList:
        if inList is not None:
            df1[col] = df1[col].map(lambda x: x if x in inList else bucketValue)
            return df1
        elif outList is not None:
            df1[col] = df1[col].map(lambda x: bucketValue if x in outList else x)
            return df1
    return df1


# Missing values summary
def missingSummary(df:pandas.DataFrame):
    df1 = df.copy()
    missingSummary = {}
    missingSummary = {col:len(df1[df1[col].isnull()][col]) for col in df1.columns.tolist()}
    missingSummary = dict(sorted(missingSummary.items(), key=lambda x: x[1], reverse=True))
    return missingSummary


# Missing value treatment
def missingTreatment(df,col,fill=True,fillValue='mean',drop=False):
    '''
    Replacement (for numerical columns): 'mean', 'median', 'mode', 'regression', 'custom'
    Replacement (for categorical columns): 'mode', 'knn', 'classification', 'custom'
    '''
    df1 = df.copy()
    if fill:
        treatment = {}
        if fillValue=='mean': treatment['mean'] = df1[-df1[col].isnull()][col].mean()
        elif fillValue=='median': treatment['median'] = df1[-df1[col].isnull()][col].median()
        elif fillValue=='mode': treatment['mode'] = df1[-df1[col].isnull()][col].mode()[0]
        else:
            treatment['custom'] = fillValue
            fillValue = 'custom'
        df1[col] = df1[col].fillna(treatment[fillValue])

    elif drop:
        df1 = df1[-df1[col].isnull()]
        
    return df1


class CategoryEncoder:
    '''
    labelEncoding : boolean -> set to True If the categorical columns are to be label encoded
    oneHotEncoding : boolean -> set to True If the categorical columns are to be one hot encoded (using pandas.get_dummies method)
    dropFirst : boolean -> set to True if first column is to be dropped (usually to avoid multi-collinearity) post one hot encoding
                           Doesn't matter if oneHotEncoding = False

    df : pandas.DataFrame() -> dataframe object that needs to be encoded
    catCols : list -> list of the categorical columns that need to be encoded
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
        
    def fit(self, df,cols):
        df1 = df.copy()
        for col in cols:
            self.normalizers[col] = self.methods[self.method]()
            self.normalizers[col].fit(df1.loc[:,[col]])
    
    def transform(self, df,cols):
        df1 = df.copy()
        for col in cols:
            df1.loc[:,[col]] = self.normalizers[col].transform(df1.loc[:,[col]])
        return df1


class Classifier:
    '''
    Expects a model json object as like below
        model = {
            'name': 'XGBClassifier',        -> Can be anything
            'estimator': XGBClassifier(),   -> should be a classifier object
            'params':                       -> should be the dictionary of acceptable parameters of the sklearn classifier object
                {
                'objective': 'binary:logistic',
                'max_depth'    : 2,
                'n_estimators' : 100,
                'learning_rate': 0.5
                }
            }
    '''
    def __init__(self, model):
        self.model = model['estimator']
        self.params = model['params']
        self.model.set_params(**self.params)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        preds = self.model.predict(X)
        return preds

    def predict_proba(self, X):
        probs = self.model.predict_proba(X)
        return probs

    def __repr__(self):
        # print('what')
        return self.model