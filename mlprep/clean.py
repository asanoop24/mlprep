import pandas
from scipy import stats


def missingSummary(df:pandas.DataFrame,returnType:str='df'):
    '''
    # Returns a dictionary or a pandas dataframe object with the following details:
    ColumnName       : column name
    MissingCount     : count of missing values
    MissingPercentage: percentage of missing values

    # Input Parameters
    df        : pandas data frame object: Required
    returnType: string object: Optional
                returns the output as pandas dataframe if value is 'df', otherwise returns the output as dictionary
    '''

    df1 = df.copy()
    missingSummary = {}
    # missingSummary = {col:len(df1[df1[col].isnull()][col]) for col in df1.columns.tolist()}
    missingSummary = {col:(df1.loc[:,col].isnull().sum(),round(1.0*df1.loc[:,col].isnull().sum()/len(df1),3)) for col in df1.columns.tolist()}
    missingSummary = dict(sorted(missingSummary.items(), key=lambda x: x[1], reverse=True))
    missingSummary_df = pandas.DataFrame.from_dict(data=missingSummary).T.reset_index().rename(columns={'index':'ColumnName',0:'MissingCount',1:'MissingPercentage'})
    return missingSummary_df if returnType == 'df' else missingSummary


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


def outlierSummary(df,cols,method='z-score'):
    df1 = df.copy()
    outlierSummary = {}

    ## Method 1: Using the Z-Score
    if method == 'z-score':
        for col in cols:
            if df1[col].dtype == 'object': continue
            lowerValue = df1[stats.zscore(df1.loc[:,col])<-3][col].max() if len(df1[stats.zscore(df1.loc[:,col])<-3]) > 0 else df1.loc[:,col].min()
            upperValue = df1[stats.zscore(df1.loc[:,col])> 3][col].min() if len(df1[stats.zscore(df1.loc[:,col])> 3]) > 0 else df1.loc[:,col].max()
            outlierSummary[col] = (lowerValue,upperValue)

    ## Method 2: Using the IQR
    if method == 'iqr':
        for col in cols:
            if df1[col].dtype == 'object': continue
            Q1 = df1.loc[:,col].quantile(0.25)
            Q3 = df1.loc[:,col].quantile(0.75)
            IQR = Q3 - Q1
            lowerValue = df1[df1.loc[:,col] < Q1 - 1.5*IQR][col].max() if len(df1[df1.loc[:,col] < Q1 - 1.5*IQR]) > 0 else df1.loc[:,col].min()
            upperValue = df1[df1.loc[:,col] > Q3 + 1.5*IQR][col].min() if len(df1[df1.loc[:,col] > Q3 + 1.5*IQR]) > 0 else df1.loc[:,col].max()
            outlierSummary[col] = (lowerValue,upperValue)
        
    return outlierSummary
