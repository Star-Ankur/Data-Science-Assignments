import pandas as pd

df = pd.read_csv('SteelPlateFaults-train.csv')

df2 = df[20:32]
cols = list(df.columns)
df3 = df2[cols[:4]]
df4=df3.copy()

print(df3)
df4.plot(kind='bar')
# We can see how skewed the data is. Some fields have very large values and some have small values.

'''Maximum absolute scaling'''
# The maximum absolute scaling rescales each feature between -1 and 1 by dividing 
# every observation by its maximum absolute value.
def max_abs_scaling(df):
    for column in df.columns:
        df[column] = df[column]  / df[column].abs().max()
    return df

max_abs_scaling(df4).plot(kind='bar')

'''Min Max feature Scaling'''
# The min-max approach (often called normalization) rescales the feature to a range of [0,1] 
# by subtracting the minimum value of the feature then dividing by the range.
def min_max_scaling(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())   
    return df

min_max_scaling(df4).plot(kind='bar')

'''Z-score method'''
# The z-score method (often called standardization) transforms the info into distribution 
# with a mean of 0 and a typical deviation of 1.  
def z_score(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std() 
    return df

z_score(df4).plot(kind='bar')
        




