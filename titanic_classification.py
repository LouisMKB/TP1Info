# %% [markdown]
# ## Predicting Survival on the Titanic
# 
# ### History
# Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time.
# 
# ### Assignment:
# 
# Build a Machine Learning Pipeline, to engineer the features in the data set and predict who is more likely to Survive the catastrophe.
# 
# Follow the Jupyter notebook below, and complete the missing bits of code, to achieve each one of the pipeline steps.

# %%
import re

# to handle datasets
import pandas as pd
import numpy as np

# for visualization
import matplotlib.pyplot as plt

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib

# to visualise al the columns in the dataframe


# %%
# load the data - it is available open source and online

def read_data(path:str) -> pd.Dataframe:
    return pd.read_csv(path)


def replace_nan(data: pd.DataFrame) -> pd.DataFrame:
    data = data.replace('?', np.nan)
    return data


def lower_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.columns.str.lower()


def retain_first_cabin(data: pd.DataFrame) -> pd.DataFrame:

    def get_first_cabin(row):
        try:
            return row.split()[0]
        except:
            return np.nan
    
    data['cabin'] = data['cabin'].apply(get_first_cabin)

    return data
    

def add_titles(data: pd.DataFrame) -> pd.DataFrame:

    def get_title(passenger):
        line = passenger
        if re.search('Mrs', line):
            return 'Mrs'
        elif re.search('Mr', line):
            return 'Mr'
        elif re.search('Miss', line):
            return 'Miss'
        elif re.search('Master', line):
            return 'Master'
        else:
            return 'Other'
        
    data['title'] = data['name'].apply(get_title)
    return data



def type_convertion(data: pd.DataFrame) -> pd.DataFrame:

    data['fare'] = data['fare'].astype('float')
    data['age'] = data['age'].astype('float')

    vars_num = [c for c in data.columns if data[c].dtypes!='O' and c!=target]
    vars_cat = [c for c in data.columns if data[c].dtypes=='O']
    logger.info('Number of numerical variables: {}'.format(len(vars_num)))
    logger.info('Number of categorical variables: {}'.format(len(vars_cat)))  

# %%
target = 'survived'



# %%
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape

# %% [markdown]
# ## Feature Engineering
# 
# ### Extract only the letter (and drop the number) from the variable Cabin

# %%
X_train['cabin'] = X_train['cabin'].str[0] # captures the first letter
X_test['cabin'] = X_test['cabin'].str[0] # captures the first letter

X_train['cabin'].unique()

# %% [markdown]
# ### Fill in Missing data in numerical variables:
# 
# - Add a binary missing indicator
# - Fill NA in original variable with the median

# %%
def fill_by_median(data: pd.DataFrame) -> pd.DataFrame:
    for var in ['age', 'fare']:

        # add missing indicator
        data[var+'_NA'] = np.where(data[var].isnull(), 1, 0)
        data[var+'_NA'] = np.where(data[var].isnull(), 1, 0)
.
        # replace NaN by median
        median_val = data[var].median()

        data[var].fillna(median_val, inplace=True)
        data[var].fillna(median_val, inplace=True)

    return data[['age', 'fare']].isnull().sum()

# %% [markdown]
# ### Replace Missing data in categorical variables with the string **Missing**

# %%
X_train[vars_cat] = X_train[vars_cat].fillna('Missing')
X_test[vars_cat] = X_test[vars_cat].fillna('Missing')

# %%
X_train.isnull().sum()

# %%
X_test.isnull().sum()

# %% [markdown]
# ### Remove rare labels in categorical variables
# 
# - remove labels present in less than 5 % of the passengers

# %%
def find_frequent_labels(df, var, rare_perc):
    
    # function finds the labels that are shared by more than
    # a certain % of the passengers in the dataset
    
    df = df.copy()
    
    tmp = df.groupby(var)[var].count() / len(df)
    
    return tmp[tmp > rare_perc].index


for var in vars_cat:
    
    # find the frequent categories
    frequent_ls = find_frequent_labels(X_train, var, 0.05)
    
    # replace rare categories by the string "Rare"
    X_train[var] = np.where(X_train[var].isin(
        frequent_ls), X_train[var], 'Rare')
    
    X_test[var] = np.where(X_test[var].isin(
        frequent_ls), X_test[var], 'Rare')


def add_rare_status(data: pd.DataFrame) -> 

# %%
X_train[vars_cat].nunique()

# %%
X_test[vars_cat].nunique()

# %% [markdown]
# ### Perform one hot encoding of categorical variables into k-1 binary variables
# 
# - k-1, means that if the variable contains 9 different categories, we create 8 different binary variables
# - Remember to drop the original categorical variable (the one with the strings) after the encoding

# %%
for var in vars_cat:
    
    # to create the binary variables, we use get_dummies from pandas
    
    X_train = pd.concat([X_train,
                         pd.get_dummies(X_train[var], prefix=var, drop_first=True)
                         ], axis=1)
    
    X_test = pd.concat([X_test,
                        pd.get_dummies(X_test[var], prefix=var, drop_first=True)
                        ], axis=1)
    

X_train.drop(labels=vars_cat, axis=1, inplace=True)
X_test.drop(labels=vars_cat, axis=1, inplace=True)

X_train.shape, X_test.shape

# %%
# Note that we have one less column in the test set
# this is because we had 1 less category in embarked.

# we need to add that category manually to the test set

X_train.head()

# %%
X_test.head()

# %%
# we add 0 as values for all the observations, as Rare
# was not present in the test set

X_test['embarked_Rare'] = 0

# %%
# Note that now embarked_Rare will be at the end of the test set
# so in order to pass the variables in the same order, we will
# create a variables variable:

variables = [c  for c in X_train.columns]

variables

# %% [markdown]
# ### Scale the variables
# 
# - Use the standard scaler from Scikit-learn

# %%
# create scaler
scaler = StandardScaler()

#  fit  the scaler to the train set
scaler.fit(X_train[variables]) 

# transform the train and test set
X_train = scaler.transform(X_train[variables])

X_test = scaler.transform(X_test[variables])

# %% [markdown]
# ## Train the Logistic Regression model
# 
# - Set the regularization parameter to 0.0005
# - Set the seed to 0

# %%
# set up the model
# remember to set the random_state / seed

model = LogisticRegression(C=0.0005, random_state=0)

# train the model
model.fit(X_train, y_train)

# %% [markdown]
# ## Make predictions and evaluate model performance
# 
# Determine:
# - roc-auc
# - accuracy
# 
# **Important, remember that to determine the accuracy, you need the outcome 0, 1, referring to survived or not. But to determine the roc-auc you need the probability of survival.**

# %%
# make predictions for test set
class_ = model.predict(X_train)
pred = model.predict_proba(X_train)[:,1]

# determine mse and rmse
print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
print()

# make predictions for test set
class_ = model.predict(X_test)
pred = model.predict_proba(X_test)[:,1]

# determine mse and rmse
print('test roc-auc: {}'.format(roc_auc_score(y_test, pred)))
print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
print()


