#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of this specific training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

train_url = 'https://raw.githubusercontent.com/edoezie/KaggleTitanic/main/train.csv'
test_url = 'https://raw.githubusercontent.com/edoezie/KaggleTitanic/main/test.csv'

data_train = pd.read_csv(train_url)
df_train = pd.DataFrame(data_train)
df_train.name = "Training Set"

data_test = pd.read_csv(test_url)
df_test = pd.DataFrame(data_test)
df_test.name = "Test Set"

df_all = concat_df(df_train, df_test)
df_all.name = "All Data"

data_set = [df_train, df_test]

total_ids = len(data_set[0])

print("--\ Train set specs: ", data_set[0].shape)
print("--\ Test  set specs: ", data_set[1].shape, "\n")
print("Columns in training set: \n", data_set[0].columns.tolist(), "\n")
print("")

survivees = (data_set[0]['Survived'] == 1).sum()
print("Survived:", survivees, "out of", total_ids, ".")

# Show columns Survived and Fare for filter "sex = female"
# women = data_set[0].loc[data_set[0].Sex == 'female'][{"Survived", "Fare"}]
women = data_set[0].loc[data_set[0].Sex == 'female']["Survived"]
rate_women = sum(women)/len(women) * 100
print ("Percentage women surviving:", rate_women, "\n")

print("Histogram of sex of the survivors: \n")
data_set[0].loc[data_set[0].Survived == 1]['Sex'].hist(bins=3)
plt.show()

print("Histogram of age of the non-survivors: \n")
data_set[0].loc[data_set[0].Survived == 0]['Age'].plot.hist(bins=20, ylim=(0,55))
plt.show()

bla = data_set[0][['Age', 'Fare', 'Survived']]
bla.plot.scatter(x='Age', y='Fare', c='Survived')

#print ("Total Fare: ", data_set[0]['Fare'].sum())

# Find and print out values for rows containing NaN values
#nan_df = data_set[0][data_set[0].isna().any(axis=1)]
#nan_df.head(5)

# Moar Plots
def plot_histogram(dataframe, column_name, title=None):
    ax = dataframe[column_name].hist(density=True, grid=False, edgecolor='white')
    xlim = ax.get_xlim()
    dataframe[column_name].plot.density()
    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.set_xlabel(column_name)

def survived_frequency(dataframe, x_axis):
    fig, ax = plt.subplots(figsize=(15, 10))
    plot = sns.countplot(x=dataframe[x_axis], hue=dataframe["Survived"], palette=["#A70000", '#00FFF0'])
    ax.legend(['Did Not Survive', 'Survived'], loc="upper right")
    return plot

corr_matrix = df_train.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".3f",
                 cmap="YlGnBu"
                 );


cont_features = ['Age', 'Fare']
surv = df_train['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(cont_features):    
    # Distribution of survival in feature
    sns.histplot(df_train[~surv][feature], label='Not Survived', kde=True, stat="density", color='#e74c3c', ax=axs[0][i])
    sns.histplot(df_train[surv][feature], label='Survived', kde=True, stat="density", color='#2ecc71', ax=axs[0][i])
    
    # Distribution of feature in dataset
    sns.histplot(df_train[feature], label='Training Set', color='#e74c3c', ax=axs[1][i])
    sns.histplot(df_test[feature], label='Test Set', color='#2ecc71', ax=axs[1][i])
    
    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')
    
    for j in range(2):        
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)
    
    axs[0][i].legend(loc='upper right', prop={'size': 20})
    axs[1][i].legend(loc='upper right', prop={'size': 20})
    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)

axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)
axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)
        
plt.show()
print ("Categorical: \n")

# Categorical related graphs
cat_features = ['Embarked', 'Sex', 'SibSp', 'Parch']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))
plt.subplots_adjust(right=1.5, top=1.25)

for i, feature in enumerate(cat_features, 1):    
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=df_train)
    
    plt.xlabel('{}'.format(feature), size=20, labelpad=15)
    plt.ylabel('Passenger Count', size=20, labelpad=15)    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})
    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)

plt.show()


def display_missing(df):    
    for col in df.columns.tolist():          
        print('{:.<13} column missing values: {:3} ({:04.1f}%)'.format(col, df[col].isnull().sum(), df[col].isnull().sum()/len(df)*100))
    print('\n')
    
for df in data_set:
    print('{} in total {} items'.format(df.name, len(df)))
    display_missing(df)

df_train_corr = df_train.corr(numeric_only=True).abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_train_corr.head(10)

df_train_corr[df_train_corr['Feature 1'] == 'Age'][1::]     # Age correlates strongly to Pclass and Pclass to sex.

# Strongest correlations, show index mod 2:
top_corr = df_train_corr[df_train_corr['Feature 1'] != df_train_corr['Feature 2']].head(14).iloc[::2]
print("Highest correlation: \n", top_corr, "\n")


# ---
# 
# Test set = Training set minus "Survived" column
# 
# ---
# 
# 

## CleanAndEnrich
## raw_input = Pandas Dataframe
## Target = Y (single column that needs to be predicted) OR '' if processing the test data (prediction)
## todrop = coloms from dataframe that need to be dropped
def CleanAndEnrich (raw_input, target, data_name):

    print ("--Processing Dataframe with raw columns (name:", data_name, "):\n", raw_input.columns.tolist(), "\n")
    print ("--Total amount of missing values: ", raw_input.isnull().sum().sum(), "\n")
    droplist = ['PassengerId', 'Ticket']
    
    if target != '':
        df_Y = raw_input[target].copy()
        #droplist.append(target)
    else:
        df_Y = None
    
    df_X = raw_input.drop(columns = droplist).copy()
    
    # Create new column to note down if a Cabin was present for this person
    #df_X['Has_cabin'] = df_X['Cabin'].notna().astype(int)
    #df_X.drop("Cabin", axis=1, inplace=True)

    # Map male/female to 0/1
    df_X['Sex'] = df_X['Sex'].map({'male': 0, 'female': 1})

    # Extract titles from Name field and create a new column, drop Name column after
    common_titles = ['Mr', 'Mrs', 'Master', 'Miss']
    df_X['Title'] = df_X['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
    df_X['Title'] = df_X['Title'].apply(lambda title: title if title in common_titles else 'Other')
    df_X = pd.get_dummies(df_X, columns=['Title'], drop_first=False)

    # Map Embarked to booleans of location
    df_X['Embarked'] = df_X['Embarked'].fillna('S')
    df_X = pd.get_dummies(df_X, columns=['Embarked'], drop_first=False)

    # Fare processing - 1 missing val in test set
    med_fare = df_X.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    df_X['Fare'] = df_X['Fare'].fillna(med_fare)
    
    # Add new column showing youth with and without parents
    df_X['GuidedKid'] = np.where((df_X['Age'] < 18) & (df_X['Parch'] > 0), 1, 0)
    df_X['SingleKid'] = np.where((df_X['Age'] < 18) & (df_X['Parch'] == 0), 1, 0)
    df_X['Familysize'] = (df_X['SibSp'] + df_X['Parch'] + 1)
    df_X['Solo'] = np.where((df_X['Parch'] + df_X['SibSp'] == 0), 1, 0)

    # Fill NaN's in Age with median age related to sex & ticket class (highly correlated)
    grouped = df_X.groupby(['Sex', 'Pclass'])['Age'].transform('median')
    df_X.fillna({'Age': grouped}, inplace=True)
    
    #df_X.drop(columns='SibSp', axis=1, inplace=True)
    #df_X.drop(columns='Parch', axis=1, inplace=True)
    df_X.drop(columns='Name', axis=1, inplace=True)
    df_X.drop(columns='Cabin', axis=1, inplace=True)

    print("--Finalized processing, resulting columns:\n", df_X.columns.tolist(), "\n")
    print ("--Amount of missing values: ", df_X.isna().sum().sum(), "\n")

    return [df_Y, df_X]


## NormaliseFrame
## Will normalize all numerical values as specified in "fields" parameter using MinMaxScaler
def NormaliseFrame (frame, fields):
    
    print("--Normalizing the following fields: \n", fields, "\n")
    scaler = MinMaxScaler()
    normalized = frame.copy()

    to_scale_columns = normalized[fields].values
    columns_scaled = scaler.fit_transform(to_scale_columns)
    df_temp = pd.DataFrame(columns_scaled, columns=fields, index = frame.index)
    normalized[fields] = df_temp

    return normalized


fields_to_normalise = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']

df_Y, df_X = CleanAndEnrich(data_set[0],'Survived', 'Training')
df_X = NormaliseFrame(df_X, fields_to_normalise)
df_T = CleanAndEnrich(data_set[1],'', 'Test')
df_T = NormaliseFrame(df_X, fields_to_normalise)

total_input_cols = len(df_X.columns)
print("Total columns: ", total_input_cols, ".\n")

corr_matrix = df_X.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".3f",
                 cmap="YlGnBu"
                 );


# # split train into 90:10 train and val sets
train, val, train_y, val_y = train_test_split(df_X, df_Y, test_size=0.20)

Tx = tf.convert_to_tensor(train, dtype=tf.float64)
Ty = tf.convert_to_tensor(train_y, dtype=tf.float64)
Vx = tf.convert_to_tensor(val, dtype=tf.float64)
Vy = tf.convert_to_tensor(val_y, dtype=tf.float64)

## Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=1)
history = model.fit(Tx, Ty)
print (history)


modelDNN = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(total_input_cols,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0003,
    beta_1=0.9,
    beta_2=0.999
)

modelDNN.compile(optimizer=optimizer,
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy']) 

print(modelDNN.summary()) 

history = modelDNN.fit(Tx, Ty, epochs=150, validation_data=(Vx, Vy), verbose=0)
print ("Training done.")


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# (100, 19) dim pred_subset
pred_subset = df_X[100:200].copy()
correct_Y = df_Y[100:200].copy()

print ("Predicting for:", len(pred_subset), "entries.")
predictions = modelDNN.predict(pred_subset)

pred_subset = pred_subset.to_numpy()
correct_Y = correct_Y.to_numpy()

print ("Head predictions: \n", predictions[0:5])
print ("Head correct_Y: \n", correct_Y[0:5])

print("Predicted value: ", predictions[18][0])
print("Correct value: ",correct_Y[18])

#for item in range(correct_Y.index.start, correct_Y.index.stop):
#    if correct_Y.iloc(item) != np.round(predictions(item)[0]):
#        print ("Mismatch! Predicted value was: ", predictions(item)[0], "\n", pred_subset.iloc(item), "\n")


# Create CSV to submit!

predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


## SAVED FUN STUFF BELOW - IGNORE - :D ##

# yank = []
# yeet = data_set[0]['Name']
# print(yeet[8])
# for x in yeet:
#     yank.append(x.split(',')[1].split('.')[0].strip())

# yank = pd.Series(yank)
# unique_values = yank.unique().tolist()
# unique_counts = yank.value_counts().tolist()
# print ("Titles used: ", unique_values[0:4])
# print ("Amounts: ", unique_counts[0:4])

#print ("Normalized df_X: \n\n", df_X.head(6), "\n")
#print ("Y values: \n", df_Y.head(6))
#print ("Unique SibSP and Parch values: ", df_X['SibSp'].nunique(), "and", df_X['Parch'].nunique())


# Remove Cabin column as nearly 80% of data is missing this field
# For now also removing Name & Ticket as no correlation expected
# for column_name in {'Cabin', 'Name', 'Ticket'}:
#  if column_name in data_set[0].columns:
#    data_set[0] = data_set[0].drop(columns = column_name)

# One Hot encode Embarked (values: CQS) using get_dummies into 3 new columns
# df_X = pd.get_dummies(df_X, columns=['Embarked'], prefix="Embarked", dummy_na=False, dtype=float)

#//////
#Nice way to update:
#traindf.loc[ traindf['Age'] <= 16, 'Age'] 					       = 0
#traindf.loc[(traindf['Age'] > 16) & (traindf['Age'] <= 32), 'Age'] = 1

#traindf['Embarked'] = traindf['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#traindf['Title'] = traindf['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
#common_titles = ['Mr', 'Mrs', 'Master', 'Miss']
#traindf['Title'] = traindf['Title'].apply(lambda title: title if title in common_titles else 'Other')

#//////
#print("Number of uniqe values per column: ")
#df_X.nunique()

#data_set[0].sort_values(by='PassengerId', ascending=False).head(10)
#data_set[0].corr(numeric_only=True)

# sns.heatmap(data_set[0].corr(numeric_only=True), annot = True)
#plt.rcParams['figure.figsize'] = (10,7)
#plt.show()

#data_set[0].groupby('SibSp').mean(numeric_only=True).sort_values(by='Age')
# df.groupby('Continent')[df.columns[5:13]].mean().sort_values(by='Ranking')
# df[df['Continent'].str.contains('Oceania')]]

#data_set[0].boxplot(figsize=(20,10))
# df.select_dtypes(include='number'/'object'/'float')

# common_titles = ['Mr', 'Mrs', 'Master', 'Miss', 'Other']
# df_X['Title'] = df_X['Title'].apply(lambda title: common_titles.index(title) if title in common_titles else common_titles.index('Other'))





# Bla!
