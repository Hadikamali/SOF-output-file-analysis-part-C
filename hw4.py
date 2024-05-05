import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

def read_file(path,encoding="utf-16"):
    df = pd.read_csv(path, encoding=encoding)
    return df

questions = read_file("Data/Questions.txt")
answers = read_file("Data/Answer.csv")
answers = answers.drop(0)
q_a = read_file("Data/Q_A.csv")
users = pd.read_csv('Data/U.csv', encoding="utf-16")


answers = pd.read_csv('Data/Answer.csv', encoding="utf-16")
answers = answers.drop(0) # remove the meaningless row(the second line)
len(answers['ClosedDate'].dropna())
questions = pd.read_csv('Data/Questions.txt', encoding="utf-16")
Q_A = pd.read_csv('Data/Q_A.csv', encoding="utf-16")
users = pd.read_csv('Data/U.csv', encoding="utf-16")
user_badge = pd.read_csv('Data/user_badge.txt', encoding="utf-16")
tags = pd.read_csv('Data/tags.txt', sep='\t')

merged_A_Q = pd.merge(answers, Q_A, left_on='Id', right_on='AId', how='inner')
merged_A_Q = pd.merge(merged_A_Q, questions, left_on='QId', right_on='Id', how='inner',suffixes=('_A', '_Q'))
merged_A_Q_U2 = pd.merge(merged_A_Q, users,  left_on='OwnerUserId_A', right_on='Id', how='left')
badge_count = user_badge.groupby('UserId').size().reset_index(name='badgeCount')
merged_A_Q_U_B = pd.merge(merged_A_Q_U2, badge_count, left_on='Id', right_on='UserId', how='inner')
merged_A_Q_U_B['Closed_Q'] = merged_A_Q_U_B['ClosedDate_Q'].notnull().astype(int)
merged_A_Q_U_B['CreationDate_A'] = pd.to_datetime(merged_A_Q_U_B['CreationDate_A'])
merged_A_Q_U_B['CreationDate_Q'] = pd.to_datetime(merged_A_Q_U_B['CreationDate_Q'])
merged_A_Q_U_B['timeDiffrence'] = (merged_A_Q_U_B['CreationDate_A'] - merged_A_Q_U_B['CreationDate_Q']).dt.total_seconds() / 60 / 60 #based on hours
merged_A_Q_U_B2 = merged_A_Q_U_B[merged_A_Q_U_B['timeDiffrence']>0]


merged_A_Q_U_B2.rename(columns = {'FavoriteCount':'FavoriteCount_Q'}, inplace = True)
merged_A_Q_U_B2.rename(columns = {'badgeCount':'badgeCount_U'}, inplace = True)
merged_A_Q_U_B2.rename(columns = {'Reputation':'Reputation_U'}, inplace = True)
merged_A_Q_U_B2.rename(columns = {'Views':'Views_U'}, inplace = True)
merged_A_Q_U_B2.rename(columns = {'UpVotes':'UpVotes_U'}, inplace = True)
merged_A_Q_U_B2.rename(columns = {'DownVotes':'DownVotes_U'}, inplace = True)
merged_A_Q_U_B2.rename(columns = {'ViewCount':'ViewCount_Q'}, inplace = True)

features = ['Score_A', 'CommentCount_A',
         'Reputation_U', 'Views_U', 'UpVotes_U',
       'DownVotes_U', 'badgeCount_U', 'timeDiffrence','FavoriteCount_Q', 'Score_Q','ViewCount_Q']
target = 'accepted'

X = merged_A_Q_U_B2[features]
y = merged_A_Q_U_B2[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)

val_predictions = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, val_predictions)
print(f'Accuracy on the validation set: {accuracy:.2f}')

accuracy = metrics.f1_score(y_test, val_predictions)
print(f'F1 on the validation set: {accuracy:.2f}')

accuracy = metrics.precision_score(y_test, val_predictions)
print(f'Precision on the validation set: {accuracy:.2f}')

accuracy = metrics.recall_score(y_test, val_predictions)
print(f'Recall on the validation set: {accuracy:.2f}')

print('\nClassification Report:\n', metrics.classification_report(y_test, val_predictions))

xgb.plot_importance(model)

plt.savefig('feature_importance_plot.png', dpi=1200, bbox_inches='tight')
