import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df1 = pd.read_csv("OxCGRT_compact_national_v1.csv")
df2 = pd.read_csv("daily_sentiment_volatility.csv")

df1['Date'] = pd.to_datetime(df1['Date'], format='%Y%m%d')
df2['date'] = pd.to_datetime(df2['date'])
policy = df1.groupby('Date').mean(numeric_only=True).reset_index()
merged = pd.merge(policy, df2, left_on='Date', right_on='date', how='inner')
merged['policy_std'] = (merged['StringencyIndex_Average'] - merged['StringencyIndex_Average'].mean()) / merged['StringencyIndex_Average'].std()
merged['sentiment_std'] = (merged['sentiment_mean'] - merged['sentiment_mean'].mean()) / merged['sentiment_mean'].std()
plt.figure(figsize=(10,5))
plt.plot(merged['Date'], merged['policy_std'], label='Policy')
plt.plot(merged['Date'], merged['sentiment_std'], label='Sentiment')
plt.legend()
plt.title("Standardized Policy vs Sentiment")
plt.show()

summary=merged[['tweet_volume','sentiment_mean','StringencyIndex_Average']].describe()
print(summary)

merged['tweet_volume'].hist(bins=100)
plt.title("Histogram of Tweet Volume")
plt.xlabel("Tweet Volume")
plt.ylabel("Frequency")
plt.show()

merged['sentiment_mean'].hist(bins=10)
plt.title("Histogram of sentiment")
plt.show()

merged['StringencyIndex_Average'].hist(bins=30)
plt.title("Histogram of Policy Stringency")
plt.show()

corr1= merged[['StringencyIndex_Average', 'sentiment_mean']].corr()
print(corr1)
corr2=merged[['StringencyIndex_Average', 'tweet_volume']].corr()
print(corr2)


plt.scatter(merged['StringencyIndex_Average'], merged['tweet_volume'])
plt.xlabel("Policy")
plt.ylabel("Tweet Volume")
plt.show()

#Normal ols
X = merged[['StringencyIndex_Average']]
y = merged['tweet_volume']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


merged = merged.sort_values('Date')
# lag
merged['policy_lag7'] = merged['StringencyIndex_Average'].shift(7)
df_ts = merged.dropna(subset=['policy_lag7', 'tweet_volume'])
X = df_ts[['policy_lag7']]
y = df_ts['tweet_volume']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


#Random forest
df = merged.copy()
df = df.sort_values('Date')

df['policy_lag7'] = df['StringencyIndex_Average'].shift(7)
df['cases_lag1'] = df['ConfirmedCases'].shift(1)
df['y_yesterday'] = df['tweet_volume'].shift(1)
df['tweet_diff'] = df['tweet_volume'].diff()
df['diff_lag1'] = df['tweet_diff'].shift(1)


df['y_lag1'] = df['tweet_volume'].shift(1)
df['y_lag3'] = df['tweet_volume'].shift(3)
features = ['policy_lag7', 'cases_lag1', 'y_lag1','y_lag3']
target = 'tweet_diff'
needed_cols = features + [target, 'y_yesterday', 'tweet_volume']

df_clean = df.dropna(subset=needed_cols).copy()

X = df_clean[features]
y = df_clean[target]

split_idx = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test_diff = y.iloc[:split_idx], y.iloc[split_idx:]

rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
rf.fit(X_train, y_train)

pred_diff = rf.predict(X_test)

y_yesterday_test = df_clean['y_yesterday'].iloc[split_idx:].values
y_pred_final = y_yesterday_test + pred_diff
y_test_actual = df_clean['tweet_volume'].iloc[split_idx:].values

r2 = r2_score(y_test_actual, y_pred_final)
print("="*30)
print(f"Random Forest (Diff Mode) R2: {r2:.4f}")
print("="*30)

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual', color='blue', alpha=0.6)
plt.plot(y_pred_final, label='Predicted (Diff-based)', color='red', linestyle='--')
plt.title(f"Final RF Prediction (R2: {r2:.3f})")
plt.legend()
plt.show()

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)


#Sarimax
from statsmodels.tsa.statespace.sarimax import SARIMAX
df1=merged.copy()
df1=df1.sort_values("Date")
df1['policy_lag7']=df1['StringencyIndex_Average'].shift(7)
df1['cases_lag1']=df1['ConfirmedCases'].shift(1)
df1clean=df1[['Date','tweet_volume','policy_lag7','cases_lag1']].dropna()
df1clean=df1clean.set_index('Date')
y1=df1clean['tweet_volume']
x1=df1clean[['policy_lag7','cases_lag1']]
model1=SARIMAX(y1,exog=x1,order=(1,0,0))
res=model1.fit()
print(res.summary())