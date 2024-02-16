#PON SWARNALAYA RAVICHANDRAN
# UNDERWATER SURFACE TEMPERATURE PREDICTION (TIME SERIES MODELLING)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import statsmodels.tsa.holtwinters as ets
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy import linalg as la
from Toolbox import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("OG.csv", encoding = 'unicode_escape')
print(data.info())
#6
#checking for null values
print(data.isnull().sum())

#ffilling the dta
# Example using forward-fill for missing values
data['Temp (°C)'].fillna(method='ffill', inplace=True)
print(data.isnull().sum())

#renaming a column which has '#'
data.rename(columns={'#': 'INDEX'}, inplace=True)
print(data.info(), data.head())


data['Date'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

print('DATA AFTER RENAMING THE DATE')
print(data.info())
#PLOT OF TIME VS TEMP(C)
plt.plot(data['Date'], data['Temp (°C)'], label='Temperature')
plt.xlabel('Time')
plt.ylabel('Temp (°C)')
plt.title('Time vs Temperature')
plt.tight_layout()
plt.legend()
plt.show()

#onehot
df = pd.get_dummies(data, columns=['Site'], prefix='Site')
print(df.info())

#downsampling
df = df.set_index('Date').resample('H').mean()
df = df.reset_index()
print(df.info())

#plotting the downsampled data
plt.plot(df['Date'], df['Temp (°C)'], label='Temperature')
plt.xlabel('Time')
plt.ylabel('Temp (°C)')
plt.title('Time vs Temperature')
plt.tight_layout()
plt.legend()
plt.show()

cor = df.corr()
plt.figure(figsize=(12, 10))
import seaborn as sns
sns.heatmap(cor,vmin=-1,vmax=1,center=0,cmap='PiYG')
plt.title("Heatmap of dataset", fontsize =15)
plt.show()
print(cor)

print("The dataset is df now")
train,test=train_test_split(df,test_size=0.2,shuffle=False)


#7
#stationarity check
cal_rolling_mean_var(df['Temp (°C)'], df['Date']) #rollingmean

ADF_cal(df["Temp (°C)"]) #adf of raw data
kpss_test(df["Temp (°C)"])

acf = ACF(df['Temp (°C)'],50)
x=np.arange(0,51)
plt.stem(x,acf,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.title("ACF plot")
plt.ylabel("ACF")
plt.xlabel("Lags")
plt.show()

ACF_PACF_Plot(df['Temp (°C)'], 3000)


print("The data is not stationary at this point")


#first order differencing
diff1=difference(df['Temp (°C)'],interval=1260)
diff_df=pd.DataFrame(diff1, index=df.index[1260:])
cal_rolling_mean_var(diff_df, np.arange(len(diff1)))
ADF_cal(diff_df)
kpss_test(diff_df)

acf2=ACF(diff1,50)
x= np.arange(0,51)
plt.stem(x,acf2,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf2,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.title("ACF plot")
plt.ylabel("ACF")
plt.xlabel("Lags")
plt.show()

ACF_PACF_Plot(diff1, 3000)

print("The data is not stationary yet")

#diff2
diff2 = difference(diff1, 1)
diff_df1 = pd.DataFrame(diff1, df.index[1260:])  # Adjust the index range
diff_df2 = pd.DataFrame(diff2, df.index[1261:])

diff_df21 = pd.DataFrame(diff2, index=df.index[1261:], columns=["Temp (°C)"])  # Adjust column name if needed

# Plotting code
plt.plot(diff_df21.index, diff_df21["Temp (°C)"], label='Differenced')
plt.xlabel('Time')
plt.ylabel('Temp (°C)')
plt.title('Time vs Temperature')
plt.tight_layout()
plt.legend()
plt.show()

cal_rolling_mean_var(diff_df2, np.arange(len(diff2)))

ADF_cal(diff2)
kpss_test(diff2)

acf3=ACF(diff2,50)
x= np.arange(0,51)
plt.stem(x,acf3,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf3,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.title("ACF plot")
plt.ylabel("ACF")
plt.xlabel("Lags")
plt.show()

ACF_PACF_Plot(diff2,3000)


#8
#time series decomposition
from statsmodels.tsa.seasonal import STL
index = pd.date_range('2013-02-20 11:40:02', periods=len(df), freq='1H')  # Assuming data is recorded every 1 hour

Temperature = pd.Series(np.array(df['Temp (°C)']), index=index)

STL= STL(Temperature)
res=STL.fit()
fig=res.plot()
plt.xlabel("Iterations")
plt.tight_layout()
plt.show()

T=res.trend
S=res.seasonal
R=res.resid


plt.plot(T,label="Trend")
plt.plot(R,label="Residual")
plt.plot(S,label="Seasonal")
plt.xlabel("Iterations")
plt.ylabel("STL")
plt.legend()
plt.title("Trend, Seasonality and residuals of data")
plt.show()

#Strength of trend
var=1-(np.var(R)/np.var(T+R))
Ft=np.max([0,var])
print("Strength of trend:",Ft)

#Strength of seasonality
var1=1-(np.var(R)/np.var(S+R))
Fs=np.max([0,var1])
print("Strength of seasonality:",Fs)

#seasonally adjusted data
seasonally_adj= Temperature-S
plt.figure(figsize=(12, 10))
plt.plot(index, df['Temp (°C)'],label="Original")
plt.plot(index, seasonally_adj,label="adjusted")
plt.xlabel("Time")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Temperature")
plt.title("Seasonally adjusted vs. Original")
plt.legend()
plt.show()
#
#detrended data
detrended=Temperature-T
plt.figure(figsize=(12, 10))
plt.plot(index, df['Temp (°C)'],label="Original")
plt.plot(index,detrended,label="Detrended")
plt.xlabel("Time")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Temperature")
plt.title("Detrended vs. Original Data")
plt.legend()
plt.show()

# 9
# Holt-Winters method
print(df.columns)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
X = df[['INDEX', 'Latitude', 'Longitude', 'Depth', 'Site_Ilha Deserta', 'Site_Ilha da Galé', 'Site_Ilha do Coral',
        'Site_Ilha dos Lobos', 'Site_Moleques do Sul', 'Site_Parcel da Pombinha', 'Site_Parcel do Xavier (Alalunga)',
        'Site_Tamboretes', 'Site_lha do Xavier']]
y = df['Temp (°C)']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Fitting Holt-Winters model
holtw1 = ets.ExponentialSmoothing(y_train, damped_trend= True,trend='add', seasonal='add',seasonal_periods=1260)
holtw = holtw1.fit()

# HW prediction on train set
holtw_pred_train = holtw.predict(start=y_train.index[0], end=y_train.index[-1])
holtw_df_train = pd.DataFrame(holtw_pred_train, columns=['Temp (°C)'], index=y_train.index)

# HW prediction on test set
holtw_pred_test = holtw.predict(start=y_test.index[0], end=y_test.index[-1])
holtw_df_test = pd.DataFrame(holtw_pred_test, columns=['Temp (°C)'], index=y_test.index)

# Plot of HW model
plt.figure(figsize=(12, 8))
plt.plot(y_train.index, y_train, label='Train')
plt.plot(y_test.index, y_test, label='Test')
plt.plot(holtw_df_test.index, holtw_df_test['Temp (°C)'], label='Holts winter prediction (Test)')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Holts winter model")
plt.show()

#Model performance on train and test data

#MSE
HW_train_mse=mean_squared_error(y_train,holtw_df_train[['Temp (°C)']])
print('MSE of Holts Winter method on train data:',HW_train_mse)
HW_test_mse=mean_squared_error(y_test,holtw_df_test['Temp (°C)'])
print('MSE of Holts Winter method on test data:',HW_test_mse)

#residual error
HW_reserror=y_train-holtw_df_train['Temp (°C)']

#Forecast error
HW_foerror=y_test-holtw_df_test['Temp (°C)']

#ACF
acf_hw_res=ACF(HW_reserror.values,60) #train
x=np.arange(0,61)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_hw_res,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_hw_res,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals (HW)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()


acf_hw_fore=ACF(HW_foerror.values,60) #test
x=np.arange(0,61)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_hw_fore,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_hw_fore,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast error (HW)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Q-value
hotl_q_t=sm.stats.acorr_ljungbox(HW_reserror, lags=5,return_df=True)
print('Q-value (residual):',hotl_q_t)
lbvalue=sm.stats.acorr_ljungbox(HW_foerror,lags=5,return_df=True)
print('Q-value (Forecast):\n',lbvalue)

#Error mean and variance
print('Holts winter: Mean of residual error is',np.mean(HW_reserror),'and Forecast error is',np.mean(HW_foerror))
print('Holts winter: Variance of residual error is',np.var(HW_reserror),'and Forecast error is',np.var(HW_foerror))

#RMSE
HW_train_rmse=mean_squared_error(y_train,holtw_df_train['Temp (°C)'],squared=False)
print('RMSE of Holts Winter method on train data:',HW_train_rmse)
HW_test_rmse=mean_squared_error(y_test,holtw_df_test['Temp (°C)'],squared=False)
print('RMSE of Holts Winter method on test data:',HW_test_rmse)

#10: Feature selection and collinearity
# Assuming X contains your predictor variables and y is the target variable
X_mat = x_train.values
Y = y_train.values
X_svd =sm.add_constant(X_mat)
H= np.matmul(X_svd.T,X_svd)
s,d,v= np.linalg.svd(H)
print('Singular Values: ',d)

#Condition number
print("The condition number is ",la.cond(X_svd))

#Feature selection
x_train_ols=sm.add_constant(x_train)
model=sm.OLS(y_train,x_train_ols).fit()
print(model.summary())

#collinearity removal process
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# Ridge Regression with Cross-Validated Grid Search
ridge = Ridge()
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters
best_alpha = grid_search.best_params_['alpha']

# Ridge Regression with optimal alpha
ridge_optimal = Ridge(alpha=best_alpha)
ridge_optimal.fit(X_train_scaled, y_train)

# Predictions on train and test sets
y_train_pred = ridge_optimal.predict(X_train_scaled)
y_test_pred = ridge_optimal.predict(X_test_scaled)

# Evaluate MSE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# VIF Calculation
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print results
print(f'MSE of Ridge regression on train data: {mse_train}')
print(f'MSE of Ridge regression on test data: {mse_test}')
print('Optimal alpha:', best_alpha)
print('VIF values:\n', vif)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#pca
pca=PCA(n_components='mle',svd_solver='full')
pca.fit(X)
x_pca=pca.transform(X)
print('Explained variance ratio: Original Feature space vs.Reduced Feature space\n',pca.explained_variance_ratio_)
ev = pca.explained_variance_ratio_
cv = np.cumsum(ev) * 100
num_com_95 = np.argmax(cv >= 95) + 1
num_com_to_remove = pca.n_components_ - num_com_95
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()
num_components = np.argmax(cumulative_explained_variance >= 0.95) + 1

# Calculate VIF for the new features after PCA
vif_data = pd.DataFrame()
vif_data["Variable"] = range(1, x_pca.shape[1] + 1)  # Use range as index for VIF calculation
vif_data["VIF"] = [variance_inflation_factor(x_pca, i) for i in range(x_pca.shape[1])]
print("VIF values after PCA:")
print(vif_data)

#11
#Base models
#Average method
train_pred_avg=avg_one(y_train)
test_pred_avg=avg_hstep(y_train,y_test)

#Plot of average method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_pred_avg,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('Average method predictions')
plt.legend()
plt.show()

#Plot of test vs predicted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_pred_avg,label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('Average method Forecast')
plt.legend()
plt.show()

#residual and forecast error
avg_res=y_train-train_pred_avg
avg_fore=y_test-test_pred_avg

#ACF of residual
acf_avg_train=ACF(avg_res.values[1:],40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_avg_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_avg_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#forecastedavg acf
acf_avg_test=ACF(avg_fore.values,40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_avg_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_avg_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (Average)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE

avg_train_mse=mean_squared_error(y_train[1:],train_pred_avg[1:])
print('MSE of Average on train data:',avg_train_mse)
avg_test_mse=mean_squared_error(y_test,test_pred_avg)
print('MSE of Average on test data:',avg_test_mse)

#Q-value
q_avg_train=acorr_ljungbox(avg_res.values[1:], lags=5, boxpierce=True,return_df=True)
print('Q-value (residual):',q_avg_train)
q_avgtest=sm.stats.acorr_ljungbox(avg_fore.values[1:],lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_avgtest)

#Error mean and variance
print('Average: Mean of residual error is',np.mean(avg_res),'and Forecast error is',np.mean(avg_fore))
print('Average: Variance of residual error is',np.var(avg_res),'and Forecast error is',np.var(avg_fore))

#RMSE
avg_train_rmse= mean_squared_error(y_train[1:],train_pred_avg[1:],squared=False)
print('RMSE of Average method on train data:',avg_train_rmse)
avg_test_rmse= mean_squared_error(y_test,test_pred_avg,squared=False)
print('RMSE of Average method on test data:',avg_test_rmse)

#Naive method
train_naive=[]
for i in range(len(y_train[1:])):
    train_naive.append(y_train.values[i-1])

test_naive=[y_train.values[-1] for i in y_test]
naive_fore= pd.DataFrame(test_naive).set_index(y_test.index)

#Plot of naive method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_naive,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('Naive method predictions')
plt.legend()
plt.show()

#Plot of test vs predicted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_naive,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('Naive method predictions')
plt.legend()
plt.show()

#residual and forecast error
naive_res= y_train[1:]-train_naive
naive_fore1 = y_test-test_naive

#ACF
acf_naive_train=ACF(naive_res.values,40)

x=np.arange(0,41)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_naive_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_naive_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals(Naive)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_naive_test=ACF(naive_fore1.values,40)

x=np.arange(0,41)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_naive_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_naive_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (Naive)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
naive_train_mse=mean_squared_error(y_train[1:],train_naive)
print('MSE of Naive on train data:',naive_train_mse)
naive_test_mse=mean_squared_error(y_test,test_naive)
print('MSE of Naive on test data:',naive_test_mse)

#Q-value
q_naive_train=acorr_ljungbox(naive_res, lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_naive_train)
q_naivetest=acorr_ljungbox(naive_fore1,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_naivetest)

#Error mean and variance
print('Naive: Mean of residual error is',np.mean(naive_res),'and Forecast error is',np.mean(naive_fore))
print('Naive: Variance of residual error is',np.var(naive_res),'and Forecast error is',np.var(naive_fore))

#RMSE
naive_train_rmse=mean_squared_error(y_train[1:],train_naive,squared=False)
print('RMSE of Naive method on train data:',naive_train_rmse)
naive_test_rmse=mean_squared_error(y_test,test_naive,squared=False)
print('RMSE of Naive method on test data:',naive_test_rmse)

#Drift method
train_drift = []
value = 0
for i in range(len(y_train)):
    if i > 1:
        slope_val = (y_train[i - 1]-y_train[0]) / (i-1)
        y_predict = (slope_val * i) + y_train[0]
        train_drift.append(y_predict)
    else:
        continue

test_drift= []
for h in range(len(y_test)):
    slope_val = (y_train.values[-1] - y_train.values[0] ) /( len(y_train) - 1 )
    y_predict= y_train.values[-1] + ((h +1) * slope_val)
    test_drift.append(y_predict)

#Plot of drift method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_drift,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('Drift method predictions')
plt.legend()
plt.show()

#Plot of test vs predicted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_drift,label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('Drift method forecast')
plt.legend()
plt.show()

#residual and forecast error
drift_res=y_train[2:]-train_drift
drift_fore=y_test-test_drift

#RESIDUALS ACF
acf_drift_train=ACF(drift_res.values,40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_drift_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_drift_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Forecast acf
acf_drift_test=ACF(drift_fore.values,40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_drift_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_drift_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (Drift)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
drift_train_mse=mean_squared_error(y_train[2:],train_drift)
print('MSE of Drift on train data:',drift_train_mse)
drift_test_mse=mean_squared_error(y_test,test_drift)
print('MSE of Drift on test data:',drift_test_mse)

#Q-value
q_drift_train=acorr_ljungbox(drift_res, lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_drift_train)
q_drifttest=acorr_ljungbox(drift_fore,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_drifttest)

#Error mean and variance
print('Drift: Mean of residual error is',np.mean(drift_res),'and Forecast error is',np.mean(drift_fore))
print('Drift: Variance of residual error is',np.var(drift_res),'and Forecast error is',np.var(drift_fore))

#RMSE
drift_train_rmse=mean_squared_error(y_train[2:],train_drift,squared=False)
print('RMSE of Drift method on train data:',drift_train_rmse)
drift_test_rmse=mean_squared_error(y_test,test_drift,squared=False)
print('RMSE of Drift method on test data:',drift_test_rmse)


#SES
ses= ets.ExponentialSmoothing(y_train,trend=None,damped_trend=False,seasonal=None).fit(smoothing_level=0.5)
train_ses= ses.forecast(steps=len(y_train))
train_ses=pd.DataFrame(train_ses).set_index(y_train.index)

test_ses= ses.forecast(steps=len(y_test))
test_ses=pd.DataFrame(test_ses).set_index(y_test.index)

#Plot of SES method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_ses[0],label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('SES method predictions')
plt.legend()
plt.show()

#Plot of test vs predicted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,test_ses[0],label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('SES method forecast')
plt.legend()
plt.show()

#residual and forecast error
ses_res= y_train[2:]-train_ses[0]
ses_fore= y_test-test_ses[0]

#ACF
acf_ses_train=ACF(ses_res.values[2:],40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_ses_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_ses_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

acf_ses_test=ACF(ses_fore.values,40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_ses_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_ses_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (SES)')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
ses_train_mse=mean_squared_error(y_train,train_ses)
print('MSE of SES on train data:',ses_train_mse)
ses_test_mse=mean_squared_error(y_test,test_ses)
print('MSE of SES on test data:',ses_test_mse)

#Q-value
q_ses_train=acorr_ljungbox(ses_res[2:], lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_ses_train)
q_sestest=acorr_ljungbox(ses_fore,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_sestest)

#Error mean and variance
print('SES: Mean of residual error is',np.mean(ses_res),'and Forecast error is',np.mean(ses_fore))
print('SES: Variance of residual error is',np.var(ses_res),'and Forecast error is',np.var(ses_fore))

#RMSE
ses_train_rmse=mean_squared_error(y_train,train_ses,squared=False)
print('RMSE of SES method on train data:',ses_train_rmse)
ses_test_rmse=mean_squared_error(y_test,test_ses,squared=False)
print('RMSE of SES method on test data:',ses_test_rmse)

# 12. Multiple Linear Regression
#Prediction on train data
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

X_train_sm = sm.add_constant(x_train)
model = sm.OLS(y_train, X_train_sm).fit()

# Make predictions on the test set
X_test_sm = sm.add_constant(x_test)
y_pred = model.predict(X_test_sm)

pred_train = model.predict(X_train_sm)
#12.b
print("F-test:")
print("F value:",model.fvalue)
print("P-value:",model.f_pvalue)


print(model.summary())

residuals = y_test - y_pred
fore =y_train-pred_train

plt.plot(y_train.index, y_train, label='Train')
plt.plot(y_test.index, y_test, label='Test')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Dependent Variable")
plt.title("Train vs. Test vs. Predicted Values")
plt.show()
#residual and forecast error


#ACF of residual
acf_avg_train=ACF(residuals,40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_avg_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_avg_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()


#Model performance on train and test data

#MSE
ML_train_mse=mean_squared_error(y_train,pred_train)
print('MSE of MLR on train data:',ML_train_mse)
ML_test_mse=mean_squared_error(y_test,y_pred)
print('MSE of MLR on test data:',ML_test_mse)

#Q-value
q_ml_train=sm.stats.acorr_ljungbox(residuals, lags=5, return_df=True)
print('Q-value (residual):',q_ml_train)
q_mltest=sm.stats.acorr_ljungbox(residuals,lags=5,return_df=True)
print('Q-value (Forecast):\n',q_mltest)

#Error mean and variance
print('MLR: Mean of residual error is',np.mean(residuals),'and Forecast error is',np.mean(fore))
print('MLR: Variance of residual error is',np.var(residuals),'and Forecast error is',np.var(fore))

#RMSE
ml_train_rmse=mean_squared_error(y_train,pred_train,squared=False)
print('RMSE of MLR method on train data:',ml_train_rmse)
ml_test_rmse=mean_squared_error(y_test,y_pred,squared=False)
print('RMSE of MLR method on test data:',ml_test_rmse)

#13: ARMA models

#Order determination
diff_train,diff_test=train_test_split(diff_df21,test_size=0.2,shuffle=False)


#ACF
ACF_PACF_Plot(diff_df21,40)
acf_gpac=ACF(diff_train['Temp (°C)'].values,40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(diff_df21))
plt.stem(x,acf_gpac,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_gpac,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot for GPAC')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#GPAC
GPAC(acf_gpac,10,10)

#order ARMA(1,0)

#14: LMA algorithm
na=1
nb=0
order = (na, 0, nb)  # ARIMA order: (p, d, q)
arma_1_0 = sm.tsa.ARIMA(y_train, order=order, trend=None).fit()

#Estimated parameters
for i in range(na):
    print(f"The AR coefficient a{i} is:",arma_1_0.params[i])
for i in range(nb):
    print(f"The MA coefficient a{i} is:",arma_1_0.params[i + na])
print(arma_1_0.summary())

#initialise values used in function
mu=0.01
delta=10**-6
epsilon=0.001
mu_max=10**10
max_iter=100
SSE,cov,params,var=step3(max_iter,mu,delta,epsilon,mu_max,na,nb,y)
print('The estimated parameter of AR is ',params)

#Prediction on train set
total_length = 18487
train_length = int(0.8 * total_length)
test_length = total_length - train_length

# For training set
arma_1_0_train = arma_1_0.predict(start=0, end=train_length - 1)
arma_1_0_res = y_train - arma_1_0_train

# For test set
arma_1_0_test = arma_1_0.predict(start=train_length, end=total_length - 1)
arma_1_0_fore = y_test - arma_1_0_test

#residual ACF
acf_arma_1_0_train=ACF(arma_1_0_res,40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_arma_1_0_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_arma_1_0_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#forecasted acf
acf_arma_1_0_test=ACF(arma_1_0_fore.values,40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_arma_1_0_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_arma_1_0_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (ARMA(1,0))')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data

#MSE
arma10_train_mse=mean_squared_error(y_train,arma_1_0_train)
print('MSE of ARMA(1,0) on train data:',arma10_train_mse)
arma10_test_mse=mean_squared_error(y_test,arma_1_0_test)
print('MSE of ARMA(1,0) on test data:',arma10_test_mse)

#Q-value
q_arma10_train=acorr_ljungbox(arma_1_0_res, lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_arma10_train)
q_ar10test=acorr_ljungbox(arma_1_0_fore,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_ar10test)

#Error mean and variance
print('ARMA(1,0): Mean of residual error is',np.mean(arma_1_0_res),'and Forecast error is',np.mean(arma_1_0_fore))
print('ARMA(1,0): Variance of residual error is',np.var(arma_1_0_res),'and Forecast error is',np.var(arma_1_0_fore))

#RMSE
ar10_train_rmse=mean_squared_error(y_train,arma_1_0_train,squared=False)
print('RMSE of ARMA(1,0) method on train data:',ar10_train_rmse)
ar10_test_rmse=mean_squared_error(y_test,arma_1_0_test,squared=False)
print('RMSE of ARMA(1,0) method on test data:',ar10_test_rmse)

#16
#Covariance matrix
print('Covariance matrix\n',arma_1_0.cov_params())

#confidence interval
print('Confidence interval:\n',arma_1_0.conf_int())

#standard error
print('Standard error:',arma_1_0.bse)

#Display the estimated variance of error
print('The estimated variance of error is',var)

#chitest
lags = 40
Q1 = q_arma10_train
error1 = arma_1_0_test
chi_test(na,nb,lags,Q1,error1)

#POLE CANCELLATION
zero_poles(params,na)

#Plot of ARMA(1,0) method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,arma_1_0_test,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('ARMA(1,0) method predictions')
plt.legend()
plt.show()

#Plot of test vs forecasted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,arma_1_0_test,label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('ARMA(1,0) method forecast')
plt.legend()
plt.show()

#Plot of train vs predicted
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_train.index,arma_1_0_train,label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('ARMA(1,0) method predictions')
plt.legend()
plt.show()

#sarima
#SARIMA(0,0,1)(0,1,1260)
#order = (1, 0, 0)
#seasonal_order = (0, 1, 1260, 1)  # SARIMA(1,0,0)(0,1,1260)


import statsmodels.api as sm
import math
# Assuming your time series data is stored in the variable 'y_train'

sarima = sm.tsa.statespace.SARIMAX(y_train, order=(1, 0, 0), seasonal_order=(0, 1, 0, 21),
                                   enforce_stationarity=False, enforce_invertibility=False)

results = sarima.fit(disp=0)
print(results.summary())

#predictions on train data
sarima_train = results.get_prediction(start=0, end=len(y_train), dynamic=False)
Sarima_pred = sarima_train.predicted_mean
Sarima_res = y_train-Sarima_pred.values[1:]

#forecast
sarima_test=results.predict(start=0, end=(len(y_test)))
sarima_fore=y_test-sarima_test.values[1:]

#ACF

#ACF of rresiduals
acf_sarima_train=ACF(Sarima_res.values,40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_train))
plt.stem(x,acf_sarima_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_sarima_train,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Residuals')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#acf of forecast
acf_sarima_test=ACF(sarima_fore.values,40)
x=np.arange(0,41)
m=1.96/np.sqrt(len(y_test))
plt.stem(x,acf_sarima_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.stem(-1*x,acf_sarima_test,linefmt='r-', markerfmt='bo', basefmt='b-')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF Plot of Forecast (SARMA(1,0,0)X(0,1,0,12))')
plt.axhspan(-m,m,alpha = .1, color = 'yellow')
plt.tight_layout()
plt.show()

#Model performance on train and test data
#MSE
sarima_train_mse=mean_squared_error(y_train,Sarima_pred[1:])
print('MSE of SARIMA on train data:',sarima_train_mse)
sarima_test_mse=mean_squared_error(y_test,sarima_test[1:])
print('MSE of SARIMA on test data:',sarima_test_mse)

#Q-value
q_sarima_train=acorr_ljungbox(Sarima_res, lags=5, boxpierce=True, return_df=True)
print('Q-value (residual):',q_sarima_train)
q_sarimatest=acorr_ljungbox(sarima_fore,lags=5,boxpierce=True,return_df=True)
print('Q-value (Forecast):\n',q_sarimatest)


#Error mean and variance
print('SARIMA: Mean of residual error is',np.mean(Sarima_res),'and Forecast error is',np.mean(sarima_fore))
print('SARIMA: Variance of residual error is',np.var(Sarima_res),'and Forecast error is',np.var(sarima_fore))

#Covariance matrix
print('Covariance matrix\n',results.cov_params())

#standard error
print('Standard error:',results.bse)

#RMSE
sarima_train_rmse=mean_squared_error(y_train,Sarima_pred[1:],squared=False)
print('RMSE of SARIMA method on train data:',sarima_train_rmse)
sarima_test_rmse=mean_squared_error(y_test,sarima_test[1:],squared=False)
print('RMSE of SARIMA method on test data:',sarima_test_rmse)

#Plot of SARIMA method
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,sarima_test[1:],label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('SARIMA method predictions')
plt.legend()
plt.show()

#Plot of test vs forecasted
plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index,sarima_test[1:],label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('SARIMA method forecast')
plt.legend()
plt.show()

#Plot of train vs predicted
plt.plot(y_train.index,y_train,label='Train')
plt.plot(y_train.index,Sarima_pred[1:],label='Predicted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title('SARIMA method predictions')
plt.legend()
plt.show()

#According to mse metrics, Naive was the best model, but if we consider all the other metrics,

#18
# sarima was more compeitive and most fitted model.
#saarima model is my final model

#15 forecast function
def holt_winters_forecast(train_data, seasonal_type='add', trend_type='add', seasonal_periods=21):
    model = sm.tsa.ExponentialSmoothing(train_data, seasonal=seasonal_type, trend=trend_type,
                                        seasonal_periods=seasonal_periods, initialization_method='estimated')
    fit_model = model.fit()
    forecast_values = fit_model.forecast()
    return forecast_values

forecast_result = holt_winters_forecast(y_train)
print(forecast_result)


#19: H-step ahead prediction
H = len(y_test)
h_step_prediction = holtw.forecast(steps= H)
print(h_step_prediction)

#predictions of test data for MLR was performed previously (see at MLR part of code)
#Let's plot the test vs forecasted values of MLR

plt.plot(y_test.index,y_test,label='Test')
plt.plot(y_test.index, h_step_prediction,label='Forecasted')
plt.xlabel('Time')
plt.xticks(rotation='vertical')
plt.ylabel('Temperature')
plt.title(' H_step prediction')
plt.legend()
plt.show()
