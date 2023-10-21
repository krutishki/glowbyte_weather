import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def plot_prediction(df: pd.DataFrame, title: str = "") -> None:
    # df - датасет с колонками таргет и предикт, а еще с колонкой datetime
    fig = px.line(df, x="datetime", y=["target", "predict"], title=title)
    fig.show()

def plot_target_boxplots(df: pd.DataFrame) -> None:
    # Draw Plot
    fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
    sns.boxplot(x=df['datetime'].dt.year, y=df['target'], ax=axes[0])
    sns.boxplot(x=df['datetime'].dt.month, y=df['target'], ax=axes[1])
    # sns.boxplot(x='month', y='Sales', data=train.loc[~train.datetime.dt.year.isin([2014,2917]), :])

    # Set Title
    axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
    axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
    plt.show()

def plot_time_series(timeseries: pd.Series, window: int = 30, title: str = "") -> None:
    """
    Рисует график временного ряда со скользящим  средним и скользящим стандартным отклонением
    """

    #Determing rolling statistics
    if title != "":
        title = 'Rolling Mean & Standard Deviation: ' + title
    else:
        title = 'Rolling Mean & Standard Deviation'
    
    plt.figure(figsize=(15, 6))
    moving_avg = timeseries.rolling(30).mean()
    moving_std = timeseries.rolling(30).std()
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(moving_avg, color='red', label='Rolling Mean')
    std = plt.plot(moving_std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title(title)
    plt.show(block=False)

def test_stationarity(timeseries, window = 12):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    plt.figure(figsize=(15, 6))

    plot_time_series(timeseries, window, title = "Stationarity test")
    
    #Perform Dickey-Fuller test:
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    dfoutput['is_stationary'] = dfoutput['p-value'] < 0.05
    print (dfoutput)