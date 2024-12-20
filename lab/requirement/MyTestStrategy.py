from pyalgotrade.barfeed.csvfeed import GenericBarFeed
from pyalgotrade.bar import Frequency

from pyalgotrade.technical import ma

from pyalgotrade import strategy
from pyalgotrade.stratanalyzer import returns
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.stratanalyzer import drawdown
from pyalgotrade.stratanalyzer import trades
from pyalgotrade import plotter

import pandas as pd
import numpy as np
import talib as ta
import pickle

import matplotlib as mpl
mpl.style.use('classic')

class MyTestStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, smaPeriod1, smaPeriod2):
        super(MyTestStrategy, self).__init__(feed, 100000)
        self.__instrument = instrument
        self.__position = None
        # We'll use adjusted close values instead of regular close values.
        self.setUseAdjustedValues(True)
        self.__prices = feed[instrument].getPriceDataSeries()
        self.__sma1 = ma.SMA(self.__prices, smaPeriod1)
        self.__sma2 = ma.SMA(self.__prices, smaPeriod2)
        
        
        self.df = pd.DataFrame( {
                'Open': [],
                'High': [],
                'Low': [],
                'Close': [],
                'Volume': []
                })
        #self.df  = pd.DataFrame(columns=['Open', 'High', 'Low','Close','Volume'])
        print("df",self.df.head())
        
        # load model
        filename = 'stock.pkl'
        self.model = pickle.load(open(filename, 'rb'))


    def getSMA1(self):
        return self.__sma1
    def getSMA2(self):
        return self.__sma2
    
    def onEnterCanceled(self, position):
        self.__position = None

    def onExitOk(self, position):
        self.__position = None

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__position.exitMarket()

    def onBars(self, bars):
        bar = bars[self.__instrument]
        
        _datetime = bar.getDateTime()
        #print("_datetime",_datetime)
        _open = bar.getOpen()
        _high = bar.getHigh()
        _low = bar.getLow()
        _close = bar.getClose()
        _volume = bar.getVolume()
        
        action = 0
        row_count = self.df.shape[0]
        if(row_count ==0):
            one_row_list = [_open, _high, _low, _close, _volume]
            one_row_array = np.asarray( one_row_list)
            one_row_array = one_row_array.reshape(1,5)
            self.df = pd.DataFrame(one_row_array, 
                                   index=[_datetime],
                                   columns=['Open','High','Low','Close','Volume'])

            print("self.df + one row",self.df.head())
        else:
            one_row_list = [_open, _high, _low, _close, _volume]
            one_row_array = np.asarray( one_row_list)
            one_row_array = one_row_array.reshape(1,5)
            df_one = pd.DataFrame(one_row_array, 
                                   index=[_datetime],
                                   columns=['Open','High','Low','Close','Volume'])

            self.df = pd.concat([self.df, df_one])
            
            if(self.df.shape[0] == 6):
                print("self.df",self.df.head())
            
            if(self.df.shape[0] >=34):
                df = self.df
                df_test = df.iloc[-34:,:].copy()
                #del df_test['Adj Close']
                
                df_test.loc[:,'DIFF_1'] = df_test.loc[:,"Close"].diff()
                df_test.loc[:,'DIFF_2'] = df_test.loc[:,"DIFF_1"].shift(1)
                df_test.loc[:,'DIFF_3'] = df_test.loc[:,"DIFF_2"].shift(1)

                df_test.loc[:,'MA13']=ta.MA(df_test.Close,timeperiod=13)
                df_test.loc[:,'MA33']=ta.MA(df_test.Close,timeperiod=33)
                df_test.loc[:,'MA13_MA33'] = df_test.loc[:,'MA13']- df_test.loc[:,'MA33']
                
                df_test.loc[:,'EMA10']=ta.EMA(df_test.Close,timeperiod=10)
                df_test.loc[:,'EMA30']=ta.EMA(df_test.Close,timeperiod=30)
                df_test.loc[:,'EMA10_EMA30'] = df_test.loc[:,'EMA10']- df_test.loc[:,'EMA30']
                    
                df_test.loc[:,'MOM10']=ta.MOM(df_test.Close,timeperiod=10)
                df_test.loc[:,'MOM30']=ta.MOM(df_test.Close,timeperiod=30)
                df_test.loc[:,'MOM10_MOM30'] = df_test.loc[:,'MOM10']- df_test.loc[:,'MOM30']
                    
                df_test.loc[:,'RSI10']=ta.RSI(df_test.Close,timeperiod=10)
                df_test.loc[:,'RSI30']=ta.RSI(df_test.Close,timeperiod=30)
                
                df_test.loc[:,'K10'],df_test.loc[:,'D10']=ta.STOCH(df_test.High,df_test.Low,df_test.Close, fastk_period=10)
                df_test.loc[:,'K10_D10'] = df_test.loc[:,'K10']- df_test.loc[:,'D10']
                
                df_test.loc[:,'K30'],df_test.loc[:,'D30']=ta.STOCH(df_test.High,df_test.Low,df_test.Close, fastk_period=30)
                df_test.loc[:,'K30_D30'] = df_test.loc[:,'K30']- df_test.loc[:,'D30']
                
                X = df_test.copy()
                X = X.iloc[-1:,:]
                y_predict = self.model.predict(X)
                #print(y_predict)
                action=y_predict[0]
                if action==1:
                    print("BUY Signal =", action)
                if action ==-1:
                    print("SelL Signal =", action)
                
        if(action == 0):
            pass
        elif (action == 1):#如果没有头寸，买入
            if self.__position is None:
                shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
                # Enter a buy market order. The order is good till canceled.
                self.__position = self.enterLong(self.__instrument, shares, True)
                print("BUY Action")
        elif (action == -1):#如果有头寸，卖出
            if self.__position is None:
                pass
            elif not self.__position.exitActive():
                self.__position.exitMarket()
                print("SELL Action")
  
def test_one_stock( stock_name, stock_file_name):
    feed = GenericBarFeed(Frequency.DAY, None, None)  
    feed.addBarsFromCSV(stock_name, stock_file_name)
    smaPeriod1 = 13
    smaPeriod2 = 33
    myTestStrategy = MyTestStrategy(feed, stock_name, smaPeriod1, smaPeriod2)
    
    # Attach analyzers to the strategy
    returnsAnalyzer = returns.Returns()
    myTestStrategy.attachAnalyzer(returnsAnalyzer)
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    myTestStrategy.attachAnalyzer(sharpeRatioAnalyzer)
    drawdownAnalyzer = drawdown.DrawDown()
    myTestStrategy.attachAnalyzer(drawdownAnalyzer)
    tradesAnalyzer = trades.Trades()
    myTestStrategy.attachAnalyzer(tradesAnalyzer)
    
    
    # Attach the plotter to the strategy.
    plt = plotter.StrategyPlotter(myTestStrategy)
    # Include the SMA in the instrument's subplot to get it displayed along with the closing prices.
    plt.getInstrumentSubplot(stock_name).addDataSeries("SMA1", myTestStrategy.getSMA1())
    plt.getInstrumentSubplot(stock_name).addDataSeries("SMA2", myTestStrategy.getSMA2())
    # Plot the simple returns on each bar.
    plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer.getReturns())
    
    # Run the strategy.
    myTestStrategy.run()
    
    #print ("Final portfolio value1: $%.2f" % (myTestStrategy.getBroker().getEquity())  )
    print ("Final portfolio value2: $%.2f" % (myTestStrategy.getResult())  )
    print ("Cumulative returns: %.2f %%" % (returnsAnalyzer.getCumulativeReturns()[-1] * 100)  )
    print ("Sharpe ratio: %.2f" % (sharpeRatioAnalyzer.getSharpeRatio(0.03))  )
    print ("Max. drawdown: %.2f %%" % (drawdownAnalyzer.getMaxDrawDown() * 100)  )
    print ("Trade Count: %d" % (tradesAnalyzer.getCount())  )
    #print ("Longest drawdown duration: %s" % (drawdownAnalyzer.getLongestDrawDownDuration())  )
    
    # Plot the strategy.
    plt.plot()            

test_one_stock("stock01", "./stock01.csv")
#test_one_stock("stock02", "./stock02.csv")
