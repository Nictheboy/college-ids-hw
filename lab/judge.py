# This file is modified from requirement/MyTestStrategy.py,
# which is given by the teacher, used to judge the performance of the model.

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
import matplotlib as mpl
from predict import Predictor
import os
from datetime import datetime

mpl.style.use("classic")


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
        self.df = pd.DataFrame(
            {"Date Time": [], "Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
        )
        self.predictor = Predictor()

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
        _open = bar.getOpen()
        _high = bar.getHigh()
        _low = bar.getLow()
        _close = bar.getClose()
        _volume = bar.getVolume()

        action = 0
        row_count = self.df.shape[0]
        if row_count == 0:
            one_row_list = [_datetime, _open, _high, _low, _close, _volume]
            one_row_array = np.asarray(one_row_list)
            one_row_array = one_row_array.reshape(1, 6)
            self.df = pd.DataFrame(
                one_row_array,
                index=[_datetime],
                columns=["Date Time", "Open", "High", "Low", "Close", "Volume"],
            )
        else:
            one_row_list = [_datetime, _open, _high, _low, _close, _volume]
            one_row_array = np.asarray(one_row_list)
            one_row_array = one_row_array.reshape(1, 6)
            df_one = pd.DataFrame(
                one_row_array,
                index=[_datetime],
                columns=["Date Time", "Open", "High", "Low", "Close", "Volume"],
            )
            self.df = pd.concat([self.df, df_one])
            if self.df.shape[0] >= 34:
                df = self.df
                # df_test = df.iloc[-34:, :].copy()
                df_test = df.copy()
                action = self.predictor.predict(df_test)
        if action == 0:
            pass
        elif action == 1:  # 如果没有头寸，买入
            if self.__position is None:
                shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
                # Enter a buy market order. The order is good till canceled.
                self.__position = self.enterLong(self.__instrument, shares, True)
        elif action == -1:  # 如果有头寸，卖出
            if self.__position is None:
                pass
            elif not self.__position.exitActive():
                self.__position.exitMarket()


def test_one_stock(stock_file_name, png_path):
    stock_name = stock_file_name
    print(f"Testing {stock_name}...")
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

    # print ("Final portfolio value1: $%.2f" % (myTestStrategy.getBroker().getEquity())  )
    print("Final portfolio value2: $%.2f" % (myTestStrategy.getResult()))
    print("Cumulative returns: %.2f %%" % (returnsAnalyzer.getCumulativeReturns()[-1] * 100))
    print("Sharpe ratio: %.2f" % (sharpeRatioAnalyzer.getSharpeRatio(0.03)))
    print("Max. drawdown: %.2f %%" % (drawdownAnalyzer.getMaxDrawDown() * 100))
    print("Trade Count: %d" % (tradesAnalyzer.getCount()))
    # print ("Longest drawdown duration: %s" % (drawdownAnalyzer.getLongestDrawDownDuration())  )

    # Plot the strategy.
    plt.savePlot(png_path)

    # Write log
    # with open("log/judge/judge.csv", "a") as f:
    with open("log/judge/judge_test.csv", "a") as f:
        log = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{stock_name},{myTestStrategy.getResult()},{returnsAnalyzer.getCumulativeReturns()[-1] * 100},{sharpeRatioAnalyzer.getSharpeRatio(0.03)},{drawdownAnalyzer.getMaxDrawDown() * 100},{tradesAnalyzer.getCount()}\n"
        f.write(log)


# while True:
#     files = os.listdir("data/converted")
#     random_file = np.random.choice(files)
#     name = random_file.split(".")[0]
#     test_one_stock(f"data/converted/{random_file}", f"log/judge/random.png")

files = os.listdir("data/converted_test")
for file in files:
    test_one_stock(f"data/converted_test/{file}", f"log/judge/random.png")
