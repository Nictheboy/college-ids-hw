from pyalgotrade.feed import csvfeed
import os


# read_write_CSV
def read_write_CSV(from_file, to_file):
    feed = csvfeed.Feed("trade_date", "%Y%m%d")
    feed.addValuesFromCSV(from_file)

    file = open(to_file, "w")
    file.write("Date Time,Open,High,Low,Close,Volume,Adj Close")
    file.write("\n")
    for dateTime, value in feed:
        # print dateTime,value['open'],value['high'],value['low'],value['close'],value['vol'],value['close']
        strdatetime = dateTime.strftime("%Y-%m-%d %H:%M:%S")
        file.write(
            "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f"
            % (
                strdatetime,
                value["open"],
                value["high"],
                value["low"],
                value["close"],
                value["vol"],
                value["close"],
            )
        )
        file.write("\n")

    file.close()


# For all files in data/download
for f in os.listdir("data/download"):
    if os.path.isfile(os.path.join("data/download", f)):
        print(f)
        read_write_CSV("data/download/" + f, "data/converted/" + f)
