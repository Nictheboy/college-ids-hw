import tushare as ts
import pandas as pd

# Initialize
token = "a953fe53f83c71c6eb63558aa20d83d7cb5a9fdbe40cc6d49027a3ce"
ts.set_token(token)
pro = ts.pro_api()

# Load stock list data
stock_list = pd.read_csv("data/stock_list.csv", sep=",", dtype="str")

# Select stocks that are listed before 2010
stock_list = stock_list[stock_list["list_date"] <= "20100101"]
stock_ids = stock_list["ts_code"].values


def down_load_one_stock(stock_id):
    token = "a953fe53f83c71c6eb63558aa20d83d7cb5a9fdbe40cc6d49027a3ce"
    ts.set_token(token)
    pro = ts.pro_api()

    df = pro.query("daily", ts_code=stock_id, start_date="20100101", end_date="20230101")
    stock_file_name = f"data/download/{stock_id}.csv"
    df.to_csv(stock_file_name)


print("Downloading data...")
download_begin_at_index = 0
stock_ids = stock_ids[download_begin_at_index:]
for i, stock_id in enumerate(stock_ids):
    print(f"  [{download_begin_at_index + i}] - {stock_id}")
    down_load_one_stock(stock_id)
