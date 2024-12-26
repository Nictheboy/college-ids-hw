import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


def describe(df: DataFrame, column: str) -> tuple[str, str, list]:
    avg = df[column].mean()
    std = df[column].std()
    description = f"{avg:.2f} ± {std:.2f}"
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    outliers = df[(df[column] < q1 - 1.5 * (q3 - q1)) | (df[column] > q3 + 1.5 * (q3 - q1))]
    df[column] = df[column].clip(
        q1 - 1.5 * (q3 - q1),
        q3 + 1.5 * (q3 - q1),
    )
    avg_adj = df[column].mean()
    std_adj = df[column].std()
    description_adj = f"{avg_adj:.2f} ± {std_adj:.2f}"
    return description, description_adj, outliers[column].tolist()


def plot(df: DataFrame, column: str, img: str):
    plt.figure(figsize=(5, 3))
    df[column].plot.hist(bins=50, alpha=0.5, color="black")
    plt.title(column)
    plt.savefig(img)
    plt.close()


def analyze(filename: str, img_prefix: str):
    log = pd.read_csv(filename)
    columns = ["Total", "Returns", "Sharpe Ratio", "Max Drawdown", "Trade Count"]
    descriptions = [describe(log, column) for column in columns]
    rows = 3 + max([len(description[2]) for description in descriptions])
    statistic = pd.DataFrame(
        {
            "": ["Avg ± Std", " - no outlier", "", "Outliers"] + [""] * (rows - 3),
            **{
                column: [description[0], description[1]]
                + [""]
                + [f"{val:.2f}" for val in description[2]]
                + [""] * (rows - 2 - len(description[2]))
                for column, description in zip(columns, descriptions)
            },
        }
    ).set_index("")
    print()
    print(statistic)
    print()

    # Draw distributions
    for column in columns:
        plot(log, column, f"{img_prefix}{column}.png")


def main():
    analyze("log/judge/judge.csv", "report/img/1/")
    analyze("log/judge/judge_test.csv", "report/img/2/")


if __name__ == "__main__":
    main()
