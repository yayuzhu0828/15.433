import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np

def read_csv(filename):
    return pd.read_csv(filename)

def print_dataframe_stats(df, frequency):
    print("Statistics for " + frequency + " frequency:")
    print("---------")
    print("Mean Market Return: ", df["mktret"].mean())
    print("Var Market Return: ", df["mktret"].var())
    print("Std Dev Market Return: ", df["mktret"].std())
    print("Skewness Market Return: ", df["mktret"].skew())
    print("Kurtosis Market Return: ", df["mktret"].kurtosis())
    print("Mean Bond Return: ", df["bondret"].mean())
    print("Var Bond Return: ", df["bondret"].var())
    print("Std Dev Bond Return: ", df["bondret"].std())
    print("Skewness Bond Return: ", df["bondret"].skew())
    print("Kurtosis Bond Return: ", df["bondret"].kurtosis())
    print("Covariance Market and Bond: ", df["mktret"].cov(df["bondret"]))
    print("Correlation Market and Bond: ", df["mktret"].corr(df["bondret"]))
    print("=======================")

def histogram(df_annual, df_monthly, df_daily):
    fig, axes = plt.subplots(3,2)

    df_annual["mktret"].hist(bins=50, ax=axes[0,0])
    df_annual["bondret"].hist(bins=50, ax=axes[0, 1])
    df_monthly["mktret"].hist(bins=50, ax=axes[1, 0])
    df_monthly["bondret"].hist(bins=50, ax=axes[1, 1])
    df_daily["mktret"].hist(bins=50, ax=axes[2, 0])
    df_daily["bondret"].hist(bins=50, ax=axes[2, 1])

    axes[0,0].set_title("Mkt Ret Annual")
    axes[0, 1].set_title("Bond Ret Annual")
    axes[1, 0].set_title("Mkt Ret Monthly")
    axes[1, 1].set_title("Bond Ret Monthly")
    axes[2, 0].set_title("Mkt Ret Daily")
    axes[2, 1].set_title("Bond Ret Daily")

    fig.tight_layout()

    plt.show()

def confidence_intervals(df, frequency):
    print("Mkt Ret Conf Interval 1 period " + frequency, confidence_interval(df["mktret"]))
    print("Bond Ret Conf Interval 1 period " + frequency, confidence_interval(df["bondret"]))
    df["mktret_ma"] = df["mktret"].rolling(window=30).mean()
    df["bondret_ma"] = df["bondret"].rolling(window=30).mean()
    print("Mkt Ret Conf Interval 30 period " + frequency, confidence_interval(df["mktret_ma"]))
    print("Bond Ret Conf Interval 30 period " + frequency, confidence_interval(df["bondret_ma"]))
    print("==================")


def confidence_interval(df):
    mean = df.mean()
    std = df.std()
    return stats.norm.interval(0.95, loc=mean, scale=std)

def shortfall_plot(df):
    k = [-.2, -.1, 0, .1, .2]
    mkt_mean = df["mktret"].mean()
    mkt_std = df["mktret"].std()
    bond_mean = df["bondret"].mean()
    bond_std = df["bondret"].std()

    mkt_probs = []
    bond_probs = []

    for i in k:
        mkt_probs.append(stats.norm(mkt_mean, mkt_std).cdf(i))
        bond_probs.append(stats.norm(bond_mean, bond_std).cdf(i))

    fig, ax = plt.subplots(1,2)
    ax[0].plot(k, mkt_probs)
    ax[1].plot(k, bond_probs)

    plt.show()

def prob_stock_lower_than_bond(df, frequency):
    mkt_mean = df["mktret"].mean()
    mkt_var = df["mktret"].var()
    bond_mean = df["bondret"].mean()
    bond_var = df["bondret"].var()

    diff_mean = bond_mean - mkt_mean
    cov = df["mktret"].cov(df["bondret"])
    diff_var = bond_var + mkt_var - 2*cov


    prob = 1 - stats.norm.cdf((0 - diff_mean) / np.sqrt(diff_var))
    print("Probability Bond Ret > Stock Ret " + frequency + ": ", prob)

def simulation(df, frequency):
    mkt_mean = df["mktret"].mean()
    mkt_std = df["mktret"].std()
    bond_mean = df["bondret"].mean()
    bond_std = df["bondret"].std()

    mkt_norm = stats.norm(mkt_mean, mkt_std)
    bond_norm = stats.norm(bond_mean, bond_std)

    mkt_samples = mkt_norm.rvs(size=10000)
    bond_samples = bond_norm.rvs(size=10000)

    k = [-.2, -.1, 0, .1, .2]

    mkt_probs = []
    bond_probs = []

    for i in k:
        mkt_k = mkt_norm.ppf(i)
        bond_k = bond_norm.ppf(i)

        num_mkt_less = 0.0
        num_bond_less = 0.0
        for j in range(0,10000):

            if mkt_samples[j] < mkt_k:
                num_mkt_less += 1.0

            if bond_samples[j] < bond_k:
                num_bond_less += 1.0

        mkt_probs.append(num_mkt_less / 10000)
        bond_probs.append(num_bond_less / 10000)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(k, mkt_probs)
    ax[1].plot(k, bond_probs)
    ax[0].set_title(frequency + " Market Fraction < K")
    ax[1].set_title(frequency + " Bond Fraction < K")

    plt.show()

def simulation_empirical(df, frequency):
    mkt_samples = df["mktret"].sample(10000, replace=True).values
    bond_samples = df["bondret"].sample(10000, replace=True).values

    k = [-.2, -.1, 0, .1, .2]

    mkt_probs = []
    bond_probs = []

    for i in k:
        mkt_k = i
        bond_k = i

        num_mkt_less = 0.0
        num_bond_less = 0.0
        for j in range(0,10000):

            if mkt_samples[j] < mkt_k:
                num_mkt_less += 1.0

            if bond_samples[j] < bond_k:
                num_bond_less += 1.0

        mkt_probs.append(num_mkt_less / 10000)
        bond_probs.append(num_bond_less / 10000)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(k, mkt_probs)
    ax[1].plot(k, bond_probs)
    ax[0].set_title(frequency + " Historical Market Fraction < K")
    ax[1].set_title(frequency + " Historical Bond Fraction < K")

    plt.show()

def five_consecutive_known(df, frequency):
    mkt_mean = df["mktret"].mean()
    mkt_var = df["mktret"].var()
    bond_mean = df["bondret"].mean()
    bond_var = df["bondret"].var()

    mkt_norm = stats.norm(5*mkt_mean, np.sqrt(5*mkt_var))
    bond_norm = stats.norm(5*bond_mean, np.sqrt(5*bond_var))

    mkt_samples = mkt_norm.rvs(size=10000)
    bond_samples = bond_norm.rvs(size=10000)

    mkt_ppf = mkt_norm.ppf(.2)
    bond_ppf = bond_norm.ppf(.2)

    num_mkt_less = 0.0
    num_bond_less = 0.0
    for j in range(0,10000):

        if mkt_samples[j] > .2:
            num_mkt_less += 1.0

        if bond_samples[j] > .2:
            num_bond_less += 1.0

    mkt_prob = num_mkt_less / 10000
    bond_prob = num_bond_less / 10000

    print("============")
    print(frequency + " 5 Consecutive Periods Known Market Prob: ", mkt_prob)
    print(frequency + " 5 Consecutive Periods Known Bond Prob: ", bond_prob)

def five_consecutive_empirical(df, frequency):
    mkt_mean = df["mktret"].mean()
    mkt_var = df["mktret"].var()
    bond_mean = df["bondret"].mean()
    bond_var = df["bondret"].var()

    df["mktret_5_sum"] = df["mktret"].rolling(window=5).sum()
    df["bondret_5_sum"] = df["bondret"].rolling(window=5).sum()

    mkt_norm = stats.norm(5 * mkt_mean, np.sqrt(5 * mkt_var))
    bond_norm = stats.norm(5 * bond_mean, np.sqrt(5 * bond_var))

    mkt_samples = df["mktret_5_sum"].sample(10000, replace=True).values
    bond_samples = df["bondret_5_sum"].sample(10000, replace=True).values

    num_mkt_less = 0.0
    num_bond_less = 0.0
    mkt_ppf = mkt_norm.ppf(.2)
    bond_ppf = bond_norm.ppf(.2)
    for j in range(0,10000):

        if mkt_samples[j] > .2:
            num_mkt_less += 1.0

        if bond_samples[j] > .2:
            num_bond_less += 1.0

    mkt_prob = num_mkt_less / 10000
    bond_prob = num_bond_less / 10000

    print("============")
    print(frequency + " 5 Consecutive Periods Empirical Market Prob: ", mkt_prob)
    print(frequency + " 5 Consecutive Periods Empirical Bond Prob: ", bond_prob)

def prob_greater_than_twenty(df, frequency):
    mkt_mean = df["mktret"].mean()
    mkt_var = df["mktret"].var()
    bond_mean = df["bondret"].mean()
    bond_var = df["bondret"].var()

    print("==============")
    mkt_prob = 1 - stats.norm(5*mkt_mean, np.sqrt(5*mkt_var)).cdf(.2)
    print("Probability Market Ret > 20% " + frequency + ": ", mkt_prob)

    bond_prob = 1 - stats.norm(5 * bond_mean, np.sqrt(5 * bond_var)).cdf(.2)
    print("Probability Bond Ret > 20% " + frequency + ": ", bond_prob)

def prob_stock_lower_than_bond_30(df, frequency):
    mkt_mean = 30*df["mktret"].mean()
    mkt_var = np.sqrt(30*df["mktret"].var())
    bond_mean = 30*df["bondret"].mean()
    bond_var = np.sqrt(30*df["bondret"].var())

    df["mktret_30_sum"] = df["mktret"].rolling(window=30).sum()
    df["bondret_30_sum"] = df["bondret"].rolling(window=30).sum()

    diff_mean = bond_mean - mkt_mean
    cov = df["mktret_30_sum"].cov(df["bondret_30_sum"])
    diff_var = bond_var + mkt_var - 2*cov

    prob = 1 - stats.norm.cdf((0 - diff_mean) / np.sqrt(diff_var))
    print("========")
    print("Probability Bond Ret > Stock Ret over 30 periods " + frequency + ": ", prob)

data_annual = read_csv("returns_annual.csv")
data_monthly = read_csv("returns_monthly.csv")
data_daily = read_csv("returns_daily.csv")

print_dataframe_stats(data_annual, "Annual")
print_dataframe_stats(data_monthly, "Monthly")
print_dataframe_stats(data_daily, "Daily")

# histogram(data_annual, data_monthly, data_daily)

confidence_intervals(data_annual, "Annual")
confidence_intervals(data_monthly, "Monthly")
confidence_intervals(data_daily, "Daily")

# shortfall_plot(data_annual)
# shortfall_plot(data_monthly)
# shortfall_plot(data_daily)

prob_stock_lower_than_bond(data_annual, "Annual")
prob_stock_lower_than_bond(data_monthly, "Monthly")
prob_stock_lower_than_bond(data_daily, "Daily")

# simulation(data_annual, "Annual")
# simulation(data_monthly, "Monthly")
# simulation(data_daily, "Daily")

# simulation_empirical(data_annual, "Annual")
# simulation_empirical(data_monthly, "Monthly")
# simulation_empirical(data_daily, "Daily")

prob_greater_than_twenty(data_annual, "Annual")
prob_greater_than_twenty(data_monthly, "Monthly")
prob_greater_than_twenty(data_daily, "Daily")

five_consecutive_known(data_annual, "Annual")
five_consecutive_known(data_monthly, "Monthly")
five_consecutive_known(data_daily, "Daily")

five_consecutive_empirical(data_annual, "Annual")
five_consecutive_empirical(data_monthly, "Monthly")
five_consecutive_empirical(data_daily, "Daily")

prob_stock_lower_than_bond_30(data_annual, "Annual")
prob_stock_lower_than_bond_30(data_monthly, "Monthly")
prob_stock_lower_than_bond_30(data_daily, "Daily")