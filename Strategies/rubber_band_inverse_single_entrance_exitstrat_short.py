import math
import logging
import itertools
import numpy as np
import pandas as pd
import quandl as qd
import janitor as jan
import datetime as dt
from colorama import *
from itertools import groupby
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file, save
from stockstats import StockDataFrame as Sdf
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool

# --- This script is not finalized yet --- #

# Logging to evaluate events.
log_format = '%(levelname)s %(asctime)s - %(message)s'
logging.basicConfig(filename='short_status.Log', level=logging.DEBUG, format=log_format, filemode='w')
logger = logging.getLogger()

# Visual configurations
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

# -- Options to configure the program -- #

folder = '/home/lpitta/Escritorio/Bendio/garage-backtesting-master/Python/'

# Data options
TIME = '1 H'  # resample size  | T = minutes, H = hours, D = days
DATA = pd.read_csv(folder + 'data/ZN_1m_kinetick.csv')

# BT FORCE SETTINGS
FID = (0, 0)
FEE_IS_PLAIN = True
type = 'backtesting'

sample = 'IS1'
strategy_name = 'rubber_band_inverse_single_entrance_exitstrat'
best_parameter = ''
# ITERATE = True  # Set False if you don't want iterate
ITERATE = False  # Set it False if you don't want iterate

#Indicators

# take_profit = [0.005,0.01,0.015,0.02]
# periods = [20]
# standarize_periods = [0, 100, 150, 300]
# z = [2,3,4]
# stop_loss = [0.005,0.01,0.015,0.02]

take_profit = [0.02]
periods = [20]
standarize_periods = [100]
z = [2]
stop_loss = [0.02]


# Parameters
FEE = 3.0
INIT_CAP = 100000
SLIPPAGE_DEF = 0.015625

TICK_SIZE = 0.015625
CONTRACT_SIZE = 1000


# --- code ---- #
def standarize(df_data, column='gap', periods=0):
    if periods == 0:
        df_data['z_' + column] = (df_data[column] - df_data[column].expanding().mean()) / df_data[
            column].expanding().std()

    if periods > 0:
        df_data['z_' + column] = (df_data[column] - df_data[column].rolling(periods).mean()) / df_data[
            column].rolling(periods).std()
    return df_data

def resample_data(data: pd.DataFrame, time: str) -> pd.DataFrame:
    """
    :param data: Historical data in minutes [Open high low close]
    :param time: String with the temporality of the data that you want to obtain
     T = minutes, H = hours, D = days
    :return: DataFrame with data resampled
    """

    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.sort_index(inplace=True)

    # Converting to OHLC format

    data_ohlc = data.resample(time).apply({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                                           'volume': 'sum'})
    data_ohlc.dropna(inplace=True)
    data_ohlc = data_ohlc.reset_index()

    logger.info('Resample of the data was successful')

    return data_ohlc

def strategy(data: pd.DataFrame, parameter) -> tuple:
    """
    :param data: DataFrame with resampled data
    :return: tuple with trades, capital line and data with indicators calculated
    """
    take_profit = parameter[0]
    periods = parameter[1]
    standarize_periods = parameter[2]
    z = parameter[3]
    stop_loss = parameter[4]


    if ITERATE is True:
        print(Fore.LIGHTCYAN_EX + f'Testing strategy with: ({take_profit}, {periods}, {standarize_periods},{z}, {stop_loss})')
        print(Fore.WHITE + f'From: {iss_date_init} To: {iss_date_end}')

    df_data = data
    df_data = Sdf.retype(df_data)
    df_data.reset_index(inplace=True)
    df_data.date = pd.to_datetime(df_data.date)

    # ma
    df_data['ma'] = df_data.close.rolling(int(periods)).mean().shift(1)

    # distance from average

    df_data['distance_mean'] = (df_data.close - df_data.ma)/df_data.ma

    df_data = standarize(df_data, 'distance_mean', periods=int(standarize_periods))

    df_data.dropna(inplace=True)
    df_data.reset_index(inplace=True)

    condition = False
    capital_line = [INIT_CAP]
    capital = INIT_CAP
    df_trades = pd.DataFrame()
    max_periods = int(max(periods, standarize_periods))

    for i in pd.RangeIndex(max_periods, len(df_data)):

        # Sell condition
        if condition is False and df_data.z_distance_mean[i] < -z:

            if i + 1 < len(df_data):

                sell_price = df_data.open[i + 1] - SLIPPAGE_DEF
                stoploss = sell_price + (sell_price * stop_loss)
                st = (sell_price * stop_loss)
                tf = sell_price - (sell_price * take_profit)
                # p_size_contracts = np.ceil((capital * 0.01) / (st * CONTRACT_SIZE))
                p_size_contracts = 1
                margin = 500
                margin_total = margin * p_size_contracts

                if margin_total > capital:
                    p_size_contracts = np.floor(capital / margin)
                if p_size_contracts == 0:
                    break

                p_size_usd_sell = p_size_contracts * CONTRACT_SIZE * sell_price
                total_fee = (FEE * 2) * p_size_contracts
                sell_date = df_data.date[i]

                condition = True

                dfm_aux = pd.DataFrame([{'buy_date': '', 'sell_date': sell_date, 'buy_price': np.NaN,
                                         'sell_price': sell_price, 'quantity': p_size_contracts, 'fee': total_fee,
                                         'trade': '', 'capital': '', 'mae': '', 'quantity_active': p_size_usd_sell,
                                         'incurred_sp': '', 'pl': ''}])

                df_trades = df_trades.append(dfm_aux)
                df_trades.reset_index(inplace=True, drop=True)

                if ITERATE is not True:
                    print(Fore.GREEN + "\n[SELL] " + Fore.WHITE + "Details: Quantity: {} contracts | Price: {} usd"
                          .format(int(p_size_contracts), round(sell_price, 2)))

        # Stop loss condition
        if condition is True and df_data['date'][i] != sell_date:

            if df_data.high[i] >= stoploss:

                if i + 1 < len(df_data):

                    buy_price = stoploss + SLIPPAGE_DEF
                    p_size_usd_buy = p_size_contracts * CONTRACT_SIZE * buy_price
                    total = p_size_usd_sell - p_size_usd_buy - total_fee
                    buy_date = df_data.date[i]
                    capital += total
                    condition = False

                    capital_line.append(capital)

                    # Incurred Slippage
                    incurred_sp = 2 * p_size_contracts * CONTRACT_SIZE * SLIPPAGE_DEF

                    df_trades.at[df_trades.index[-1], 'buy_price'] = buy_price
                    df_trades.at[df_trades.index[-1], 'buy_date'] = buy_date
                    df_trades.at[df_trades.index[-1], 'trade'] = 'stop loss'
                    df_trades.at[df_trades.index[-1], 'capital'] = capital_line[-1]
                    df_trades.at[df_trades.index[-1], 'mae'] = 0
                    df_trades.at[df_trades.index[-1], 'incurred_sp'] = incurred_sp
                    df_trades.at[df_trades.index[-1], 'pl'] = total

                    df_trades.reset_index(inplace=True, drop=True)

                    if ITERATE is not True:
                        print(Fore.RED + "[STOP LOSS] " + Fore.WHITE + "Details: Quantity: {} contracts | Price: {}\n"
                              .format(p_size_contracts, buy_price))

                        print("---------- Trade details ----------")
                        print(f'p size usd buy: {round(p_size_usd_sell, 2)} usd')
                        print(f'p size usd sell: {round(p_size_usd_buy, 2)},  usd')
                        print(f'fee: {round(total_fee, 2)} usd')
                        print(f'result trade: {round(total, 2)} usd')
                        print(f'capital: {round(capital, 2)} usd\n')
                        print(Fore.GREEN + 'sell date: ' + Fore.WHITE + f'{sell_date}')
                        print(Fore.RED + 'buy date: ' + Fore.WHITE + f'{buy_date}')
                        print("------------------------------------")

        # Buy condition

        if condition is True and df_data['date'][i] != sell_date:

            if df_data.close[i] < df_data.ma[i]:
            # if df_data.close[i] <= tf:

                if i + 1 < len(df_data):
                    buy_price = df_data.open[i + 1] + SLIPPAGE_DEF
                    p_size_usd_buy = p_size_contracts * CONTRACT_SIZE * buy_price
                    total = p_size_usd_sell - p_size_usd_buy - total_fee
                    buy_date = df_data.date[i]
                    capital += total
                    condition = False

                    capital_line.append(capital)

                    # Here mae is calculated
                    data_aux = df_data[(df_data['date'] >= f'{sell_date}') & (df_data['date'] <= f'{buy_date}')]
                    max_price = max(data_aux.high)
                    mae = (max_price / sell_price) - 1

                    # Incurred Slippage
                    incurred_sp = 2 * p_size_contracts * CONTRACT_SIZE * SLIPPAGE_DEF

                    df_trades.at[df_trades.index[-1], 'buy_price'] = buy_price
                    df_trades.at[df_trades.index[-1], 'buy_date'] = buy_date
                    df_trades.at[df_trades.index[-1], 'trade'] = 'strategy'
                    df_trades.at[df_trades.index[-1], 'capital'] = capital_line[-1]
                    df_trades.at[df_trades.index[-1], 'mae'] = mae
                    df_trades.at[df_trades.index[-1], 'incurred_sp'] = incurred_sp
                    df_trades.at[df_trades.index[-1], 'pl'] = total

                    df_trades.reset_index(inplace=True, drop=True)

                if ITERATE is not True:
                    print(Fore.RED + "[BUY] " + Fore.WHITE + "Details: Quantity: {} contracts | Price: {} usd\n"
                          .format(int(p_size_contracts), round(buy_price, 2)))

                    print("---------- Trade details ----------")
                    print(f'p size usd buy: {round(p_size_usd_sell, 2)} usd')
                    print(f'p size usd sell: {round(p_size_usd_buy, 2)},  usd')
                    print(f'fee: {round(total_fee, 2)} usd')
                    print(f'result trade: {round(total, 2)} usd')
                    print(f'capital: {round(capital, 2)} usd\n')
                    print(Fore.GREEN + 'sell date: ' + Fore.WHITE + f'{sell_date}')
                    print(Fore.RED + 'buy date: ' + Fore.WHITE + f'{buy_date}')
                    print("------------------------------------")

    logger.info('Correctly created DataFrame with strategy trades')
    df_trades = df_trades[np.isfinite(df_trades['buy_price'])]
    df_trades.reset_index(inplace=True, drop=True)
    return df_trades, capital_line, df_data

def optimizer(iterator: list):
    all_data_optimizer = pd.DataFrame()

    for par in iterator:

        print('')
        print("----- OPTIMIZER BEGIN! -----")

        df_trades, capital_line, data = strategy(data=df_data_fixed, parameter=par)

        if ITERATE is True:
            if not df_trades.empty:
                df_trades = df_trades[np.isfinite(df_trades['buy_price'])]
                max_dd = max_drawdown(capital_line)
                acum_ret = (capital_line[-1] / capital_line[0]) - 1
                num_trades = len(capital_line) - 1
                result_trade = df_trades['pl'].apply(lambda x: 1 if x > 0 else 0).sum()
                percent_profitable = result_trade / num_trades
                percent_loss = 1 - percent_profitable
                avg_profit = df_trades['pl'][df_trades['pl'] > 0].mean()
                avg_loss = df_trades['pl'][df_trades['pl'] < 0].mean()
                profit_factor = avg_profit * percent_profitable / (-1 * avg_loss * percent_loss)

                percent_strategy_exit = 0
                percent_stoploss_exit = 0

                for i in range(0, len(df_trades)):
                    if df_trades.trade[i] == 'strategy':
                        percent_strategy_exit += 1
                    else:
                        percent_stoploss_exit += 1
                if percent_stoploss_exit == 0:
                    percent_strategy_exit = percent_strategy_exit / num_trades

                else:
                    percent_strategy_exit = percent_strategy_exit / num_trades
                    percent_stoploss_exit = 1 - percent_strategy_exit

                mae_mean = df_trades.mae.mean()
                mae_max = df_trades.mae.max()
                mae_min = df_trades.mae.min()

                risk_return = acum_ret / max_dd

                trades_consectv = (df_trades['pl'].apply(lambda x: 1 if x > 0 else -1))
                trades_consectv = [(k, sum(1 for k in g)) for k, g in groupby(trades_consectv)]
                r_trades_aux = []

                for i in trades_consectv:
                    r_trades_aux.append(i[0] * i[1])

                max_consectv_wins = max(r_trades_aux)
                max_consectv_loss = min(r_trades_aux) * -1

                results_isi_par_row = pd.DataFrame({
                    'par': [par],
                    'acum_ret': [acum_ret],
                    'max_drawdown': [max_dd],
                    'num_trades': [num_trades],
                    'percent_profitable': percent_profitable,
                    'percent_loss': percent_loss,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'percent_strategy_exit': percent_strategy_exit,
                    'percent_stoploss_exit': percent_stoploss_exit,
                    'mae_mean': mae_mean,
                    'mae_max': mae_max,
                    'mae_min': mae_min,
                    'risk_return': risk_return,
                    'max_consectv_wins': max_consectv_wins,
                    'max_consectv_loss': max_consectv_loss

                })

                all_data_optimizer = pd.concat([all_data_optimizer, results_isi_par_row])
                all_data_optimizer = all_data_optimizer.round(3)

            else:
                max_dd = max_drawdown(capital_line)
                acum_ret = (capital_line[-1] / capital_line[0]) - 1
                num_trades = len(capital_line) - 1

                results_isi_par_row = pd.DataFrame({
                    'par': [par],
                    'acum_ret': [acum_ret],
                    'max_drawdown': [max_dd],
                    'num_trades': [num_trades]

                })

                all_data_optimizer = pd.concat([all_data_optimizer, results_isi_par_row])
        else:
            # df_trades, capital_line, data = strategy(data=df_data_fixed, parameter=par)
            return df_trades, capital_line, data

    logger.info('all posible parameters has been tested...')
    all_data_optimizer.reset_index(inplace=True, drop=True)

    scoring_optimizer = pd.DataFrame()

    for i in range(0, len(all_data_optimizer)):
        scoring_risk_return = 0.4 * (all_data_optimizer.risk_return[i] - all_data_optimizer.risk_return.min()) / \
                              (all_data_optimizer.risk_return.max() - all_data_optimizer.risk_return.min())

        scoring_acum_ret = 0.2 * (all_data_optimizer.acum_ret[i] - all_data_optimizer.acum_ret.min()) / \
                           (all_data_optimizer.acum_ret.max() - all_data_optimizer.acum_ret.min())

        scoring_profit_factor = 0.2 * (all_data_optimizer.profit_factor[i] - all_data_optimizer.profit_factor.min()) / \
                                (all_data_optimizer.profit_factor.max() - all_data_optimizer.profit_factor.min())

        scoring_max_drawdown = 0.2 * (all_data_optimizer.max_drawdown.max() - all_data_optimizer.max_drawdown[i]) / \
                               (all_data_optimizer.max_drawdown.max() - all_data_optimizer.max_drawdown.min())

        scoring = scoring_risk_return + scoring_acum_ret + scoring_profit_factor + scoring_max_drawdown

        scoring_optimizer_aux = pd.DataFrame([{
            'scoring': scoring
        }])

        scoring_optimizer = scoring_optimizer.append(scoring_optimizer_aux)

    scoring_optimizer.reset_index(inplace=True, drop=True)

    all_data_optimizer = all_data_optimizer.join(scoring_optimizer)
    all_data_optimizer = all_data_optimizer.round(3)
    all_data_optimizer.sort_values('scoring', inplace=True, ascending=False)

    all_data_optimizer.to_csv(f'{folder}results/{sample}_optimizer_{strategy_name}_short.csv')

    print('-' * 270)
    print(Fore.GREEN + 'Iteration results' + Fore.WHITE)
    print(all_data_optimizer)
    print('-' * 270)
    print(Fore.YELLOW + '\n# --- BT FORCE ENDS --- #')
    exit()

def statistics(trades: pd.DataFrame, capital_line: list) -> tuple:
    """
    :param trades: Dataframe with trades
    :param capital_line: Capital line of trades

    :return: Drawdown, positive and negative trades
    """
    if len(trades) > 1:

        # Drawdown

        capital_1 = np.array(capital_line[1:])
        capital_line1 = np.array(capital_line[:-1])

        capital_return = (capital_1 / capital_line1) - 1
        cum_prod = np.cumprod(capital_return + 1)
        cum_max = np.maximum.accumulate(cum_prod)

        drawdown = cum_prod / cum_max - 1
        drawdown = np.delete(drawdown, 0)
        drawdown = np.round(drawdown, 5)

        # statistics

        num_trade_positive = (trades[trades['pl'] > 0].shape[0])
        num_trade_negative = (trades[trades['pl'] < 0].shape[0])
        largest_win_trade = trades.pl.max()
        largest_loss_trade = trades.pl.min()
        return_accumulated = (capital_line[-1] / capital_line[0]) - 1
        num_trades = num_trade_positive + num_trade_negative
        percent_profit = num_trade_positive / num_trades
        risk_return = return_accumulated / (drawdown.min() * -1)
        result_trade = trades['pl'].apply(lambda x: 1 if x > 0 else 0).sum()
        percent_profitable = result_trade / num_trades
        percent_loss = 1 - percent_profitable
        avg_profit = trades['pl'][trades['pl'] > 0].mean()
        avg_loss = trades['pl'][trades['pl'] < 0].mean()
        profit_factor = avg_profit * percent_profitable / (-1 * avg_loss * percent_loss)

        percent_strategy_exit = 0
        percent_stoploss_exit = 0

        for i in range(0, len(trades)):
            if trades.trade[i] == 'strategy':
                percent_strategy_exit += 1
            else:
                percent_stoploss_exit += 1
        if percent_stoploss_exit == 0:
            percent_strategy_exit = percent_strategy_exit / num_trades

        else:
            percent_strategy_exit = percent_strategy_exit / num_trades
            percent_stoploss_exit = 1 - percent_strategy_exit

        mae_mean = trades.mae.mean()
        mae_max = trades.mae.max()
        mae_min = trades.mae.min()

        trades_consectv = (trades['pl'].apply(lambda x: 1 if x > 0 else -1))
        trades_consectv = [(k, sum(1 for k in g)) for k, g in groupby(trades_consectv)]
        r_trades_aux = []

        for i in trades_consectv:
            r_trades_aux.append(i[0] * i[1])

        max_consectv_wins = max(r_trades_aux)
        max_consectv_loss = min(r_trades_aux) * -1

        print(Fore.CYAN + "\n# ------- basic statistics ------- #")
        print(Fore.WHITE + 'Cumulative return: ', round(return_accumulated, 2))
        print('Risk return', round(risk_return, 2))
        print('Largest win trade: ', round(largest_win_trade, 2))
        print('Largest loss trade: ', round(largest_loss_trade, 2))
        print('Percent profit: ', round(percent_profit, 2))
        print('Percent loss: ', round(percent_loss, 2))
        print('Avg profit: ', round(avg_profit, 2))
        print('Avg loss: ', round(avg_loss, 2))
        print('Profit factor: ', round(profit_factor, 2))
        print('Percent strategy exit', round(percent_strategy_exit, 2))
        print('Percent stoploss exit', round(percent_stoploss_exit, 2))
        print('Mae mean: ', round(mae_mean, 3))
        print('Mae max: ', round(mae_max, 3))
        print('Mae min: ', round(mae_min, 3))
        print('Max consectv wins: ', max_consectv_wins)
        print('Max consectv loss: ', max_consectv_loss)
        print('Number of positive trades: ', num_trade_positive)
        print('Number of negative trades: ', num_trade_negative)
        print('Num trades: ', num_trades)
        print(Fore.CYAN + "# --------------------------------- #\n")

        logger.info('Statistics created successfully')

        return drawdown, num_trade_positive, num_trade_negative

    else:

        print(Fore.RED + 'The number of trades is not enough to calculate the statistics and show the graphs')
        logger.info('Error: The number of trades is not enough to calculate the statistics and show the graphs')
        exit()

def graphics():
    """
    :param max_drawdown: Drawdown from statistics function
    :param po_trades: Positive trades
    :param ne_trades: Negative trades
    :return: Graphics of the strategy, capital line, drawdown, positive and negative trades
    """

    # Candlestick
    inc = data.close > data.open
    dec = data.open > data.close
    w = 30 * 30 * 30 * 30

    source = ColumnDataSource(df_trades)

    p_hover_entry = HoverTool(
        names=["buy_condition"],

        tooltips=[
            ("buy date", "@buy_date{%Y-%m-%d %H hour}"),
            ("buy price", "@buy_price"),
            ("type", "Buy")
        ],

        formatters={
            'buy_date': 'datetime',
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    )
    p_hover_exit = HoverTool(
        names=["sell_condition"],

        tooltips=[
            ("sell date", "@sell_date{%Y-%m-%d %H hour}"),
            ("sell price", "@sell_price"),
            ("type", "@trade")
        ],

        formatters={
            'sell_date': 'datetime',
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    )

    crosshair = CrosshairTool(dimensions='both')

    #data['stoploss'] = data.close - (data.atr * ATR_COEF)  # ATR

    # Figures
    p = figure(x_axis_type="datetime", plot_height=500, plot_width=1500, title="10 Year T-Note")
    p_aux = figure(x_axis_type="datetime", plot_height=350, plot_width=1500, title="Standard Deviations", x_range=p.x_range)

    p.segment(data.date, data.high, data.date, data.low, color="black")
    p.vbar(data.date[inc], w, data.open[inc], data.close[inc], fill_color="green", line_color="black")
    p.vbar(data.date[dec], w, data.open[dec], data.close[dec], fill_color="red", line_color="black")

    # Tools
    p.add_tools(p_hover_entry, p_hover_exit,  crosshair)
    p_aux.add_tools(crosshair)

    # Graphics
    p.line(data.date, data.ma, line_color="red", legend='MA')
    p_aux.line(data.date, data.z_distance_mean, line_color="blue", legend='Z Mean Distance')

    # Axis of graphics
    p.xaxis.axis_label = 'TIME'
    p.yaxis.axis_label = 'PRICE (USD)'

    # Buy and Sell condition
    p.circle('buy_date', 'buy_price', fill_color="green", line_color="black", legend='BUY CONDITION',
             size=12, fill_alpha=0.8, source=source, name='buy_condition')

    p.circle('sell_date', 'sell_price', fill_color="red", line_color="black", legend='SELL CONDITION',
             size=12, fill_alpha=0.8, source=source, name='sell_condition')

    g = gridplot([[p], [p_aux]], sizing_mode='scale_width')
    show(g)
    output_file(f'{folder}graphic_validation/{strategy_name}_short_graphic_validation.html', mode='inline')
    save(g)

    logger.info('Graphics generated correctly')


def max_drawdown(capital_line):
    """
    It's a simple function that takes a capital line and outputs the max drawdown.

    Beware: max_drawdown could be zero!

    :param capital_line: capital line shows the evolution of the capital over time.
    :return: float max drawdown
    """

    if len(capital_line) <= 2:
        # FIXME: I NEED TO RETURN A VALUE DISTINCT FROM 0 SOME TIMES...
        return 0

    capital_1 = np.array(capital_line[1:])
    capital_line = np.array(capital_line[:-1])

    capital_return = capital_1 / capital_line - 1
    cum_prod = np.cumprod(capital_return + 1)
    cum_max = np.maximum.accumulate(cum_prod)

    drawdown = cum_prod / cum_max - 1
    drawdown = np.delete(drawdown, 0)

    max_dd = math.fabs(drawdown.min())

    return max_dd


def mender(df_to_change, pos_fid):
    """
    Mender takes care of the name of the columns, basically in order to push the results to the DB the
    name variables must coincide with the ones programmed in R so... mender takes care of transforming
    the trades_df in to the proposed format.

    :param df_to_change: usually this is the result of trades_matrix
    :param pos_fid: this is the feature id code that every strategy must have to identify itself in the DB,
    in this case is the position of the corresponding feature id in the tuple defined above in the first
    section of the code.
    :return: in the mended trades matrix, ready to be push to the DB!
    """

    result_to_change = df_to_change.copy()

    try:
        result_to_change['capital'] = (result_to_change['capital']
                                       .astype('float64'))
    except ValueError:
        result_to_change = result_to_change[:-1]
        return mender(result_to_change, pos_fid)

    result_to_change.rename(
        columns={
            'sell_date': 'entry_date',
            'buy_date': 'exit_date',
            'sell_price': 'entry_price',
            'buy_price': 'exit_price',
            'quantity': 'quantity_passive',
            'trade': 'exit'
        },
        inplace=True
    )

    try:
        if FEE_IS_PLAIN is False:
            result_to_change['fees'] = (result_to_change['fee']
                                        * result_to_change['entry_price']
                                        * 2)
        else:
            result_to_change['fees'] = result_to_change['fee']

    except KeyError:
        return result_to_change

    result_to_change['feature_id'] = FID[pos_fid]

    result_to_change['risk_perc'] = 0

    result_to_change['pl'] = (result_to_change['capital']
                              - result_to_change['capital'].shift(1))

    result_to_change['pl'].iloc[0] = (result_to_change['capital'].iloc[0]
                                      - INIT_CAP)

    result_to_change['cap_shift'] = result_to_change['capital'].shift(1)
    result_to_change.at[0, 'cap_shift'] = INIT_CAP

    result_to_change['cap_return'] = (
            np.log(result_to_change['capital'].astype('float64'))
            - np.log(result_to_change['cap_shift'].astype('float64'))
    )

    result_to_change['price_diff'] = result_to_change['exit_price'] - result_to_change['entry_price']

    result_to_change['return_trade'] = result_to_change['price_diff'] / result_to_change['entry_price']

    try:

        del result_to_change['cap_shift']
        del result_to_change['fee']
        del result_to_change['out_of_sample']
        del result_to_change['parameter']

    except KeyError:
        pass

    return result_to_change


def ws(trades, pos_fid, is_iss=False):
    """
    The glory hell of all functions here in walk_forward algorithm.
    This is suppose top calculate all the indicator corresponding to an strategy, it must show them
    for every trading window.
    :param trades: the corresponding trades_matrix result DataFrame
    :param is_iss: this is just a temporary parameters, if True them force the capital line to init with
    INIT_CAP as is did in the apply_strategy.
    :return: the ws DataFrame ready to be uploaded to the DB!
    """

    target_variables = {'to', 'mae_max', 'best_params', 'avg_loss', 'avg_bars_win', 'trading_months',
                        'percent_profitable', 'avg_bars_loss', 'largest_loss_trade', 'mean_duration_trans',
                        'feature_id', 'trans_over_24h', 'percent_time_in_market', 'max_consectv_wins', 'sample',
                        'sharpe_ratio', 'accum_return', 'avg_profits', 'window', 'fromm', 'mae_mean',
                        'max_consectv_loss', 'annual_return', 'calmar_ratio', 'net_profit', 'profit_factor',
                        'largest_win_trade', 'perc_stoploss_exit', 'perc_stra_exit', 'percent_loss', 'mae_min',
                        'n_trades', 'risk_return', 'max_drawdown'}

    df = trades.copy()

    # it doesn't matter which pos_fid we pass to mender, we only want the pl column
    jf = mender(df, pos_fid).copy()

    try:
        df['pl'] = jf['pl']
    except KeyError:
        return pd.DataFrame()

    # how manny trades were won
    df['result_trade'] = df['pl'].apply(lambda x: 1 if x > 0 else 0)

    # duration of the a trade

    df['capital_end'] = df['capital']

    # this is like this because GroupBy orders alphabetically and not by the number of window (that's the goal), so
    # in order to GroupBy correctly we have to group by this variable by the number of the sample instead of the sample
    # eg: group by 8, 10, 15 instead of IS8, IS10, IS15. If not, the order will be wrong: IS10, IS15, 1S8 despite IS8
    # needs to be first. This problem was debugged with a lot o suffer.

    df['out_of_sample_number'] = df['out_of_sample'].apply(lambda x: int(x[2:]))

    # group the trades into windows

    df['diff_time'] = ((df['buy_date'].astype('datetime64[h]')
                        - df['sell_date'].astype('datetime64[h]'))
                       / np.timedelta64(1, 'h'))
    capaux = (df
              .groupby('out_of_sample_number')
              .agg({'capital': lambda x: x.iloc[0],
                    'capital_end': lambda x: x.iloc[-1],
                    'fee': 'count',
                    'result_trade': 'sum',
                    'buy_date': 'max',
                    'sell_date': 'min',
                    'diff_time': 'mean',
                    'out_of_sample': 'min'}))

    capaux['long_in_days'] = ((capaux['buy_date'].astype('datetime64[h]') -
                               capaux['sell_date'].astype('datetime64[h]')) / np.timedelta64(1, 'D'))

    # indexes is aka as the windows.
    indexes = list(capaux.index)
    first_index = indexes[0]

    if is_iss is True:
        # if here you want to initiate every window whit the same capital, since it is a iss.
        capaux['capital'] = INIT_CAP
    else:
        # here we are handling the capital in order to calculate net profit, acum return...
        capaux['capital'] = capaux['capital_end'].shift(1)
        capaux.at[first_index, 'capital'] = INIT_CAP

    # creating some needed variables
    capaux['mean_duration_trans'] = capaux['diff_time']

    try:
        capaux['net_profit'] = capaux['capital_end'] - capaux['capital']
    except Exception as e:
        print(e)
        print('error')

    percent_strategy_exit = 0
    percent_stoploss_exit = 0

    for i in range(0, len(df)):
        if df.trade[i] == 'strategy':
            percent_strategy_exit += 1
        else:
            percent_stoploss_exit += 1
    if percent_stoploss_exit == 0:
        percent_strategy_exit = percent_strategy_exit / capaux['fee']

    else:
        percent_strategy_exit = percent_strategy_exit / capaux['fee']
        percent_stoploss_exit = 1 - percent_strategy_exit

    capaux['perc_stra_exit'] = percent_strategy_exit
    capaux['perc_stoploss_exit'] = percent_stoploss_exit

    capaux['percent_profitable'] = capaux['result_trade'] / capaux['fee']
    capaux['percent_loss'] = 1 - capaux['percent_profitable']

    avg_profit = df[df['pl'] > 0].groupby('out_of_sample_number').agg({'pl': 'mean'})
    avg_loss = df[df['pl'] < 0].groupby('out_of_sample_number').agg({'pl': 'mean'})

    capaux['best_params'] = (df
        .groupby('out_of_sample_number')
        .agg({'parameter': lambda x: x.iloc[0]})['parameter'])

    capaux['avg_profits'] = avg_profit['pl']
    capaux['avg_loss'] = avg_loss['pl']

    capaux['profit_factor'] = (
            (capaux['avg_profits'] * capaux['percent_profitable'])
            / (-1 * capaux['avg_loss'] * capaux['percent_loss'])
    )

    # getting the start and end date of every window
    merge_dict = {**period}
    merge_dict_df = pd.DataFrame(merge_dict).transpose()

    merge_dict_df.rename(columns={0: 'fromm', 1: 'to'}, inplace=True)

    # from now and on we do not need the 'out_of_sample_number' var to be the index
    capaux.set_index('out_of_sample', inplace=True)

    capaux['fromm'] = merge_dict_df['fromm']
    capaux['to'] = merge_dict_df['to']

    capaux['feature_id'] = jf['feature_id'].iloc[0]

    bucket1, bucket2 = [], []
    for idx in capaux.index:
        aux = df[df['out_of_sample'] == idx]

        if is_iss is True:
            capi = [INIT_CAP] + list(aux['capital'])
        else:
            capi = list(aux['capital'])

        md = max_drawdown(capi)
        # cr = cumulative_return(capi) No se utiliza aparentemente

        mean_pl = aux['pl'].mean()
        std_pl = aux['pl'].std()

        bucket2.append(mean_pl / std_pl)

        bucket1.append(md)

    capaux['accum_return'] = capaux['capital_end'] / capaux['capital'] - 1

    capaux['annual_return'] = -1 + (capaux['accum_return'] + 1) ** (365 / capaux['long_in_days'])

    capaux['max_drawdown'] = bucket1

    # capaux['sharpe_ratio'] = bucket2 Ya esta en el dashboard
    capaux['sharpe_ratio'] = 0

    capaux.rename(columns={'fee': 'n_trades'}, inplace=True)

    capaux.reset_index(inplace=True)
    capaux['sample'] = capaux['out_of_sample'].apply(lambda x: 'in' if x[0] == 'I' else 'out')
    capaux['window'] = capaux['out_of_sample'].apply(lambda x: x[2:])
    capaux['risk_return'] = capaux['accum_return'] / capaux['max_drawdown']

    # hall of shame:
    capc = set(capaux.columns)

    for col in target_variables - capc:
        capaux[col] = 0
    for col in capc - target_variables:
        del capaux[col]

    return capaux


if __name__ == '__main__':

    print(Fore.YELLOW + '\n# --- BT FORCE BEGINS --- #')
    print(Fore.WHITE)
    logger.info('--- BACK TESTING PROCESS BEGINS ---')

    iterator = list(itertools.product(take_profit, periods, standarize_periods, z, stop_loss))
    sample = sample.upper()
    df_data_fixed = resample_data(data=DATA, time=TIME)

    window_iss = len(df_data_fixed) // 2
    iss_date_init = df_data_fixed.date[0]
    iss_date_end = df_data_fixed.date[window_iss]

    if sample == 'IS1':

        df_data_fixed = df_data_fixed[(df_data_fixed['date'] >= f'{iss_date_init}')
                                      & (df_data_fixed['date'] <= f'{iss_date_end}')]

        period = {f'{sample}': [f'{iss_date_init}', f'{iss_date_end}']}

    else:
        oss_date_init = df_data_fixed.date[window_iss]
        oss_date_end = df_data_fixed.date.iloc[-1]
        df_data_fixed = df_data_fixed[
            (df_data_fixed['date'] >= f'{oss_date_init}') & (df_data_fixed['date'] <= f'{oss_date_end}')]

        period = {f'{sample}': [f'{oss_date_init}', f'{oss_date_end}']}

    df_trades, capital_line, data = optimizer(iterator=iterator)

    # Back testing Force
    df_trades['out_of_sample'] = sample
    df_trades['parameter'] = best_parameter

    trades_fix = mender(df_trades, 1)
    ws = ws(df_trades, 1)

    # Statistics and graphics
    statistics(trades=df_trades, capital_line=capital_line)

    # Graphics disabled by default
    graphics()

    # Saving trades and ws into .csv
    if sample == 'IS1':
        trades_fix['sample'] = 'in'
        trades_fix.to_csv(f'{folder}results/{strategy_name}_trades_short_in.csv', index=False)
        ws.to_csv(f'{folder}results/{strategy_name}_short_windows_summary.csv', index=False)

    else:
        try:
            ws_oss = pd.read_csv(f'database_{strategy_name}_short_windows_summary.csv')
            ws = ws.append(ws_oss)

        except FileNotFoundError:
            print(Fore.RED + 'WARNING: Run first the IS ! | sample = IS1')
            exit()

        ws.to_csv(f'{folder}results/{strategy_name}_short_windows_summary.csv', index=False)

        trades_fix['sample'] = 'out'
        trades_fix.to_csv(f'{folder}results/{strategy_name}_short_trades_short_out.csv', index=False)

    logger.info('Successful! all functions worked perfectly')
    logger.info('--- BT FORCE ENDS ---')
    print(Fore.YELLOW + '\n# --- BT FORCE ENDS --- #')
