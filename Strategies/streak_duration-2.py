import warnings
import itertools
import numpy as np
import pandas as pd
from colorama import *
from quantstats.stats import *
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from stockstats import StockDataFrame as Sdf
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 1000)

# Hide Pandas Future Warning
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# Variables
object_variables = {'symbols_1': 'CL',
                    'temporality': '1H',
                    'fee': 1.52,
                    'tick_size': 0.1,
                    'contract_size': 100,
                    'direction': 'Short',
                    'capital': 500000,
                    'margin': 5600,
                    'strategy_name': 'streak_duration',
                    'parameters': {'profit_target': [0.01],
                                   'zv_value': [2],
                                   'periods': [50],
                                   'stop_loss': [0.005]}}


def get_data(symbol: str):
    df_data = pd.read_csv(f'{symbol}.csv')

    print(f"[{symbol}] - Get_data done.\n")
    return df_data


def resample_data(data: pd.DataFrame, time: str):
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.sort_index(inplace=True)

    # Converting to OHLC format
    data_ohlc = data.resample(time).apply({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                                           'volume': 'sum'})
    data_ohlc.dropna(inplace=True)
    data_ohlc = data_ohlc.reset_index()

    print("Function resample_data done.\n")

    return data_ohlc

def long_direction(df_data, object_variables, take_profit, zv_value, stop_loss):
    capital = float(object_variables["capital"])
    capital_line = [capital]
    condition = False

    df_trades = pd.DataFrame()
    # max_periods = round(max(variable1, variable2, variable3))
    max_periods = 14

    for i in range(max_periods, len(df_data)):

        # ENTRY LONG CONDITION
        if not condition and df_data['zv'][i] >= zv_value and df_data['close'][i] > df_data['open'][i]:

            if i + 1 < len(df_data):

                entry_price = df_data.open[i + 1] + object_variables["tick_size"]
                stoploss = entry_price - (entry_price * stop_loss)
                position_size = np.round((capital * 0.01) / (entry_price * stop_loss *
                                                             object_variables["contract_size"]))
                margin_total = object_variables["margin"] * position_size

                if margin_total > capital:
                    position_size = np.floor(capital / object_variables["margin"])

                if position_size == 0:
                    break

                tf = entry_price + (entry_price * take_profit)

                entry_position_size_usd = position_size * object_variables["contract_size"] * entry_price
                total_fee = object_variables["fee"] * 2 * position_size
                entry_date = df_data.date[i]

                condition = True

                dfm_aux = pd.DataFrame([{'entry_date': str(entry_date), 'exit_date': '', 'entry_price': entry_price,
                                         'exit_price': np.NaN, 'quantity': position_size, 'fee': total_fee,
                                         'capital': '', 'mae': '', 'quantity_active': margin_total, 'profit_loss': ''}])

                df_trades = df_trades.append(dfm_aux)
                df_trades.reset_index(inplace=True, drop=True)
                continue

        # EXIT LONG STOPLOSS CONDITION
        if condition:

            if df_data.low[i] <= stoploss:

                if i + 1 < len(df_data):
                    exit_price = stoploss - object_variables["tick_size"]
                    exit_position_size_usd = position_size * object_variables["contract_size"] * exit_price
                    profit_loss = exit_position_size_usd - entry_position_size_usd - total_fee
                    exit_date = df_data.date[i]
                    capital += profit_loss
                    condition = False
                    capital_line.append(capital)

                    df_trades.at[df_trades.index[-1], 'mae'] = 0
                    df_trades.at[df_trades.index[-1], 'exit_date'] = str(exit_date)
                    df_trades.at[df_trades.index[-1], 'exit_price'] = exit_price
                    df_trades.at[df_trades.index[-1], 'capital'] = capital_line[-1]
                    df_trades.at[df_trades.index[-1], 'profit_loss'] = profit_loss
                    df_trades.at[df_trades.index[-1], 'exit_type'] = 'StopLoss'
                    df_trades.at[df_trades.index[-1], 'trade_direction'] = 'Long'

                    print("---------- Trade details ----------")
                    print('exit_type: STOPLOSS')
                    print(f'entry_price: {entry_price}')
                    print(f'exit_price: {exit_price}')
                    print(f'position_size: {position_size}')
                    print(f'entry_position_size_usd: {entry_position_size_usd}')
                    print(f'exit_position_size_usd: {exit_position_size_usd}')
                    print(f'fee: {total_fee}')
                    print(f'profit_loss: {profit_loss}')
                    print(f'capital: {capital}')
                    print(Fore.GREEN + 'entry_date: ' + Fore.WHITE + f'{entry_date}')
                    print(Fore.RED + 'exit_date: ' + Fore.WHITE + f'{exit_date}\n')

                    df_trades.reset_index(inplace=True, drop=True)
                    continue

        # EXIT LONG CONDITION
        if condition:

            # if df_data.date[i - variable2] == entry_date:
            if df_data.close[i] >= tf:

                if i + 1 < len(df_data):
                    exit_price = df_data.open[i + 1] - object_variables["tick_size"]
                    exit_position_size_usd = (position_size * object_variables[
                        "contract_size"]) * exit_price
                    profit_loss = exit_position_size_usd - entry_position_size_usd - total_fee
                    exit_date = df_data.date[i]
                    capital += profit_loss
                    condition = False
                    capital_line.append(capital)

                    # MAE
                    data_aux = df_data[
                        (df_data['date'] >= f'{entry_date}') & (df_data['date'] <= f'{exit_date}')]
                    min_price = min(data_aux.low)
                    mae = (min_price / entry_price) - 1

                    df_trades.at[df_trades.index[-1], 'mae'] = mae
                    df_trades.at[df_trades.index[-1], 'exit_date'] = str(exit_date)
                    df_trades.at[df_trades.index[-1], 'exit_price'] = exit_price
                    df_trades.at[df_trades.index[-1], 'capital'] = capital_line[-1]
                    df_trades.at[df_trades.index[-1], 'profit_loss'] = profit_loss
                    df_trades.at[df_trades.index[-1], 'exit_type'] = 'ExitLong'
                    df_trades.at[df_trades.index[-1], 'trade_direction'] = 'Long'

                    print("---------- Trade details ----------")
                    print('exit_type: EXITLONG')
                    print(f'entry_price: {entry_price}')
                    print(f'exit_price: {exit_price}')
                    print(f'position_size: {position_size}')
                    print(f'entry_position_size_usd: {entry_position_size_usd}')
                    print(f'exit_position_size_usd: {exit_position_size_usd}')
                    print(f'fee: {total_fee}')
                    print(f'profit_loss: {profit_loss}')
                    print(f'capital: {capital}')
                    print(Fore.GREEN + 'entry_date: ' + Fore.WHITE + f'{entry_date}')
                    print(Fore.RED + 'exit_date: ' + Fore.WHITE + f'{exit_date}\n')

                    df_trades.reset_index(inplace=True, drop=True)
                    continue

    return df_trades, capital_line


def short_direction(df_data, object_variables, take_profit, zv_value,  stop_loss):
    capital = float(object_variables["capital"])
    capital_line = [capital]
    condition = False

    df_trades = pd.DataFrame()
    # max_periods = round(max(variable1, variable2, variable3))
    max_periods = 14

    for i in range(max_periods, len(df_data)):

        # ENTRY SHORT CONDITION
        if not condition and df_data['zv'][i] >= zv_value and df_data['close'][i] < df_data['open'][i]:

            if i + 1 < len(df_data):

                entry_price = df_data.open[i + 1] - object_variables["tick_size"]
                stoploss = entry_price + (entry_price * stop_loss)
                position_size = np.round(
                    (capital * 0.01) / (entry_price * stop_loss * object_variables["contract_size"]))
                margin_total = object_variables["margin"] * position_size

                if margin_total > capital:
                    position_size = np.floor(capital / object_variables["margin"])

                if position_size == 0:
                    break

                tf = entry_price - (entry_price * take_profit)

                entry_position_size_usd = (position_size * object_variables["contract_size"]) * entry_price
                total_fee = (object_variables["fee"] * 2) * position_size
                entry_date = df_data.date[i]

                condition = True

                dfm_aux = pd.DataFrame([{'entry_date': str(entry_date), 'exit_date': '', 'entry_price': entry_price,
                                         'exit_price': np.NaN, 'quantity': position_size, 'fee': total_fee,
                                         'capital': '', 'mae': '', 'quantity_active': margin_total, 'profit_loss': ''}])

                df_trades = df_trades.append(dfm_aux)
                df_trades.reset_index(inplace=True, drop=True)
                continue

        # EXIT SHORT STOPLOSS CONDITION
        if condition:

            if df_data.high[i] >= stoploss:

                if i + 1 < len(df_data):
                    exit_price = stoploss + object_variables["tick_size"]
                    exit_position_size_usd = position_size * object_variables["contract_size"] * exit_price
                    profit_loss = ((exit_position_size_usd - entry_position_size_usd) * -1) - total_fee
                    exit_date = df_data.date[i]
                    capital += profit_loss
                    condition = False
                    capital_line.append(capital)

                    df_trades.at[df_trades.index[-1], 'mae'] = 0
                    df_trades.at[df_trades.index[-1], 'exit_date'] = str(exit_date)
                    df_trades.at[df_trades.index[-1], 'exit_price'] = exit_price
                    df_trades.at[df_trades.index[-1], 'capital'] = capital_line[-1]
                    df_trades.at[df_trades.index[-1], 'profit_loss'] = profit_loss
                    df_trades.at[df_trades.index[-1], 'exit_type'] = 'StopLoss'
                    df_trades.at[df_trades.index[-1], 'trade_direction'] = 'Short'

                    # print("---------- Trade details ----------")
                    # print('exit_type: STOPLOSS')
                    # print(f'entry_price: {entry_price}')
                    # print(f'exit_price: {exit_price}')
                    # print(f'position_size: {position_size}')
                    # print(f'entry_position_size_usd: {entry_position_size_usd}')
                    # print(f'exit_position_size_usd: {exit_position_size_usd}')
                    # print(f'fee: {total_fee}')
                    # print(f'profit_loss: {profit_loss}')
                    # print(f'capital: {capital}')
                    # print(Fore.GREEN + 'entry_date: ' + Fore.WHITE + f'{entry_date}')
                    # print(Fore.RED + 'exit_date: ' + Fore.WHITE + f'{exit_date}\n')

                    df_trades.reset_index(inplace=True, drop=True)
                    continue

        # EXIT SHORT CONDITION
        if condition:

            # if df_data.date[i - variable2] == entry_date:
            if df_data.close[i] <= tf:

                if i + 1 < len(df_data):
                    exit_price = df_data.open[i + 1] + object_variables["tick_size"]
                    exit_position_size_usd = (position_size * object_variables[
                        "contract_size"]) * exit_price
                    profit_loss = ((exit_position_size_usd - entry_position_size_usd) * -1) - total_fee
                    exit_date = df_data.date[i]
                    capital += profit_loss
                    condition = False

                    capital_line.append(capital)

                    # MAE
                    data_aux = df_data[
                        (df_data['date'] >= f'{entry_date}') & (df_data['date'] <= f'{exit_date}')]
                    min_price = min(data_aux.low)
                    mae = (min_price / entry_price) - 1

                    df_trades.at[df_trades.index[-1], 'mae'] = mae
                    df_trades.at[df_trades.index[-1], 'exit_date'] = str(exit_date)
                    df_trades.at[df_trades.index[-1], 'exit_price'] = exit_price
                    df_trades.at[df_trades.index[-1], 'capital'] = capital_line[-1]
                    df_trades.at[df_trades.index[-1], 'profit_loss'] = profit_loss
                    df_trades.at[df_trades.index[-1], 'exit_type'] = 'ExitShort'
                    df_trades.at[df_trades.index[-1], 'trade_direction'] = 'Short'

                    # print("---------- Trade details ----------")
                    # print('exit_type: EXITSHORT')
                    # print(f'entry_price: {entry_price}')
                    # print(f'exit_price: {exit_price}')
                    # print(f'position_size: {position_size}')
                    # print(f'entry_position_size_usd: {entry_position_size_usd}')
                    # print(f'exit_position_size_usd: {exit_position_size_usd}')
                    # print(f'fee: {total_fee}')
                    # print(f'profit_loss: {profit_loss}')
                    # print(f'capital: {capital}')
                    # print(Fore.GREEN + 'entry_date: ' + Fore.WHITE + f'{entry_date}')
                    # print(Fore.RED + 'exit_date: ' + Fore.WHITE + f'{exit_date}\n')

                    df_trades.reset_index(inplace=True, drop=True)
                    continue

    return df_trades, capital_line


def strategy(data: pd.DataFrame, parameters: list, object_variables):

    variable1 = float(parameters[0])  # take profit
    variable2 = float(parameters[1])  # zv_value
    variable3 = float(parameters[2])  # periods
    variable4 = float(parameters[3])  # stop_loss

    df_data = data
    df_data = Sdf.retype(df_data)
    df_data.reset_index(inplace=True)
    df_data.date = pd.to_datetime(df_data.date)

    def standarize(df_data, column='gap', periods=0):
        if periods == 0:
            df_data['z_' + column] = (df_data[column] - df_data[column].expanding().mean()) / df_data[
                column].expanding().std()

        if periods > 0:
            df_data['z_' + column] = (df_data[column] - df_data[column].rolling(periods).mean()) / df_data[
                column].rolling(periods).std()
        return df_data

    df_data = standarize(df_data, 'volume', int(variable3))

    df_data.dropna(inplace=True)
    df_data.reset_index(inplace=True)

    if object_variables["direction"] == 'Long':
        df_trades, capital_line = long_direction(df_data, object_variables, variable1, variable2, variable4)

    elif object_variables["direction"] == 'Short':
        df_trades, capital_line = short_direction(df_data, object_variables, variable1, variable2, variable4)

    if len(df_trades) > 1:
        df_trades.reset_index(inplace=True, drop=True)
        df_trades['type_trade'] = 'Backtesting'
        df_trades = df_trades[np.isfinite(df_trades['exit_price'])]
    else:
        df_trades = pd.DataFrame()

    return df_trades, capital_line


def get_stats(trades: pd.DataFrame(), capital_line: list):
    df_trades = trades[np.isfinite(trades['exit_price'])]

    try:
        diff_capital = abs(trades["capital"].diff())
        del diff_capital[0]
        noise = abs(trades["capital"].iloc[-1] - trades["capital"].iloc[0]) / sum(diff_capital)
    except:
        noise = 0

    df_aux = pd.DataFrame()
    df_aux["capital"] = capital_line
    df_aux["exit_date"] = pd.to_datetime(trades["exit_date"].shift(1))
    df_aux["exit_date"].iloc[0] = df_aux["exit_date"].iloc[1] - pd.DateOffset(1)
    df_aux.dropna(inplace=True)

    df_aux.set_index('exit_date', inplace=True)

    drawdown_series = to_drawdown_series(df_aux)
    dd_details = drawdown_details(drawdown_series)
    dd_details.reset_index(inplace=True, drop=True)
    dd_details.sort_values(by=[('capital', 'max drawdown')], ascending=True, inplace=True)

    try:
        max_dd = (dd_details["capital"]["max drawdown"].iloc[0] / 100).astype(float) * -1
        drawdown_recovery = (dd_details["capital"]["days"].iloc[0]).astype(float)
    except:
        max_dd = 0
        drawdown_recovery = 0

    cumulative_return = (df_aux["capital"].iloc[-1] / df_aux["capital"].iloc[0]) - 1

    risk_return = cumulative_return / max_dd if max_dd != 0 else 0

    num_trades = len(trades["capital"])

    result_trade = df_trades['profit_loss'].apply(lambda x: 1 if x > 0 else 0).sum()
    percent_profitable = result_trade / num_trades
    percent_loss = 1 - percent_profitable

    avg_profit = df_trades['profit_loss'][df_trades['profit_loss'] > 0].mean()
    avg_loss = df_trades['profit_loss'][df_trades['profit_loss'] < 0].mean()
    profit_factor = avg_profit * percent_profitable / (-1 * avg_loss * percent_loss)

    percent_strategy_exit = 0
    percent_stoploss_exit = 0

    for i in range(0, len(df_trades)):
        if df_trades.exit_type[i] == 'ExitLong' or df_trades.exit_type[i] == 'ExitShort':
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

    trades_consectv = (df_trades['profit_loss'].apply(lambda x: 1 if x > 0 else -1))
    trades_consectv = [(k, sum(1 for k in g)) for k, g in itertools.groupby(trades_consectv)]
    r_trades_aux = []

    for i in trades_consectv:
        r_trades_aux.append(i[0] * i[1])

    max_consectv_wins = max(r_trades_aux)
    max_consectv_loss = min(r_trades_aux) * -1

    df_trades.exit_date.iloc[-1] = pd.to_datetime(df_trades.exit_date.iloc[-1])
    df_trades.entry_date[0] = pd.to_datetime(df_trades.entry_date[0])

    long_in_days = (df_trades.exit_date.iloc[-1] - df_trades.entry_date[0]) / np.timedelta64(1, 'D')

    cumulative_return = -0.99 if cumulative_return < -1 else cumulative_return

    annual_return = -1 + (cumulative_return + 1) ** (365 / long_in_days)

    results_isi_par_row = pd.DataFrame({
        'cumulative_return': [cumulative_return],
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
        'max_consectv_loss': max_consectv_loss,
        'annual_return': annual_return,
        'noise': noise,
        "drawdown_recovery": drawdown_recovery
    })

    del results_isi_par_row["drawdown_recovery"]
    del results_isi_par_row["noise"]

    return results_isi_par_row


def get_scores(stats):

    min_risk_return = stats.risk_return.min()
    diff_risk_return = stats.risk_return.max() - min_risk_return

    min_cumulative_return = stats.cumulative_return.min()
    diff_cumulative_return = stats.cumulative_return.max() - min_cumulative_return

    min_profit_factor = stats.profit_factor.min()
    diff_profit_factor = stats.profit_factor.max() - min_profit_factor

    max_max_drawdown = stats.max_drawdown.max()
    diff_max_drawdown = max_max_drawdown - stats.max_drawdown.min()

    scores = [((0.4 * (r.risk_return - min_risk_return) / diff_risk_return) +
               (0.2 * (r.cumulative_return - min_cumulative_return) / diff_cumulative_return) +
               (0.2 * (r.profit_factor - min_profit_factor) / diff_profit_factor) +
               (0.2 * (max_max_drawdown - r.max_drawdown) / diff_max_drawdown)) for r in stats.itertuples()]

    return scores


def optimization(parameter_combinations, data_resampled):

    is_stats = pd.DataFrame()

    for parameter in parameter_combinations:

        trades, capital_line = strategy(data_resampled, parameter, object_variables)

        if not trades.empty:
            statistics = get_stats(trades, capital_line)
            statistics["parameters"] = str(parameter)

            is_stats = pd.concat([is_stats, statistics])
        else:
            print(f"Trades empty for the combination {parameter} please check your strategy")

    if len(is_stats) > 1:
        is_stats['score'] = get_scores(is_stats)
        is_stats.sort_values(by='score', ascending=False, inplace=True)
        is_stats.reset_index(inplace=True, drop=True)

        is_stats.to_csv(f'{object_variables["strategy_name"]}_{object_variables["direction"]}_'
                        f'{object_variables["temporality"]}.csv')

    else:
        is_stats['score'] = 1
        is_stats.to_csv(f'{object_variables["strategy_name"]}_{object_variables["direction"]}_'
                        f'{object_variables["temporality"]}.csv')
        graphic_validation(trades, data_resampled)


def graphic_validation(trades, data_resampled):

    # Candlestick
    inc = data_resampled.close > data_resampled.open
    dec = data_resampled.open > data_resampled.close
    w = 17 * 30 * 30 * 25  # For 2H

    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    source = ColumnDataSource(trades)

    p_hover_entry = HoverTool(
        names=["entry_condition"],

        tooltips=[
            ("entry_date", "@entry_date{%Y-%m-%d %H:%m:%S}"),
            ("entry_price", "@entry_price"),
            ("type", "Entry")
        ],

        formatters={
            'entry_date': 'datetime',
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    )

    p_hover_exit = HoverTool(
        names=["exit_condition"],

        tooltips=[
            ("exit_date", "@exit_date{%Y-%m-%d %H:%m:%S}"),
            ("exit_price", "@exit_price"),
            ("type", "@exit_type")
        ],

        formatters={
            'exit_date': 'datetime',
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    )

    crosshair = CrosshairTool(dimensions='both')

    # Figures
    p = figure(x_axis_type="datetime", plot_height=500, plot_width=1000, title="PRICES")

    p.segment(data_resampled.date, data_resampled.high, data_resampled.date, data_resampled.low, color="black")
    p.vbar(data_resampled.date[inc], w, data_resampled.open[inc], data_resampled.close[inc], fill_color="green",
           line_color="black")
    p.vbar(data_resampled.date[dec], w, data_resampled.open[dec], data_resampled.close[dec], fill_color="red",
           line_color="black")

    # Tools
    p.add_tools(p_hover_entry, p_hover_exit, crosshair)

    # Axis
    p.xaxis.axis_label = 'TIME'
    p.yaxis.axis_label = 'PRICE'

    p.circle('entry_date', 'entry_price', fill_color="green", line_color="black", legend='ENTRY CONDITION',
             size=12, fill_alpha=0.8, source=source, name='entry_condition')

    p.circle('exit_date', 'exit_price', fill_color="red", line_color="black", legend='EXIT CONDITION',
             size=12, fill_alpha=0.8, source=source, name='exit_condition')

    g = gridplot([[p]], sizing_mode='scale_width')
    show(g)


if __name__ == '__main__':

    stoploss_values = object_variables["parameters"]["stop_loss"]
    del object_variables["parameters"]["stop_loss"]
    object_variables["parameters"]["stop_loss"] = stoploss_values
    del stoploss_values

    aux = object_variables["parameters"].values()
    parameter_combinations = list(itertools.product(*aux))

    data = get_data(object_variables["symbols_1"])

    data_half_length = len(data) // 2
    data_half = data.date[data_half_length]
    is_data = data[(data['date'] >= f'{data.date[0]}') & (data['date'] <= f'{data_half}')]
    # is_data = is_data[:50000]
    data_resampled = resample_data(is_data, object_variables["temporality"])

    optimization(parameter_combinations, data_resampled)



class Streak(bt.ind.PeriodN):
    '''
    Keeps a counter of the current upwards/downwards/neutral streak
    '''
    lines = ('streak',)
    params = dict(period=2)  # need prev/cur days (2) for comparisons

    curstreak = 0

    def next(self):
        d0, d1 = self.data[0], self.data[-1]

        if d0 > d1:
            self.l.streak[0] = self.curstreak = max(1, self.curstreak + 1)
        elif d0 < d1:
            self.l.streak[0] = self.curstreak = min(-1, self.curstreak - 1)
        else:
            self.l.streak[0] = self.curstreak = 0