import pandas as pd
import numpy as np

df = pd.DataFrame({"close": np.random.normal(65, 35, 100)})
df['streak'] = 0

def standarize(df, column, periods=0):
    if periods == 0:
        df['z_' + column] = (df[column] - df[column].expanding().mean()) / df[column].expanding().std()

    if periods > 0:
        df['z_' + column] = (df[column] - df[column].rolling(periods).mean()) / df[column].rolling(periods).std()
    return df

# Streak
cur_streak = 0
for i in range(len(df)):
    if i == 0:
        continue

    if df.close[i] > df.close[i - 1]:
        cur_streak = max(0, cur_streak)
        cur_streak += 1
        df['streak'].iloc[i] = cur_streak
    elif df.close[i] < df.close[i - 1]:
        cur_streak = min(0, cur_streak)
        cur_streak += -1
        df['streak'].iloc[i] = cur_streak
    else:
        cur_streak = 0
        df['streak'].iloc[i] = cur_streak


# print(df)


# Standarization

df = standarize(df, column='streak', periods=0)

df.dropna(inplace=True)
df.reset_index(inplace=True)

print(df)
