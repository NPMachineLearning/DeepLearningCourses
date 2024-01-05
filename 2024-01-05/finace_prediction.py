import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

display=pd.options.display
display.max_columns=None
display.max_rows=None
display.width=None
display.max_colwidth=None

stock = "2330.TW"
current = datetime.now()
df = yf.download(stock, "1970-01-01", current, auto_adjust=True)
ma1 = 5
ma2 = 10
df["s1"] = df["Close"].rolling(window=ma1).mean()
df["s2"] = df["Close"].rolling(window=ma2).mean()
df = df.dropna()
train = df[["Close", "s1", "s2"]]
train["next_day_price"] = train["Close"].shift(-1)
train = train.dropna()
print(train)

X_train = train[["s1", "s2"]]
y_train = train["next_day_price"]

model = LinearRegression()
model.fit(X_train, y_train)

df["predict_price"] = model.predict(df[["s1", "s2"]])
print(df)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"].values,
        mode="lines",
        name="Price",
        line=dict(color="royalblue", width=2)
    )
)

pred = df[["predict_price"]]
s = (pred.tail(1).index+timedelta(days=1))[0]
date = pd.date_range(s, periods=1)
pred.loc[date[0]] = [0]
pred["predict_price"] = pred["predict_price"].shift(1)

fig.add_trace(
    go.Scatter(
        x=pred.index,
        y=pred["predict_price"].values,
        mode="lines",
        name="prediction",
        line=dict(color="orange", width=2)
    )
)
fig.update_layout(
    dragmode="pan",
    title_text=f"{stock}",
    xaxis=go.layout.XAxis(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1 month", step="month", stepmode="backward"),
                dict(count=6, label="6 month", step="month", stepmode="backward"),
                dict(count=1, label="1 year", step="year", stepmode="backward"),
                dict(count=1, label="1 day", step="day", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True)
    )
)
fig.show()