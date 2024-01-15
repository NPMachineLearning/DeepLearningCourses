import numpy as np
import plotly.graph_objs as go
import pandas as pd

z = np.zeros([100, 100])
for x in range(100):
    for y in range(100):
        z[x, y] = ( 0.0006 * x ** 6 - 0.005 * y ** 6 + 0.5 * x ** 5 - 0.1 * y ** 5 + 0.005 * x ** 4 + 0.003 * y ** 4) / 10000000

trace = go.Surface(z=z)
layout = go.Layout(title="3D", autosize=True, margin=dict(l=50, r=50, b=50, t=50))
figure = go.Figure(data=[trace], layout=layout)
figure.show()

# 多元多皆方程式
# f(x_{1},x_{2},x_{3},x_{4},x_{5})=ax_{1}^5+bx_{2}^5+cx_{3}^5+dx_{4}^2+ex_{5}^3+.....
