# %%
import numpy as np
import pandas as pd
from IPython.display import display
import plotly.graph_objects as go
import plotly.express as px
# %%
data = pd.DataFrame(
    {
        "bug": [4, 1, 13, 0, 3, 0, 2, 0],
        "test": [15, 25, 102, 18, 97, 26, 158, 131]
    }
)
display(data)
data = pd.DataFrame(
    {
        "Cumsum_bug": np.cumsum(data["bug"]),
        "Cumsum_test": np.cumsum(data["test"])
    }
)
display(data)
# %%


def gompertz(x: np.array, k=10, b=0.01, c=0.01) -> np.array:
    '''Gompertz
    '''
    return k * np.power(b, np.exp(-c * x))


def normal_eq(X: np.array, Y: np.array) -> np.array:
    '''正規方程式
    '''
    return np.linalg.pinv(X.T @ X) @ X.T @ Y  # 一般逆行列


def fit(t: np.array, input_y: np.array) -> dict:
    '''
    cの設定値と実際の値には誤差がある。
    その誤差でc設定値を補正し新たな回帰モデルを作成、これを繰り返す。
    '''
    c_init = 0.02
    c = c_init
    Y_i = np.log(input_y)

    i = 0
    while True:
        t_x = np.insert(
            np.vstack([np.exp(-c * t), t * np.exp(-c * t)]).T,
            0, 1,
            axis=1
        )
        alpha, beta, gamma = normal_eq(t_x, Y_i)
        delta = -gamma / beta
        c += delta
        i += 1
        if abs(delta) < 1e-15 or i >= 100:
            break

    print(f"c初期値 : {c_init} ->  {i}回目 : {c}")
    print(f"収束予測値 : {np.round(np.power(np.e, alpha))}")

    return {
        "k": np.power(np.e, alpha),
        "b": np.power(np.e, beta),
        "c": c
    }


# %%
parameter = fit(data["Cumsum_test"], data["Cumsum_bug"])
x_arr = np.arange(0, np.round(data["Cumsum_test"].max() * 1.2))
result = gompertz(x_arr, **parameter)
# %%
chart = [
    go.Scatter(
        x=x_arr, y=result,
        line={"dash": "solid", "width": 7, "color": "red"},
        name="予測線"
    ),
    go.Scatter(
        x=data["Cumsum_test"], y=data["Cumsum_bug"],
        line={"dash": "solid", "width": 5, "color": "blue"},
        name="テスト結果"
    ),
    go.Scatter(
        x=data["Cumsum_test"], y=data["Cumsum_bug"],
        mode="markers", marker={"size": 15, "symbol": "diamond"},
        name="",
    )
]
fig = go.Figure(chart)
fig.update_layout(
    title={
        "text": "<b>累積バグ数の予測</b>",
        "font": {
            "size": 22,
            "color": "grey"
        },
        "x": 0.5,
        "y": 0.9,
    },
    legend={
        "xanchor": "right",
        "yanchor": "bottom",
        "x": 0.95,
        "y": 0.05,
        "orientation": "h"
    },
    xaxis={
        "title": "テスト実施総数",
        "dtick": 100
    },
    yaxis={
        "title": "累積バグ数",
        "dtick": 2
    }
)
fig.update_layout(
    template="plotly_dark",
    autosize=False,
    width=800,
    height=500,
    margin={
        "l": 50,
        "r": 50,
        "t": 80,
        "b": 80,
        "pad": 4
    },
)
fig.show()
fig.write_image("output_image.svg")
# %%
