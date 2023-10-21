import pandas as pd
import plotly.express as px


def plot_prediction(df: pd.DataFrame, title: str = "") -> None:
    # df - датасет с колонками таргет и предикт, а еще с колонкой datetime
    fig = px.line(df, x="datetime", y=["target", "predict"], title=title)
    fig.show()
