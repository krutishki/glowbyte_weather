# import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def plot_prediction(df, title = ""):
    # df - датасет с колонками таргет и предикт, а еще с колонкой datetime
    fig = px.line(df, x='datetime', y=['target', 'predict'], title = title)
    fig.show()