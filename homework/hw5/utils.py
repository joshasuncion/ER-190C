import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import plotly.offline as py
py.init_notebook_mode(connected=False)
import plotly.graph_objs as go

def run_plotly(observed_data, predicted_data, total_data, hour, k):
    
    # Access token for mapbox
    access_token='pk.eyJ1IjoianNqaWFuZyIsImEiOiJjamswaHVkenAwNnp6M3RyeWlkZjRmeHFrIn0.tfl-yxZeQ1M2UQmE4rnSOA'

    predicted = go.Scattermapbox(
        name = 'Predicted',
        lon = predicted_data['Longitude'],
        lat = predicted_data['Latitude'],
        mode = 'markers',
        marker=dict(
            size=8,
            opacity=0.8,
            cmin = 0,
            color = predicted_data['Log Sample Measurement'],
            cmax = total_data['Log Sample Measurement'].max(),
            colorscale = [
                [0, 'rgb(34,139,34)'],
                [0.5, 'rgb(204,204,0)'],
                [1, 'rgb(178,34,34)']
            ],
            showscale=True,
            symbol="circle",
            colorbar=dict(
                tickmode='array',
                tickvals=np.arange(0, 7),
                ticktext=[str(x) + ' LC' for x in np.round(np.e**np.arange(0, 7), 1)]
            )
        ),
        text = predicted_data['Text'],
        hoverinfo='text'
    )
    
    observed = go.Scattermapbox(
        name = 'Observed',
        lon = observed_data['Longitude'],
        lat = observed_data['Latitude'],
        mode = 'markers',
        marker=dict(
            size=8,
            opacity=0.8,
            cmin = 0,
            color = observed_data['Log Sample Measurement'],
            cmax = total_data['Log Sample Measurement'].max(),
            colorscale = [
                [0, 'rgb(34,139,34)'],
                [0.5, 'rgb(204,204,0)'],
                [1, 'rgb(178,34,34)']
            ],
            showscale=True,
            symbol="circle",
            colorbar=dict(
                tickmode='array',
                tickvals=np.arange(0, 7),
                ticktext=[str(x) + ' LC' for x in np.round(np.e**np.arange(0, 7), 1)]
            )
        ),
        text = observed_data['Text'],
        hoverinfo='text'
    )
    
    data = [predicted, observed]

    layout = go.Layout(
            title='PM2.5 Concentration on Oct 13, 2017 at ' + hour + ' with K = ' + str(k),
            autosize=True,
            hovermode='closest',
            mapbox=dict(
                accesstoken=access_token,
                bearing = 0,
                center=dict(
                    lat=37.387526,
                    lon=-121.784217
                ),
                pitch=0,
                zoom=5
            ),
            legend=dict(
                x=0,
                y=1,
                traceorder='normal',
                font=dict(
                    size=12,
                    color='#000'
                ),
                bgcolor='#E2E2E2',
                bordercolor='#FFFFFF',
                borderwidth=2
            )
    )

    fig = dict(data=data, layout=layout)
    return py.iplot(fig)
