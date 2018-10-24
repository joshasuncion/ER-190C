def interact_plot(basis_k, models, poly_phi, X, Y):

    import warnings
    warnings.filterwarnings('ignore')

    import plotly.offline as py
    py.init_notebook_mode(connected=False)

    from IPython.core.display import display, HTML
    # The polling here is to ensure that plotly.js has already been loaded before
    # setting display alignment in order to avoid a race condition.
    display(HTML(
        '<script>'
            'var waitForPlotly = setInterval( function() {'
                'if( typeof(window.Plotly) !== "undefined" ){'
                    'MathJax.Hub.Config({ SVG: { font: "STIX-Web" }, displayAlign: "center" });'
                    'MathJax.Hub.Queue(["setRenderer", MathJax.Hub, "SVG"]);'
                    'clearInterval(waitForPlotly);'
                '}}, 5000 );'
        '</script>'
    ))

    import plotly.graph_objs as go
    import plotly.figure_factory as ff
    import cufflinks as cf

    cf.set_config_file(offline=False, world_readable=True, theme='ggplot')

    import numpy as np
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.4)

    old_points = go.Scatter(name="Old Data", x=X_train, y=y_train, mode='markers',  
                        marker=dict(color="blue", symbol="circle"))
    new_points = go.Scatter(name="New Data", x=X_test, y=y_test, mode='markers', 
                        marker=dict(color="red", symbol="diamond"))

    # Make the x values where plot points will be generated
    X_plt = np.linspace(np.min(X)-1, np.max(X)+1, 200)

    # Generate the Plotly line objects by predicting the value at each X_plt
    lines = []
    for k in sorted(models.keys()):
        ytmp = models[k].predict(poly_phi(k)(X_plt))
        # Plotting software fails with large numbers
        ytmp[ytmp > 500] = 500
        ytmp[ytmp < -500] = -500
        lines.append(
            go.Scatter(name="degree "+ str(k), x=X_plt,
                       y = ytmp,visible=False))

    # Construct steps for the interactive slider
    steps = []
    data = [old_points, new_points]
    data_visibility = [True, True]
    lines[0].visible=True
    for i, line in enumerate(lines):
        line_visibility = [False] * len(lines)
        line_visibility[i] = True  # Toggle i'th trace to "visible"
        step = dict(
            label = line['name'],
            method = 'restyle',
            args = ['visible', data_visibility + line_visibility],
        )
        steps.append(step)

    # Build the slider object
    sliders = [dict(active = 0, pad = {"t": 20}, steps = steps)]

    # render the plot
    layout = go.Layout(xaxis=dict(range=[np.min(X_plt), np.max(X_plt)]), 
                       yaxis=dict(range=[np.min(Y) -5 , np.max(Y)+5]),
                       sliders=sliders)
    py.iplot(go.Figure(data = data + lines, layout=layout))