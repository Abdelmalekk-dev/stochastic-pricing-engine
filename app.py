import sys
import os
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Ensure the models directory can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.engine import MonteCarloPricingEngine

# Initialize the Dash app
app = dash.Dash(__name__, title="Quantitative Pricing Engine")

# Define the layout of the dashboard
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("European Options Pricing: Monte Carlo Simulation", style={'textAlign': 'center', 'color': '#2C3E50'}),
    html.P("Adjust the market parameters below to dynamically simulate Geometric Brownian Motion paths and price the option.", 
           style={'textAlign': 'center', 'color': '#7F8C8D'}),
    
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'marginTop': '30px'}, children=[
        
        # Left Control Panel
        html.Div(style={'width': '25%', 'padding': '20px', 'backgroundColor': '#F8F9FA', 'borderRadius': '10px'}, children=[
            html.H3("Market Parameters"),
            
            html.Label("Initial Stock Price (S0)"),
            dcc.Slider(id='s0-slider', min=50, max=200, step=1, value=100, 
                       marks={i: f'€{i}' for i in range(50, 201, 50)}),
            html.Br(),
            
            html.Label("Strike Price (K)"),
            dcc.Slider(id='k-slider', min=50, max=200, step=1, value=105, 
                       marks={i: f'€{i}' for i in range(50, 201, 50)}),
            html.Br(),
            
            html.Label("Time to Maturity in Years (T)"),
            dcc.Slider(id='t-slider', min=0.1, max=5.0, step=0.1, value=1.0, 
                       marks={i: f'{i}Y' for i in range(1, 6)}),
            html.Br(),
            
            html.Label("Volatility (\u03c3)"), # \u03c3 is the unicode for sigma
            dcc.Slider(id='sigma-slider', min=0.05, max=1.0, step=0.05, value=0.2, 
                       marks={i/10: f'{int(i*10)}%' for i in range(1, 11, 2)}),
            html.Br(),
            
            html.Label("Risk-Free Rate (r)"),
            dcc.Slider(id='r-slider', min=0.0, max=0.15, step=0.01, value=0.05, 
                       marks={i/100: f'{i}%' for i in range(0, 16, 5)}),
            html.Br(),
            
            html.Div(id='price-output', style={
                'marginTop': '30px', 'padding': '20px', 'backgroundColor': '#E8F6F3', 
                'borderRadius': '8px', 'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold', 'color': '#117A65'
            })
        ]),
        
        # Right Graph Panel
        html.Div(style={'width': '75%', 'paddingLeft': '20px'}, children=[
            dcc.Graph(id='monte-carlo-graph', style={'height': '70vh'})
        ])
    ])
])

# Define the callback to update the graph and price based on slider inputs
@app.callback(
    [Output('monte-carlo-graph', 'figure'),
     Output('price-output', 'children')],
    [Input('s0-slider', 'value'),
     Input('k-slider', 'value'),
     Input('t-slider', 'value'),
     Input('sigma-slider', 'value'),
     Input('r-slider', 'value')]
)
def update_dashboard(S0, K, T, sigma, r):
    # Instantiate the engine
    # We use 250 simulations for the dashboard to keep the web rendering fast and responsive
    engine = MonteCarloPricingEngine(S0=S0, K=K, T=T, r=r, sigma=sigma, num_simulations=250)
    
    # Run the simulation
    call_price, paths = engine.price_european_call()
    
    # Create the Plotly figure
    time_steps = np.linspace(0, T, engine.num_steps + 1)
    fig = go.Figure()
    
    # Add a sample of the simulated paths to the graph
    for i in range(min(100, paths.shape[1])):  # Plot up to 100 paths to avoid browser lag
        fig.add_trace(go.Scatter(
            x=time_steps, 
            y=paths[:, i], 
            mode='lines', 
            line=dict(width=1, color='rgba(52, 152, 219, 0.2)'),
            showlegend=False
        ))
        
    # Add a horizontal line for the Strike Price
    fig.add_trace(go.Scatter(
        x=[0, T], 
        y=[K, K], 
        mode='lines', 
        name=f'Strike Price (K={K})',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Format the layout
    fig.update_layout(
        title='Simulated Asset Paths (Geometric Brownian Motion)',
        xaxis_title='Time to Maturity (Years)',
        yaxis_title='Asset Price (€)',
        template='plotly_white',
        hovermode=False,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Format the output text
    price_text = f"Estimated Call Price: €{call_price:.2f}"
    
    return fig, price_text

if __name__ == '__main__':
    # Run the Dash server
    app.run(debug=True)