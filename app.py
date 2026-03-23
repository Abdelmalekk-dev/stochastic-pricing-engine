import sys
import os
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import yfinance as yf

# Ensure the models directory can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models import MonteCarloPricingEngine

# --- LIVE DATA FETCHING ---
def get_risk_free_rate():
    """Fetches the current 10Y German Bund yield as the risk-free rate proxy."""
    try:
        bund = yf.Ticker("^GDBR10")
        hist = bund.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1] / 100
    except Exception:
        pass
    return 0.03  # Fallback to 3%

LIVE_R = get_risk_free_rate()

# --- APP INITIALIZATION ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Quantitative Pricing Engine")

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Options Pricing & Risk Analytics", className="text-center mt-4 mb-2"),
            html.P("Stochastic Volatility & Monte Carlo Simulation Engine", className="text-center text-muted mb-4"),
        ], width=12)
    ]),

    dbc.Row([
        # Left Sidebar: Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Market Parameters", className="mb-0")),
                dbc.CardBody([
                    html.Label("Option Type"),
                    dcc.RadioItems(
                        id='option-type',
                        options=[{'label': ' Call ', 'value': 'call'}, {'label': ' Put ', 'value': 'put'}],
                        value='call',
                        className="mb-3",
                        labelStyle={'display': 'inline-block', 'marginRight': '15px'}
                    ),

                    html.Label("Initial Price (S0)"),
                    dcc.Slider(id='s0-slider', min=50, max=200, value=100, step=1, 
                               marks={i: f'€{i}' for i in range(50, 201, 50)}, className="mb-3"),
                    
                    html.Label("Strike Price (K)"),
                    dcc.Slider(id='k-slider', min=50, max=200, value=105, step=1, 
                               marks={i: f'€{i}' for i in range(50, 201, 50)}, className="mb-3"),
                    
                    html.Label("Time to Maturity (Years)"),
                    dcc.Slider(id='t-slider', min=0.1, max=5.0, value=1.0, step=0.1, 
                               marks={i: f'{i}Y' for i in range(1, 6)}, className="mb-3"),

                    html.Label("Volatility (σ)"),
                    dcc.Slider(id='sigma-slider', min=0.05, max=1.0, value=0.2, step=0.05, 
                               marks={i/10: f'{int(i*10)}%' for i in range(2, 11, 2)}, className="mb-3"),

                    html.Label("Risk-Free Rate (r)"),
                    dcc.Slider(id='r-slider', min=0.0, max=0.15, value=LIVE_R, step=0.01, 
                               marks={i/100: f'{i}%' for i in range(0, 16, 5)}, className="mb-3"),

                    html.Hr(),
                    html.Div([
                        html.P("Live Market Proxy:", className="mb-0 text-muted small"),
                        dbc.Badge(f"10Y Bund: {LIVE_R*100:.2f}%", color="success", className="p-2 w-100", id="r-badge")
                    ]),
                    
                    html.Div(id='price-output', className="mt-4 p-3 border rounded bg-light text-center")
                ])
            ], className="shadow-sm")
        ], width=3),

        # Right Panel: Visualization & Risk Metrics
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='monte-carlo-graph', style={'height': '60vh'}), width=12)
            ]),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Delta (Δ)", className="card-subtitle text-muted"),
                        html.H3(id="delta-val", className="text-primary")
                    ])
                ]), width=6),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Vega (ν)", className="card-subtitle text-muted"),
                        html.H3(id="vega-val", className="text-warning")
                    ])
                ]), width=6),
            ], className="mt-3")
        ], width=9)
    ])
], fluid=True)

@app.callback(
    [Output('monte-carlo-graph', 'figure'),
     Output('price-output', 'children'),
     Output('delta-val', 'children'),
     Output('vega-val', 'children')],
    [Input('s0-slider', 'value'),
     Input('k-slider', 'value'),
     Input('t-slider', 'value'),
     Input('sigma-slider', 'value'),
     Input('r-slider', 'value'),
     Input('option-type', 'value')]
)
def update_dashboard(S0, K, T, sigma, r, option_type):
    # Instantiate the engine
    engine = MonteCarloPricingEngine(S0=S0, K=K, T=T, r=r, sigma=sigma, num_simulations=500)
    
    # Run the simulation and calculate Greeks
    price, paths = engine.price_european_option(option_type=option_type)
    greeks = engine.calculate_greeks(option_type=option_type)
    
    # --- PLOTLY FIGURE LOGIC ---
    time_steps = np.linspace(0, T, engine.num_steps + 1)
    fig = go.Figure()
    
    # Plot a subset of paths for performance
    for i in range(min(100, paths.shape[1])):
        fig.add_trace(go.Scatter(
            x=time_steps, y=paths[:, i], 
            mode='lines', line=dict(width=1, color='rgba(52, 152, 219, 0.15)'),
            showlegend=False
        ))
        
    # Add Strike Price line
    fig.add_trace(go.Scatter(
        x=[0, T], y=[K, K], 
        mode='lines', name=f'Strike (K={K})',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'Simulated {option_type.capitalize()} Asset Paths',
        xaxis_title='Years to Maturity', yaxis_title='Asset Price (€)',
        template='plotly_white', margin=dict(l=40, r=40, t=40, b=40)
    )

    # Format output strings
    price_text = f"Estimated {option_type.capitalize()} Price: €{price:.2f}"
    delta_text = f"{greeks['Delta']:.3f}"
    vega_text = f"{greeks['Vega']:.3f}"
    
    return fig, price_text, delta_text, vega_text

if __name__ == '__main__':
    app.run(debug=True)