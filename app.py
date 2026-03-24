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
from models import MonteCarloPricingEngine, FiniteDifferencePricingEngine

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
            html.P("Stochastic Volatility, PDE Solvers & Measure Change (Girsanov)", className="text-center text-muted mb-4"),
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

                    html.Label("Risk-Free Rate (r) - Measure Q"),
                    dcc.Slider(id='r-slider', min=0.0, max=0.15, value=LIVE_R, step=0.01, 
                               marks={i/100: f'{i}%' for i in range(0, 16, 5)}, className="mb-3"),

                    # --- NEW: Historical Drift Slider for Measure P ---
                    html.Label("Historical Drift (μ) - Measure P", className="text-success fw-bold"),
                    dcc.Slider(id='mu-slider', min=-0.1, max=0.25, value=0.08, step=0.01, 
                               marks={i/100: f'{i}%' for i in range(-10, 26, 10)}, className="mb-3"),

                    html.Hr(),
                    
                    # --- UPGRADED PRICING OUTPUT PANEL ---
                    html.Div([
                        html.H6("Feynman-Kac Convergence", className="text-center mt-3 mb-2 text-muted"),
                        dbc.Row([
                            dbc.Col([
                                html.Div("Monte Carlo", className="text-muted small"),
                                html.H5(id='mc-price-output', className="text-primary mb-0")
                            ], width=6, className="text-center border-end"),
                            dbc.Col([
                                html.Div("PDE (Implicit FDM)", className="text-muted small"),
                                html.H5(id='fdm-price-output', className="text-info mb-0")
                            ], width=6, className="text-center")
                        ], className="p-2 border rounded bg-light")
                    ], className="mt-3")
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
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Vega (ν)", className="card-subtitle text-muted"),
                        html.H3(id="vega-val", className="text-warning")
                    ])
                ]), width=4),
                # --- NEW: Market Price of Risk Card ---
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Market Price of Risk (θ)", className="card-subtitle text-muted"),
                        html.H3(id="theta-val", className="text-success")
                    ])
                ]), width=4),
            ], className="mt-3")
        ], width=9)
    ])
], fluid=True)

@app.callback(
    [Output('monte-carlo-graph', 'figure'),
     Output('mc-price-output', 'children'),
     Output('fdm-price-output', 'children'),
     Output('delta-val', 'children'),
     Output('vega-val', 'children'),
     Output('theta-val', 'children')],
    [Input('s0-slider', 'value'),
     Input('k-slider', 'value'),
     Input('t-slider', 'value'),
     Input('sigma-slider', 'value'),
     Input('r-slider', 'value'),
     Input('mu-slider', 'value'),
     Input('option-type', 'value')]
)
def update_dashboard(S0, K, T, sigma, r, mu, option_type):
    # 1. Q-Measure Engine (Risk-Neutral, driven by r) - Used for Pricing
    mc_engine_Q = MonteCarloPricingEngine(S0=S0, K=K, T=T, r=r, sigma=sigma, num_simulations=500)
    fdm_engine = FiniteDifferencePricingEngine(S0=S0, K=K, T=T, r=r, sigma=sigma)
    
    # 2. P-Measure Engine (Real-World, driven by mu) - Used for Visualization
    mc_engine_P = MonteCarloPricingEngine(S0=S0, K=K, T=T, r=mu, sigma=sigma, num_simulations=500)
    
    # 3. Calculations
    # Turn ON the Martingale Control Variate for the Q-Measure (where discounted price is a martingale)
    mc_price, paths_Q = mc_engine_Q.price_european_option(option_type=option_type, use_control_variate=True)
    # We only need P-Measure for the visual paths, so Control Variate is False (and mathematically invalid under P anyway)
    _, paths_P = mc_engine_P.price_european_option(option_type=option_type, use_control_variate=False)
    fdm_price = fdm_engine.price_european_option(option_type=option_type)
    greeks = mc_engine_Q.calculate_greeks(option_type=option_type)
    
    # Calculate Girsanov's Theta
    theta = (mu - r) / sigma

    # --- PLOTLY FIGURE LOGIC ---
    time_steps = np.linspace(0, T, mc_engine_Q.num_steps + 1)
    fig = go.Figure()
    
    # Plot a subset of P-Measure paths (Green)
    for i in range(min(50, paths_P.shape[1])):
        fig.add_trace(go.Scatter(
            x=time_steps, y=paths_P[:, i], 
            mode='lines', line=dict(width=1, color='rgba(46, 204, 113, 0.1)'),
            showlegend=False, hoverinfo='skip'
        ))
        
    # Plot a subset of Q-Measure paths (Blue)
    for i in range(min(50, paths_Q.shape[1])):
        fig.add_trace(go.Scatter(
            x=time_steps, y=paths_Q[:, i], 
            mode='lines', line=dict(width=1, color='rgba(52, 152, 219, 0.15)'),
            showlegend=False, hoverinfo='skip'
        ))

    # Add Mean Paths to clearly show the drift difference
    fig.add_trace(go.Scatter(
        x=time_steps, y=np.mean(paths_P, axis=1), 
        mode='lines', name='Expected Path (Measure P)',
        line=dict(color='#27AE60', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_steps, y=np.mean(paths_Q, axis=1), 
        mode='lines', name='Expected Path (Measure Q)',
        line=dict(color='#2980B9', width=3)
    ))
        
    # Add Strike Price line
    fig.add_trace(go.Scatter(
        x=[0, T], y=[K, K], 
        mode='lines', name=f'Strike (K={K})',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Girsanov Measure Change: Real-World (P) vs. Risk-Neutral (Q) Paths',
        xaxis_title='Years to Maturity', yaxis_title='Asset Price (€)',
        template='plotly_white', margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Format output strings
    mc_text = f"€{mc_price:.2f}"
    fdm_text = f"€{fdm_price:.2f}"
    delta_text = f"{greeks['Delta']:.3f}"
    vega_text = f"{greeks['Vega']:.3f}"
    theta_text = f"{theta:.3f}"
    
    return fig, mc_text, fdm_text, delta_text, vega_text, theta_text

if __name__ == '__main__':
    app.run(debug=True)