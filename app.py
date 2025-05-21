import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
import json
from arch import arch_model
from gbm_optimizer import gbm, optimize_gbm
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import requests
from yahooquery import Ticker
from sklearn.preprocessing import MinMaxScaler

# Default stocks list
default_stocks = ["CHWY", "IBIT", "ASO", "MARA", "ET", "DVN", "INTC", "SPLG", "TOST", "NBIS", "ON"]

st.set_page_config(page_title="Wheel Trading", layout="wide")

# Sidebar header
st.sidebar.title("Wheel Trading")
st.sidebar.markdown("Analyze covered option contracts for potential investment opportunities")

# Sidebar controls
analysis_mode = st.sidebar.radio("Analysis Mode", ["Cash Secured Puts Analysis", "Stock Trend Analysis", "Covered Calls Analysis"])

if analysis_mode == "Cash Secured Puts Analysis":
    st.title("Cash Secured Puts (CSP) Analyzer")
    
    # Date selector for expiration
    today = datetime.today()
    default_expiration = today + timedelta(days=30)
    expiration_date = st.sidebar.date_input(
        "Option Expiration Date", 
        value=default_expiration,
        min_value=today
    )
    
    # Stock selection
    st.sidebar.subheader("Stock Selection")

    custom_stocks = st.sidebar.text_input("Enter stock symbols (comma separated)", "SPY, QQQ")
    selected_stocks = [symbol.strip() for symbol in custom_stocks.split(",")]
    
    # Start analysis button
    start_analysis = st.sidebar.button("Analyze CSP Options")
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    simulation_attempts = st.sidebar.slider("Simulation Attempts", 100, 1000, 500)
    
    if start_analysis and selected_stocks:
        st.subheader(f"Analyzing CSP options for {', '.join(selected_stocks)} expiring on {expiration_date.strftime('%Y-%m-%d')}")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results container
        results_container = st.container()
        
        all_results = []
        
        for i, ticker in enumerate(selected_stocks):
            status_text.text(f"Analyzing {ticker}...")
            progress_bar.progress((i + 0.5) / len(selected_stocks))
            
            try:
                # Function to get option chain (similar to your notebook)
                def get_option_chain(symbol, option_type="puts", expiration_date=None):
                    try:
                        stock = Ticker(symbol)
                        df = stock.option_chain  # Fetch all available option chain data
                        
                        if df is None or df.empty:
                            st.warning(f"No option data available for {symbol}.")
                            return None
                        
                        # Filter by expiration date if provided
                        if expiration_date:
                            df = df.loc[df.index.get_level_values('expiration') == expiration_date.strftime("%Y-%m-%d")]
                        
                        # Filter by option type if provided ('calls' or 'puts')
                        if option_type in ['calls', 'puts']:
                            df = df.xs(option_type, level=2)
                        
                        # Reset index to remove multi-indexing
                        df = df.reset_index()
                        
                        # Rename 'expiration' to 'expiration_date'
                        df = df.rename(columns={'expiration': 'expiration_date'})
                        
                        # Calculate ROI as (bid + ask) / strike * 100
                        df['mid_price'] = (df['bid'] + df['ask']) / 2
                        df['ROI'] = (df['mid_price'] / df['strike']) * 100
                        
                        # Filter rows based on open interest, bid, in-the-money, and ROI
                        df = df[(df['openInterest'] >= 50) & 
                                (df['bid'] > 0.00) & 
                                (df['ROI'] > 0.1) &
                                (((df['ask'] - df['bid']) / df['bid']) * 100 < 40)
                                ] 
                        
                        return df
                    except Exception as e:
                        st.error(f"Error fetching option chain for {symbol}: {e}")
                        return None
                
                # Function to get current stock price
                def get_current_stock_price(symbol):
                    ticker = yf.Ticker(symbol)
                    todays_data = ticker.history(period='1d')
                    return todays_data['Close'].iloc[-1]
                
                # Function to fit GARCH model
                def fit_garch(symbol, period):
                    stock_data = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
                    real_prices = stock_data["Close"].dropna().values.flatten()
                    returns = np.diff(np.log(real_prices))
                    
                    # Fit GARCH(1,1) model
                    model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
                    garch_fit = model.fit(disp="off")
                    
                    conditional_volatilities = garch_fit.conditional_volatility
                    
                    N = len(conditional_volatilities)
                    weights = np.linspace(1, 2, N)  
                    
                    weighted_volatility = np.sum(weights * conditional_volatilities) / np.sum(weights)
                    
                    return weighted_volatility
                
                # Get option chain
                option_chain = get_option_chain(
                    ticker, 
                    option_type="puts", 
                    expiration_date=expiration_date
                )
                
                if option_chain is None or option_chain.empty:
                    status_text.text(f"No valid options found for {ticker}")
                    continue
                
                # Get current stock price
                price = get_current_stock_price(ticker)
                
                # Optimize GBM parameters
                optimizer_training_period = "6mo"
                bin_length = 18
                days_to_expiration = np.busday_count(
                    datetime.today().date(), 
                    expiration_date
                )
                
                with st.spinner(f"Optimizing parameters for {ticker}..."):
                    try:
                        optimized_mu, optimized_sigma = optimize_gbm(
                            symbol=ticker, 
                            training_period=optimizer_training_period, 
                            bin_length=bin_length
                        )
                        
                        # Apply GARCH for more accurate volatility
                        optimized_sigma = fit_garch(
                            symbol=ticker, 
                            period=optimizer_training_period
                        )
                    except Exception as e:
                        st.error(f"Error optimizing parameters for {ticker}: {e}")
                        continue
                
                # Daily risk-free rate (T-bill 3-month rate: 4.19%)
                daily_risk_free_rate = (1 + 0.0419) ** (1/252) - 1
                
                # Analyze each contract
                put_chain = option_chain.copy()
                
                for index, contract in put_chain.iterrows():
                    strike_price = contract['strike']
                    mid_price = (contract['bid'] + contract['ask']) / 2
                    simulated_returns = []
                    simulated_final_prices = []
                    profitable_count = 0
                    
                    # Run simulations
                    for _ in range(simulation_attempts):
                        prices = gbm(
                            s0=price, 
                            mu=optimized_mu, 
                            sigma=optimized_sigma, 
                            deltaT=days_to_expiration, 
                            dt=1
                        )
                        final_price = prices[-1]
                        
                        # Option expires worthless or at-the-money
                        if final_price >= strike_price:
                            profitable_count += 1
                            net_return = (mid_price / strike_price) * 100
                        else:
                            # Assigned: premium - (loss from assignment)
                            net_return = ((mid_price - (strike_price - final_price)) / strike_price) * 100
                        
                        simulated_returns.append(net_return)
                        simulated_final_prices.append(final_price)
                    
                    profitability_chance = (profitable_count / simulation_attempts) * 100
                    avg_return = np.mean(simulated_returns)
                    avg_price = np.mean(simulated_final_prices)
                    risk_free_return = daily_risk_free_rate * days_to_expiration
                    
                    downside_returns = [r for r in simulated_returns if r < risk_free_return]
                    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
                    
                    sortino_ratio = ((avg_return - risk_free_return) / downside_std) * np.sqrt(252 / days_to_expiration) if downside_std else 0
                    
                    put_chain.at[index, 'current_price'] = price
                    put_chain.at[index, 'final_price'] = avg_price
                    put_chain.at[index, 'profitability_likelihood'] = profitability_chance
                    put_chain.at[index, 'average_roi'] = avg_return
                    put_chain.at[index, 'sortino_ratio'] = sortino_ratio
                
                # Filter out in-the-money options and ones with high ROI
                put_chain = put_chain[(put_chain['inTheMoney'] != True) & (put_chain['ROI'] <= 4.0)]
                
                # Score the options
                if not put_chain.empty:
                    # Apply same scoring approach as in the notebook
                    def select_optimal_contract(contracts):
                        """Compute the weighted score for contracts using normalized values."""
                        temp_contracts = contracts.copy()
                        
                        scaler = MinMaxScaler()
                        temp_contracts[['profitability_likelihood', 'ROI', 'sortino_ratio']] = scaler.fit_transform(
                            temp_contracts[['profitability_likelihood', 'ROI', 'sortino_ratio']]
                        )
                        
                        temp_contracts['score'] = (
                            0.60 * temp_contracts['profitability_likelihood'] +
                            0.40 * temp_contracts['ROI'] +
                            0.10 * temp_contracts['sortino_ratio']
                        )
                        
                        contracts['score'] = temp_contracts['score']
                        
                        return contracts.sort_values(by='score', ascending=False)
                    
                    put_chain = select_optimal_contract(put_chain)
                    
                    all_results.append(put_chain)
                else:
                    st.warning(f"No suitable put options found for {ticker}")
            
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {e}")
            
            progress_bar.progress((i + 1) / len(selected_stocks))
        
        status_text.text("Analysis complete!")
        progress_bar.progress(100)
        
        # Display results
        if all_results:
            combined_results = pd.concat(all_results)
            
            with results_container:
                st.subheader("Top CSP Opportunities")
                
                # Display top options
                top_results = combined_results.sort_values('score', ascending=False).head(10)
                
                # Format the dataframe for display
                display_cols = [
                    'symbol', 'strike', 'current_price', 'profitability_likelihood', 
                    'ROI', 'mid_price', 'openInterest', 'volume', 'score'
                ]
                
                if all(col in top_results.columns for col in display_cols):
                    display_df = top_results[display_cols].copy()
                    
                    # Format numeric columns
                    display_df['profitability_likelihood'] = display_df['profitability_likelihood'].apply(lambda x: f"{x:.1f}%")
                    display_df['ROI'] = display_df['ROI'].apply(lambda x: f"{x:.2f}%")
                    display_df['score'] = display_df['score'].apply(lambda x: f"{x:.3f}")
                    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
                    display_df['strike'] = display_df['strike'].apply(lambda x: f"${x:.2f}")
                    display_df['mid_price'] = display_df['mid_price'].apply(lambda x: f"${x:.2f}")
                    
                    st.dataframe(display_df)
                else:
                    missing_cols = [col for col in display_cols if col not in top_results.columns]
                    st.error(f"Missing columns in results: {', '.join(missing_cols)}")
                    st.dataframe(top_results)
                
                # Interactive chart with Plotly
                st.subheader("Risk vs. Reward Analysis")
                
                fig = px.scatter(
                    combined_results, 
                    x='profitability_likelihood', 
                    y='ROI',
                    size='openInterest',
                    color='symbol',
                    hover_data=['strike', 'current_price', 'sortino_ratio', 'score'],
                    title='CSP Risk-Reward Profile',
                    labels={
                        'profitability_likelihood': 'Probability of Success (%)',
                        'ROI': 'Return on Investment (%)'
                    }
                )
                
                fig.update_layout(
                    xaxis_title='Probability of Success (%)',
                    yaxis_title='Return on Investment (%)',
                    legend_title='Symbol',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed view of individual options
                st.subheader("Individual Option Analysis")
                selected_option = st.selectbox(
                    "Select an option to view detailed analysis",
                    options=combined_results.index,
                    format_func=lambda idx: (
                        f"{combined_results.at[idx, 'symbol']} - "
                        f"Strike ${combined_results.at[idx, 'strike']:.2f} "
                        f"(Score: {combined_results.at[idx, 'score']:.3f})"
                    )
                )
                if selected_option is not None:
                    option_data = combined_results.loc[selected_option]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"{option_data['symbol']} - ${option_data['strike']} Strike")
                        st.metric("Current Stock Price", f"${option_data['current_price']:.2f}")
                        st.metric("Option Premium", f"${option_data['mid_price']:.2f}")
                        st.metric("Return on Investment", f"{option_data['ROI']:.2f}%")
                        st.metric("Success Probability", f"{option_data['profitability_likelihood']:.2f}%")
                    with col2:
                        st.subheader("Contract Details")
                        st.metric("Open Interest", f"{option_data['openInterest']}")
                        st.metric("Volume", f"{option_data['volume']}")
                        st.metric("Sortino Ratio", f"{option_data['sortino_ratio']:.3f}")
                        st.metric("Score", f"{option_data['score']:.3f}")
                    # Simulated price distribution
                    st.subheader("Simulated Price Distribution at Expiration")
                    avg_final_price = option_data['final_price']
                    current_price = float(option_data['current_price'])
                    strike_price = float(option_data['strike'])
                    simulated_prices = np.random.normal(
                        avg_final_price, 
                        optimized_sigma * np.sqrt(days_to_expiration) * current_price,
                        1000
                    )
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=simulated_prices,
                        name='Simulated Final Prices',
                        opacity=0.75,
                        nbinsx=50
                    ))
                    fig.add_vline(
                        x=current_price, 
                        line_dash="dash", 
                        line_color="blue",
                        annotation_text="Current Price"
                    )
                    fig.add_vline(
                        x=strike_price, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Strike Price"
                    )
                    fig.add_vline(
                        x=avg_final_price, 
                        line_dash="solid", 
                        line_color="green",
                        annotation_text="Average Final Price"
                    )
                    fig.update_layout(
                        title=f"Simulated Price Distribution for {option_data['symbol']} at Expiration",
                        xaxis_title="Stock Price ($)",
                        yaxis_title="Frequency",
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("### Analysis Explanation")
                    st.markdown(f"""
                    This analysis simulates {simulation_attempts} potential price paths for {option_data['symbol']} 
                    from today until the expiration date ({expiration_date.strftime('%Y-%m-%d')}). The simulation uses 
                    a Geometric Brownian Motion model with optimized parameters based on historical data.
                    
                    - **Success Probability**: {option_data['profitability_likelihood']:.2f}% chance the option expires worthless (stock above strike)
                    - **Potential Return**: {option_data['ROI']:.2f}% return on investment if the option expires worthless
                    - **Sortino Ratio**: {option_data['sortino_ratio']:.3f} (higher is better, measures return relative to downside risk)
                    
                    If assigned, your cost basis would be ${float(strike_price) - float(option_data['mid_price']):.2f} per share.
                    """)
        else:
            st.warning("No valid CSP options found for the selected stocks and parameters.")

elif analysis_mode == "Stock Trend Analysis":
    st.title("Stock Trend Analysis")
    
    # Stock symbol input
    ticker_symbol = st.sidebar.text_input("Enter Stock Symbol", "NBIS")
    lookback_days = st.sidebar.slider("Lookback Period (days)", 1, 30, 1)
    
    # Start analysis button
    analyze_button = st.sidebar.button("Analyze Trend")
    
    if analyze_button and ticker_symbol:
        st.subheader(f"Analyzing trend for {ticker_symbol}")
        
        try:
            def analyze_stock_trend(ticker_symbol, lookback_days=1):
                """
                Analyzes 1-minute candlesticks for a given stock to determine trend direction
                and identify potential entry points when price has bottomed out.
                
                Parameters:
                ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
                lookback_days (int): Number of trading days to look back (default: 1)
                
                Returns:
                dict: Analysis results including trend direction and entry signal
                """
                # Calculate start date based on lookback period
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)
                
                # Fetch 1-minute candlestick data
                ticker = Ticker(ticker_symbol)
                data = ticker.history(period='1d', interval='1m')
                
                # Reset index to make date a column and keep only relevant columns
                data = data.reset_index()
                data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
                
                # Calculate technical indicators
                # 1. Moving Averages
                data['sma_fast'] = data['close'].rolling(window=5).mean()
                data['sma_slow'] = data['close'].rolling(window=20).mean()
                data['ema_fast'] = data['close'].ewm(span=9, adjust=False).mean()
                data['ema_slow'] = data['close'].ewm(span=21, adjust=False).mean()
                
                # 2. MACD
                data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
                data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
                data['macd'] = data['ema_12'] - data['ema_26']
                data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
                data['macd_diff'] = data['macd'] - data['macd_signal']
                
                # 3. RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))
                
                # 4. Bollinger Bands
                data['bb_middle'] = data['close'].rolling(window=20).mean()
                data['bb_std'] = data['close'].rolling(window=20).std()
                data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
                data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
                data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
                
                # Determine trend direction based on multiple indicators
                # Look at the most recent data points
                recent_data = data.dropna().tail(30)
                
                # Trend determination
                sma_trend = 1 if recent_data['sma_fast'].iloc[-1] > recent_data['sma_slow'].iloc[-1] else -1
                ema_trend = 1 if recent_data['ema_fast'].iloc[-1] > recent_data['ema_slow'].iloc[-1] else -1
                macd_trend = 1 if recent_data['macd'].iloc[-1] > recent_data['macd_signal'].iloc[-1] else -1
                
                # Overall trend score (-3 to 3, higher is more bullish)
                trend_score = sma_trend + ema_trend + macd_trend
                
                # Entry point detection (bottoming out)
                entry_signal = False
                entry_strength = 0
                
                # Check for potential reversal signals
                # 1. RSI oversold and starting to rise
                if (recent_data['rsi'].iloc[-2] < 30 and recent_data['rsi'].iloc[-1] > recent_data['rsi'].iloc[-2]):
                    entry_signal = True
                    entry_strength += 1
                
                # 2. Price near or below lower Bollinger Band
                if (recent_data['close'].iloc[-1] <= recent_data['bb_lower'].iloc[-1] * 1.01):
                    entry_signal = True
                    entry_strength += 1
                
                # 3. MACD histogram starting to rise from negative territory
                if (recent_data['macd_diff'].iloc[-2] < 0 and 
                    recent_data['macd_diff'].iloc[-1] > recent_data['macd_diff'].iloc[-2]):
                    entry_signal = True
                    entry_strength += 1
                
                # 4. Volume increasing (possible accumulation)
                vol_avg = recent_data['volume'].iloc[-6:-1].mean()
                if recent_data['volume'].iloc[-1] > vol_avg * 1.2:
                    entry_strength += 1
                
                # Generate analysis results
                if trend_score >= 2:
                    trend_direction = "Strong Uptrend"
                elif trend_score > 0:
                    trend_direction = "Weak Uptrend"
                elif trend_score == 0:
                    trend_direction = "Sideways/Neutral"
                elif trend_score > -2:
                    trend_direction = "Weak Downtrend"
                else:
                    trend_direction = "Strong Downtrend"
                
                # Check for bottom confirmation based on trend and entry signals
                bottom_confirmation = False
                if trend_score >= 0 and entry_signal and entry_strength >= 2:
                    bottom_confirmation = True
                elif trend_score < 0 and entry_signal and entry_strength >= 3:
                    bottom_confirmation = True
                
                # Create analysis result
                analysis_result = {
                    'ticker': ticker_symbol,
                    'last_price': data['close'].iloc[-1],
                    'trend_direction': trend_direction,
                    'trend_score': trend_score,
                    'entry_signal': entry_signal,
                    'entry_strength': entry_strength,
                    'bottom_confirmation': bottom_confirmation,
                    'recommendation': "BUY" if bottom_confirmation else "WAIT",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return analysis_result, data
            
            def plot_analysis(data, ticker_symbol):
                """
                Creates plotly figures for the trend analysis.
                """
                # Price chart with MA and Bollinger Bands
                fig1 = go.Figure()
                
                # Add price
                fig1.add_trace(go.Scatter(
                    x=data['date'], 
                    y=data['close'],
                    name='Close Price',
                    line=dict(color='black', width=1)
                ))
                
                # Add moving averages
                fig1.add_trace(go.Scatter(
                    x=data['date'], 
                    y=data['sma_fast'],
                    name='SMA (5)',
                    line=dict(color='blue', width=1)
                ))
                
                fig1.add_trace(go.Scatter(
                    x=data['date'], 
                    y=data['sma_slow'],
                    name='SMA (20)',
                    line=dict(color='red', width=1)
                ))
                
                # Add Bollinger Bands
                fig1.add_trace(go.Scatter(
                    x=data['date'], 
                    y=data['bb_upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.7
                ))
                
                fig1.add_trace(go.Scatter(
                    x=data['date'], 
                    y=data['bb_middle'],
                    name='BB Middle',
                    line=dict(color='gray', width=1),
                    opacity=0.7
                ))
                
                fig1.add_trace(go.Scatter(
                    x=data['date'], 
                    y=data['bb_lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.7
                ))
                
                # Fill between bands
                fig1.add_trace(go.Scatter(
                    x=data['date'],
                    y=data['bb_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig1.add_trace(go.Scatter(
                    x=data['date'],
                    y=data['bb_lower'],
                    fill='tonexty',
                    mode='lines',
                    fillcolor='rgba(173, 216, 230, 0.2)',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                # Update layout
                fig1.update_layout(
                    title=f'{ticker_symbol} - Price with Moving Averages & Bollinger Bands',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                
                # MACD chart
                fig2 = go.Figure()
                
                # Add MACD line
                fig2.add_trace(go.Scatter(
                    x=data['date'], 
                    y=data['macd'],
                    name='MACD',
                    line=dict(color='blue', width=1)
                ))
                
                # Add signal line
                fig2.add_trace(go.Scatter(
                    x=data['date'], 
                    y=data['macd_signal'],
                    name='Signal',
                    line=dict(color='red', width=1)
                ))
                
                # Add histogram
                colors = ['green' if val >= 0 else 'red' for val in data['macd_diff']]
                
                fig2.add_trace(go.Bar(
                    x=data['date'], 
                    y=data['macd_diff'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.7
                ))
                
                # Add zero line
                fig2.add_hline(
                    y=0, 
                    line_dash="solid", 
                    line_color="black",
                    opacity=0.3
                )
                
                # Update layout
                fig2.update_layout(
                    title='MACD Indicator',
                    xaxis_title='Time',
                    yaxis_title='MACD',
                    height=400,
                    xaxis_rangeslider_visible=False
                )
                
                # RSI chart
                fig3 = go.Figure()
                
                # Add RSI line
                fig3.add_trace(go.Scatter(
                    x=data['date'], 
                    y=data['rsi'],
                    name='RSI',
                    line=dict(color='purple', width=1)
                ))
                
                # Add overbought/oversold lines
                fig3.add_hline(
                    y=70, 
                    line_dash="dash", 
                    line_color="red",
                    opacity=0.7,
                    annotation_text="Overbought"
                )
                
                fig3.add_hline(
                    y=30, 
                    line_dash="dash", 
                    line_color="green",
                    opacity=0.7,
                    annotation_text="Oversold"
                )
                
                # Fill overbought/oversold regions
                y_vals = data['rsi'].values
                x_vals = data['date'].values
                
                # Fill oversold region (RSI < 30)
                y_oversold = [min(30, val) for val in y_vals]
                fig3.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_oversold,
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig3.add_trace(go.Scatter(
                    x=x_vals,
                    y=[30] * len(x_vals),
                    fill='tonexty',
                    mode='lines',
                    fillcolor='rgba(0, 255, 0, 0.1)',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                # Fill overbought region (RSI > 70)
                y_overbought = [max(70, val) for val in y_vals]
                fig3.add_trace(go.Scatter(
                    x=x_vals,
                    y=[70] * len(x_vals),
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig3.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_overbought,
                    fill='tonexty',
                    mode='lines',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                # Update layout
                fig3.update_layout(
                    title='Relative Strength Index (RSI)',
                    xaxis_title='Time',
                    yaxis_title='RSI',
                    height=400,
                    yaxis=dict(range=[0, 100]),
                    xaxis_rangeslider_visible=False
                )
                
                return fig1, fig2, fig3
            
            # Run the analysis
            with st.spinner(f"Analyzing {ticker_symbol}..."):
                analysis_result, data = analyze_stock_trend(ticker_symbol, lookback_days)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Create a card-like display for the results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${analysis_result['last_price']:.2f}")
                
                with col2:
                    trend_delta = f"{analysis_result['trend_score']:+d}" if analysis_result['trend_score'] != 0 else "0"
                    st.metric("Trend Direction", analysis_result['trend_direction'], trend_delta)
                
                with col3:
                    entry_value = "Yes" if analysis_result['entry_signal'] else "No"
                    st.metric("Entry Signal", entry_value, analysis_result['entry_strength'])
                
                with col4:
                    st.metric("Recommendation", analysis_result['recommendation'])
                
                # Plot the analysis
                fig1, fig2, fig3 = plot_analysis(data.dropna(), ticker_symbol)
                
                st.plotly_chart(fig1, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig2, use_container_width=True)
                with col2:
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Detailed explanation
                st.subheader("Analysis Explanation")
                
                st.markdown(f"""
                ### Summary
                The analysis for **{ticker_symbol}** shows a **{analysis_result['trend_direction'].lower()}** with 
                a trend score of **{analysis_result['trend_score']}** (range from -3 to +3, positive is bullish).
                
                ### Technical Indicators
                
                - **Moving Averages**: {("Bullish - Fast MA above Slow MA" if analysis_result['trend_score'] > 0 else "Bearish - Fast MA below Slow MA")}
                - **MACD**: {("Bullish signal" if 'macd' in data.columns and data['macd'].iloc[-1] > data['macd_signal'].iloc[-1] else "Bearish signal")}
                - **RSI**: {data['rsi'].iloc[-1]:.1f} ({("Oversold" if data['rsi'].iloc[-1] < 30 else "Overbought" if data['rsi'].iloc[-1] > 70 else "Neutral")})

                ### Entry Signal Analysis

                Entry signal strength: **{analysis_result['entry_strength']}/4**

                Reversal signals detected:
                """)

                # Add each signal check separately to avoid reference errors
                recent_data = data.dropna().tail(30)

                # RSI check
                rsi_signal = recent_data['rsi'].iloc[-2] < 30 and recent_data['rsi'].iloc[-1] > recent_data['rsi'].iloc[-2]
                st.markdown(f"- {'✓' if rsi_signal else '✗'} RSI oversold and rising")

                # Bollinger Band check
                bb_signal = recent_data['close'].iloc[-1] <= recent_data['bb_lower'].iloc[-1] * 1.01
                st.markdown(f"- {'✓' if bb_signal else '✗'} Price near lower Bollinger Band")

                # MACD histogram check
                macd_signal = recent_data['macd_diff'].iloc[-2] < 0 and recent_data['macd_diff'].iloc[-1] > recent_data['macd_diff'].iloc[-2]
                st.markdown(f"- {'✓' if macd_signal else '✗'} MACD histogram rising from negative")

                # Volume check
                vol_avg = recent_data['volume'].iloc[-6:-1].mean()
                vol_signal = recent_data['volume'].iloc[-1] > vol_avg * 1.2
                st.markdown(f"- {'✓' if vol_signal else '✗'} Volume increasing (possible accumulation)")

                st.markdown(f"""
                ### Recommendation

                **{analysis_result['recommendation']}**: {("Bottom confirmation signals identified. Consider entering a position." if analysis_result['bottom_confirmation'] else "Wait for stronger confirmation signals before entering.")}
                """)
                
                # Additional information about the analysis
                with st.expander("How this analysis works"):
                    st.markdown("""
                    This analysis examines 1-minute price data to determine the current trend direction and identify potential entry points, 
                    particularly when a stock may be bottoming out after a decline.
                    
                    ### Methodology
                    
                    1. **Trend Determination**:
                       - Simple Moving Averages (5 vs 20 periods)
                       - Exponential Moving Averages (9 vs 21 periods)
                       - MACD (12, 26, 9)
                       - Combined trend score ranges from -3 (strong downtrend) to +3 (strong uptrend)
                    
                    2. **Entry Signal Detection**:
                       - RSI below 30 and starting to rise (oversold bounce)
                       - Price at or below lower Bollinger Band (potential reversal point)
                       - MACD histogram starting to rise from negative territory
                       - Increasing volume (possible accumulation)
                    
                    3. **Bottom Confirmation**:
                       - For uptrends or neutral markets: 2+ entry signals needed
                       - For downtrends: 3+ entry signals needed
                    
                    The algorithm is designed to identify potential bottoming patterns and reversal signals for short-term trading opportunities.
                    """)
                    
        except Exception as e:
            st.error(f"Error analyzing {ticker_symbol}: {e}")

elif analysis_mode == "Covered Calls Analysis":
    st.title("Covered Calls Analyzer")
    
    # Sidebar inputs
    today = datetime.today()
    default_expiration = today + timedelta(days=30)
    expiration_date = st.sidebar.date_input(
        "Option Expiration Date", 
        value=default_expiration,
        min_value=today
    )
    ticker = st.sidebar.text_input("Stock Symbol", "NBIS")
    cost_basis = st.sidebar.number_input("Cost Basis per Share ($)", min_value=0.0, value=30.0, step=0.01)
    simulation_attempts = st.sidebar.slider("Simulation Attempts", 100, 1000, 300)
    start_analysis = st.sidebar.button("Analyze Covered Calls")

    if start_analysis and ticker:
        st.subheader(f"Analyzing covered calls for {ticker} expiring on {expiration_date.strftime('%Y-%m-%d')}")
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        def get_option_chain(symbol, option_type="calls", expiration_date=None):
            try:
                stock = Ticker(symbol)
                df = stock.option_chain
                if df is None or df.empty:
                    st.warning(f"No option data available for {symbol}.")
                    return None
                if expiration_date:
                    df = df.loc[df.index.get_level_values('expiration') == expiration_date.strftime("%Y-%m-%d")]
                if option_type in ['calls', 'puts']:
                    df = df.xs(option_type, level=2)
                df = df.reset_index()
                df = df.rename(columns={'expiration': 'expiration_date'})
                df['mid_price'] = (df['bid'] + df['ask']) / 2
                df['ROI'] = (df['mid_price'] / df['strike']) * 100
                df = df[(df['openInterest'] >= 40) & (df['bid'] > 0.00) & (df['ROI'] > 0.1) & (((df['ask'] - df['bid']) / df['bid']) * 100 < 40)]
                return df
            except Exception as e:
                st.error(f"Error fetching option chain for {symbol}: {e}")
                return None
        
        def get_current_stock_price(symbol):
            ticker = yf.Ticker(symbol)
            todays_data = ticker.history(period='1d')
            return todays_data['Close'].iloc[-1]
        
        def fit_garch(symbol, period):
            stock_data = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
            real_prices = stock_data["Close"].dropna().values.flatten()
            returns = np.diff(np.log(real_prices))
            model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            garch_fit = model.fit(disp="off")
            conditional_volatilities = garch_fit.conditional_volatility
            N = len(conditional_volatilities)
            weights = np.linspace(1, 2, N)
            weighted_volatility = np.sum(weights * conditional_volatilities) / np.sum(weights)
            return weighted_volatility
        
        # Get option chain
        option_chain = get_option_chain(ticker, option_type="calls", expiration_date=expiration_date)
        if option_chain is None or option_chain.empty:
            st.warning(f"No valid call options found for {ticker}")
        else:
            price = get_current_stock_price(ticker)
            optimizer_training_period = "6mo"
            bin_length = 18
            days_to_expiration = np.busday_count(datetime.today().date(), expiration_date)
            with st.spinner(f"Optimizing parameters for {ticker}..."):
                try:
                    optimized_mu, optimized_sigma = optimize_gbm(
                        symbol=ticker, 
                        training_period=optimizer_training_period, 
                        bin_length=bin_length
                    )
                    optimized_sigma = fit_garch(
                        symbol=ticker, 
                        period=optimizer_training_period
                    )
                except Exception as e:
                    st.error(f"Error optimizing parameters for {ticker}: {e}")
                    st.stop()
            daily_risk_free_rate = (1 + 0.0419) ** (1/252) - 1
            call_chain = option_chain.copy()
            for index, contract in call_chain.iterrows():
                strike_price = contract['strike']
                mid_price = (contract['bid'] + contract['ask']) / 2
                simulated_returns = []
                simulated_final_prices = []
                profitable_count = 0
                for _ in range(simulation_attempts):
                    prices = gbm(
                        s0=price, 
                        mu=optimized_mu, 
                        sigma=optimized_sigma, 
                        deltaT=days_to_expiration, 
                        dt=1
                    )
                    final_price = prices[-1]
                    # Option expires worthless (stock below strike): keep premium
                    if final_price <= strike_price:
                        profitable_count += 1
                        net_return = (mid_price / strike_price) * 100
                    else:
                        # Assigned: premium + (strike - cost_basis)
                        net_return = ((strike_price - (cost_basis - mid_price)) / strike_price) * 100
                    simulated_returns.append(net_return)
                    simulated_final_prices.append(final_price)
                profitability_chance = (profitable_count / simulation_attempts) * 100
                percent_return = (mid_price / strike_price) * 100
                avg_price = np.mean(simulated_final_prices)
                call_chain.at[index, 'mid_price'] = mid_price
                call_chain.at[index, 'current_price'] = price
                call_chain.at[index, 'final_price'] = avg_price
                call_chain.at[index, 'profitability_likelihood'] = profitability_chance
                call_chain.at[index, 'return_percent'] = percent_return
                call_chain.at[index, 'return_if_assignment'] = ((strike_price - (cost_basis - mid_price)) / strike_price) * 100
            # Score the options
            def select_optimal_contract(contracts):
                temp_contracts = contracts.copy()
                scaler = MinMaxScaler()
                temp_contracts[['profitability_likelihood', 'return_percent']] = scaler.fit_transform(
                    temp_contracts[['profitability_likelihood', 'return_percent']]
                )
                temp_contracts['score'] = (
                    0.60 * temp_contracts['profitability_likelihood'] +
                    0.40 * temp_contracts['return_percent']
                )
                contracts['score'] = temp_contracts['score']
                return contracts.sort_values(by='score', ascending=False)
            call_chain = select_optimal_contract(call_chain)
            with results_container:
                st.subheader("Top Covered Call Opportunities")
                display_cols = [
                    'symbol', 'strike', 'current_price', 'profitability_likelihood', 
                    'return_percent', 'mid_price', 'openInterest', 'volume', 'score', 'return_if_assignment'
                ]
                if all(col in call_chain.columns for col in display_cols):
                    display_df = call_chain[display_cols].copy()
                    display_df['profitability_likelihood'] = display_df['profitability_likelihood'].apply(lambda x: f"{x:.1f}%")
                    display_df['return_percent'] = display_df['return_percent'].apply(lambda x: f"{x:.2f}%")
                    display_df['score'] = display_df['score'].apply(lambda x: f"{x:.3f}")
                    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
                    display_df['strike'] = display_df['strike'].apply(lambda x: f"${x:.2f}")
                    display_df['mid_price'] = display_df['mid_price'].apply(lambda x: f"${x:.2f}")
                    display_df['return_if_assignment'] = display_df['return_if_assignment'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(display_df)
                else:
                    missing_cols = [col for col in display_cols if col not in call_chain.columns]
                    st.error(f"Missing columns in results: {', '.join(missing_cols)}")
                    st.dataframe(call_chain)
                # Interactive chart
                st.subheader("Risk vs. Reward Analysis")
                fig = px.scatter(
                    call_chain, 
                    x='profitability_likelihood', 
                    y='return_percent',
                    size='openInterest',
                    color='symbol',
                    hover_data=['strike', 'current_price', 'score', 'return_if_assignment'],
                    title='Covered Call Risk-Reward Profile',
                    labels={
                        'profitability_likelihood': 'Probability of Expiring Worthless (%)',
                        'return_percent': 'Return on Investment (%)'
                    }
                )
                fig.update_layout(
                    xaxis_title='Probability of Expiring Worthless (%)',
                    yaxis_title='Return on Investment (%)',
                    legend_title='Symbol',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                # Detailed view
                st.subheader("Individual Option Analysis")
                selected_option = st.selectbox(
                    "Select an option to view detailed analysis",
                    options=call_chain.index,
                    format_func=lambda idx: (
                        f"{call_chain.at[idx, 'symbol']} - "
                        f"Strike ${call_chain.at[idx, 'strike']:.2f} "
                        f"(Score: {call_chain.at[idx, 'score']:.3f})"
                    )
                )
                if selected_option is not None:
                    option_data = call_chain.loc[selected_option]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"{option_data['symbol']} - ${option_data['strike']} Strike")
                        st.metric("Current Stock Price", f"${option_data['current_price']}")
                        st.metric("Option Premium", f"${option_data['mid_price']:.2f}")
                        st.metric("Return on Investment", f"{option_data['return_percent']:.2f}%")
                        st.metric("Success Probability", f"{option_data['profitability_likelihood']:.2f}%")
                    with col2:
                        st.subheader("Contract Details")
                        st.metric("Open Interest", f"{option_data['openInterest']}")
                        st.metric("Volume", f"{option_data['volume']}")
                        st.metric("Score", f"{option_data['score']:.3f}")
                        st.metric("Return if Assigned", f"{option_data['return_if_assignment']:.2f}%")
                    # Simulated price distribution
                    st.subheader("Simulated Price Distribution at Expiration")
                    avg_final_price = option_data['final_price']
                    current_price = float(option_data['current_price'])
                    strike_price = float(option_data['strike'])
                    simulated_prices = np.random.normal(
                        avg_final_price, 
                        optimized_sigma * np.sqrt(days_to_expiration) * current_price,
                        1000
                    )
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=simulated_prices,
                        name='Simulated Final Prices',
                        opacity=0.75,
                        nbinsx=50
                    ))
                    fig.add_vline(
                        x=current_price, 
                        line_dash="dash", 
                        line_color="blue",
                        annotation_text="Current Price"
                    )
                    fig.add_vline(
                        x=strike_price, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Strike Price"
                    )
                    fig.add_vline(
                        x=avg_final_price, 
                        line_dash="solid", 
                        line_color="green",
                        annotation_text="Average Final Price"
                    )
                    fig.update_layout(
                        title=f"Simulated Price Distribution for {option_data['symbol']} at Expiration",
                        xaxis_title="Stock Price ($)",
                        yaxis_title="Frequency",
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("### Analysis Explanation")
                    st.markdown(f"""
                    This analysis simulates {simulation_attempts} potential price paths for {option_data['symbol']} 
                    from today until the expiration date ({expiration_date.strftime('%Y-%m-%d')}). The simulation uses 
                    a Geometric Brownian Motion model with optimized parameters based on historical data.
                    
                    - **Success Probability**: {option_data['profitability_likelihood']:.2f}% chance the option expires worthless (stock below strike)
                    - **Potential Return**: {option_data['return_percent']:.2f}% return on investment if the option expires worthless
                    - **Return if Assigned**: {option_data['return_if_assignment']:.2f}% 
                    
                    If assigned, your net sale price would be ${float(strike_price) + float(option_data['mid_price']):.2f} per share.
                    
                    """)

