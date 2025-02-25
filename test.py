import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import requests
import QuantLib as ql
import numpy as np
import json
import matplotlib.pyplot as plt
from arch import arch_model
from yahooquery import Ticker
import datetime as dt
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px

# Try importing optional dependencies (may need to be installed)
try:
    from tradingview_screener import Query, col
    import rookiepy
    import gbm_optimizer
    from gbm_optimizer import optimize_gbm, gbm
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False
    st.warning("Some advanced features are disabled due to missing dependencies. Install tradingview_screener, rookiepy, and gbm_optimizer for full functionality.")

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Analysis Tool",
    page_icon="📈",
    layout="wide"
)

# Sidebar configuration
st.sidebar.title("Stock Analysis Tool")
st.sidebar.markdown("Configure your analysis parameters")

# App modes
app_mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    ["Technical Analysis", "Options Strategy", "Stock Screener"]
)

# Load configuration if available
try:
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    api_key = config.get("api_key")
    secret_key = config.get("secret_key")
    has_config = True
except FileNotFoundError:
    has_config = False
    api_key = st.sidebar.text_input("Alpaca API Key (optional)", type="password")
    secret_key = st.sidebar.text_input("Alpaca Secret Key (optional)", type="password")

# Functions from original code
def get_current_stock_price(symbol: str):
    if not api_key or not secret_key:
        # Use yfinance as fallback when API keys aren't available
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return data["Close"].iloc[-1]
            return None
        except Exception as e:
            st.error(f"Error fetching stock price with yfinance: {e}")
            return None
    
    url = "https://data.alpaca.markets/v2/stocks/trades/latest"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    params = {
        "symbols": symbol,
        "feed": "iex"
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("trades", {}).get(symbol, {}).get("p")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching stock price: {e}")
        return None

def get_option_chain(symbol, option_type="puts", expiration_date=None):
    try:
        stock = Ticker(symbol)
        df = stock.option_chain  # Fetch all available option chain data

        if df is None or df.empty:
            st.warning(f"No option data available for {symbol}.")
            return None

        # Filter by expiration date if provided
        if expiration_date:
            df = df.loc[df.index.get_level_values('expiration') == expiration_date]

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
        min_open_interest = st.sidebar.slider("Min Open Interest", 0, 500, 50, key="open_interest_slider")
        min_roi = st.sidebar.slider("Min ROI (%)", 0.00, 5.00, 0.10, key="min_roi_slider")
        max_spread = st.sidebar.slider("Max Bid-Ask Spread (%)", 0, 100, 40, key="max_spread_slider")
        
        df = df[(df['openInterest'] >= min_open_interest) & 
                (df['bid'] > 0.00) & 
                (df['ROI'] > min_roi) &
                (((df['ask'] - df['bid']) / df['bid']) * 100 < max_spread)
                ] 

        return df

    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None

def fit_garch(symbol, period):
    with st.spinner(f"Fitting GARCH model for {symbol}..."):
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

def analyze_stock_trend(ticker_symbol, lookback_days=1):
    """
    Analyzes 1-minute candlesticks for a given stock to determine trend direction
    and identify potential entry points when price has bottomed out.
    """
    with st.spinner(f"Analyzing {ticker_symbol} trend data..."):
        # Calculate start date based on lookback period
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=lookback_days)
        
        # Fetch 1-minute candlestick data
        ticker = Ticker(ticker_symbol)
        data = ticker.history(period='1d', interval='1m')
        
        # Reset index to make date a column and keep only relevant columns
        data = data.reset_index()
        data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        # Calculate technical indicators
        # 1. Moving Averages
        data['sma_fast'] = SMAIndicator(close=data['close'], window=5).sma_indicator()
        data['sma_slow'] = SMAIndicator(close=data['close'], window=20).sma_indicator()
        data['ema_fast'] = EMAIndicator(close=data['close'], window=9).ema_indicator()
        data['ema_slow'] = EMAIndicator(close=data['close'], window=21).ema_indicator()
        
        # 2. MACD
        macd = MACD(close=data['close'], window_slow=26, window_fast=12, window_sign=9)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()
        
        # 3. RSI
        data['rsi'] = RSIIndicator(close=data['close'], window=14).rsi()
        
        # 4. Bollinger Bands
        bollinger = BollingerBands(close=data['close'], window=20, window_dev=2)
        data['bb_upper'] = bollinger.bollinger_hband()
        data['bb_middle'] = bollinger.bollinger_mavg()
        data['bb_lower'] = bollinger.bollinger_lband()
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Determine trend direction based on multiple indicators
        # Look at the most recent data points
        recent_data = data.tail(30)
        
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
            'timestamp': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return analysis_result, data

def plot_analysis_plotly(data, ticker_symbol):
    """
    Creates Plotly visualization of technical analysis
    """
    # Create subplots
    fig = go.Figure()
    
    # Add price with Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['close'],
        name='Close Price',
        line=dict(color='black')
    ))
    
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['sma_fast'],
        name='SMA (5)',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['sma_slow'],
        name='SMA (20)',
        line=dict(color='red', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['bb_upper'],
        name='BB Upper',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['bb_lower'],
        name='BB Lower',
        line=dict(color='gray', width=1, dash='dash'),
        fill='tonexty',
        opacity=0.7
    ))
    
    # Add buttons to show/hide different indicators
    fig.update_layout(
        title=f'{ticker_symbol} - Technical Analysis',
        xaxis_title='Time',
        yaxis_title='Price',
        height=600,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.1,
                y=1.15,
                buttons=list([
                    dict(label="All",
                         method="update",
                         args=[{"visible": [True, True, True, True, True]}]),
                    dict(label="Price Only",
                         method="update",
                         args=[{"visible": [True, False, False, False, False]}]),
                    dict(label="Moving Avgs",
                         method="update",
                         args=[{"visible": [True, True, True, False, False]}]),
                    dict(label="Bollinger",
                         method="update",
                         args=[{"visible": [True, False, False, True, True]}]),
                ]),
            )
        ]
    )
    
    # Create additional figures for MACD and RSI
    fig_macd = px.line(data, x='date', y=['macd', 'macd_signal'], 
                      title=f"{ticker_symbol} - MACD")
    fig_macd.add_bar(x=data['date'], y=data['macd_diff'], name='Histogram')
    fig_macd.update_layout(height=300)
    
    fig_rsi = px.line(data, x='date', y='rsi', title=f"{ticker_symbol} - RSI")
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(height=300)
    
    return fig, fig_macd, fig_rsi

def select_optimal_contract(contracts):
    """Compute the weighted score for contracts using normalized values."""
    if contracts.empty:
        return contracts
        
    temp_contracts = contracts.copy()
    
    # Check if required columns exist
    required_cols = ['profitability_likelihood', 'ROI', 'sortino_ratio']
    if not all(col in temp_contracts.columns for col in required_cols):
        st.warning("Missing required columns for scoring. Using ROI only.")
        if 'ROI' in temp_contracts.columns:
            return contracts.sort_values(by='ROI', ascending=False)
        return contracts
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    temp_contracts[required_cols] = scaler.fit_transform(temp_contracts[required_cols])
    
    # Calculate score
    weight_prof = st.sidebar.slider("Profitability Weight", 0.0, 1.0, 0.6)
    weight_roi = st.sidebar.slider("ROI Weight", 0.0, 1.0, 0.4)
    weight_sortino = st.sidebar.slider("Risk-Adjusted Weight", 0.0, 1.0, 0.1)
    
    temp_contracts['score'] = (
        weight_prof * temp_contracts['profitability_likelihood'] +
        weight_roi * temp_contracts['ROI'] +
        weight_sortino * temp_contracts['sortino_ratio']
    )
    
    contracts['score'] = temp_contracts['score']
    
    return contracts.sort_values(by='score', ascending=False)

# Conditional import for screen_stocks if dependencies are available
if ADVANCED_FEATURES:
    def screen_stocks():
        st.info("Fetching stocks from TradingView...")
        try:
            # Get cookies for TradingView session
            cookies = rookiepy.to_cookiejar(rookiepy.chrome(['.tradingview.com']))
            
            min_price = st.sidebar.slider("Min Price ($)", 1, 100, 15)
            max_price = st.sidebar.slider("Max Price ($)", min_price, 200, 30)
            min_change = st.sidebar.slider("Min Change (%)", -10.0, 0.0, -4.0)
            max_change = st.sidebar.slider("Max Change (%)", min_change, 10.0, -2.0)
            min_3m_perf = st.sidebar.slider("Min 3-Month Performance (%)", -20.0, 50.0, 0.0)
            min_6m_perf = st.sidebar.slider("Min 6-Month Performance (%)", -20.0, 50.0, 5.0)
            
            _, df = Query().select('close', 'change', 'Perf.3M', 'Perf.6M').where(
                col('close').between(min_price, max_price),
                col('change').between(min_change, max_change),
                col('Perf.3M') > min_3m_perf,
                col('Perf.6M') > min_6m_perf,
                col('exchange').isin(['AMEX', 'NASDAQ', 'NYSE']),
            ).limit(1000).get_scanner_data(cookies=cookies)
            
            df[['exchange', 'ticker']] = df['ticker'].str.split(':', expand=True)
            
            return df
        except Exception as e:
            st.error(f"Error in stock screening: {e}")
            return pd.DataFrame()

    def audit_single_ticker(ticker, expiration_date):
        """
        Analyze a single ticker for option strategies
        """
        with st.spinner(f"Analyzing options for {ticker}..."):
            # T-bill 3-month rate 
            daily_risk_free_rate = (1 + 0.0419) ** (1/252) - 1

            simulation_attempts = st.sidebar.slider("Simulation Attempts", 100, 1000, 500, key="simulation_slider")

            optimizer_training_period = st.sidebar.selectbox(
                "Training Period", 
                ["1mo", "3mo", "6mo", "1y"], 
                index=2
            )
            
            bin_length = st.sidebar.slider("Bin Length", 5, 30, 18, key="bin_length_slider")
            days_to_expiration = np.busday_count(datetime.today().date(), expiration_date.date())

            option_chain = get_option_chain(symbol=ticker, expiration_date=expiration_date)

            if option_chain is None or option_chain.empty:
                st.warning(f"No options available for {ticker} on {expiration_date}")
                return pd.DataFrame()

            price = get_current_stock_price(ticker)
            if price is None:
                st.error(f"Could not get current price for {ticker}")
                return pd.DataFrame()

            try:
                optimized_mu, initial_sigma = optimize_gbm(
                    symbol=ticker, 
                    training_period=optimizer_training_period, 
                    bin_length=bin_length
                )
                optimized_sigma = fit_garch(symbol=ticker, period=optimizer_training_period)
                
                st.info(f"GBM Parameters - μ: {optimized_mu:.6f}, σ: {optimized_sigma:.6f}")
                
                # Create a toggle to show GBM vs real graph
                if st.checkbox("Show GBM vs Real Price Comparison"):
                    gbm_vs_real_fig = gbm_vs_real_graph(
                        symbol=ticker, 
                        mu=optimized_mu, 
                        sigma=optimized_sigma, 
                        period=optimizer_training_period, 
                        return_figure=True
                    )
                    st.pyplot(gbm_vs_real_fig)
            except Exception as e:
                st.error(f"Error optimizing GBM parameters: {e}")
                return pd.DataFrame()

            put_chain = option_chain.copy()

            for index, contract in put_chain.iterrows():
                strike_price = contract['strike']
                mid_price = (contract['bid'] + contract['ask']) / 2
                simulated_returns = []
                simulated_final_prices = []
                profitable_count = 0

                for _ in range(simulation_attempts):
                    prices = gbm(s0=price, mu=optimized_mu, sigma=optimized_sigma, deltaT=days_to_expiration, dt=1)
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

            # Drop unnecessary columns
            put_chain.drop(columns=['contractSymbol', 'currency', 'contractSize', 'lastTradeDate', 'impliedVolatility'], 
                        errors='ignore', inplace=True)

            # Filter in-the-money and high ROI
            max_roi = st.sidebar.slider("Max ROI Filter (%)", 1.00, 10.00, 4.00, key="max_roi_slider")
            put_chain = put_chain[(put_chain['inTheMoney'] != True) & (put_chain['ROI'] <= max_roi)]

            return put_chain

    def gbm_vs_real_graph(symbol, mu, sigma, period, return_figure=False):
        stock_data = yf.download(symbol, period=period, interval="1d", progress=False)
        real_prices = stock_data["Close"].dropna().values
        time_steps = np.arange(len(real_prices))

        gbm_path = gbm(s0=real_prices[0], mu=mu, sigma=sigma, deltaT=len(real_prices), dt=1)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_steps, real_prices, label="Real Prices", color="blue")
        ax.plot(time_steps, gbm_path, label="GBM Simulated", linestyle="dashed", color="red")
        
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Price")
        ax.set_title(f"GBM vs Real Prices for {symbol}")
        ax.legend()
        ax.grid()
        
        if return_figure:
            return fig
        else:
            st.pyplot(fig)

    def audit_stocks(expiration_date, tickers=None):
        if tickers is None or not tickers:
            if st.checkbox("Use screened stocks"):
                try:
                    screened_stocks = screen_stocks()
                    if not screened_stocks.empty:
                        tickers = screened_stocks['ticker'].to_list()
                    else:
                        st.warning("No stocks found from screening")
                        return pd.DataFrame()
                except Exception as e:
                    st.error(f"Error during stock screening: {e}")
                    return pd.DataFrame()
            else:
                tickers_input = st.text_input(
                    "Enter stock tickers (comma separated)",
                    "CHWY, IBIT, ASO, MARA, ET, DVN, INTC, SPLG, TOST, NBIS, ON"
                )
                tickers = [ticker.strip() for ticker in tickers_input.split(',')]

        # Display progress
        progress_bar = st.progress(0)
        all_options = []
        
        for i, ticker in enumerate(tickers):
            st.info(f"Processing {ticker} ({i+1}/{len(tickers)})")
            put_chain = audit_single_ticker(ticker=ticker, expiration_date=expiration_date)
            if not put_chain.empty:
                all_options.append(put_chain)
            # Update progress bar
            progress_bar.progress((i + 1) / len(tickers))

        # Combine all results into a single DataFrame
        if all_options:
            result_df = pd.concat(all_options, ignore_index=True)
            return result_df

        return pd.DataFrame()

# Main app layout based on selected mode
if app_mode == "Technical Analysis":
    st.title("Stock Technical Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticker_symbol = st.text_input("Enter Stock Ticker", "AAPL")
    
    with col2:
        lookback_days = st.number_input("Lookback Days", min_value=1, max_value=7, value=1)
    
    if st.button("Analyze Stock"):
        if ticker_symbol:
            try:
                analysis_result, data = analyze_stock_trend(ticker_symbol.upper(), lookback_days)
                
                # Display results in a styled card
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Analysis Results")
                    st.metric("Current Price", f"${analysis_result['last_price']:.2f}")
                    st.metric("Trend Direction", analysis_result['trend_direction'])
                    st.metric("Trend Score", analysis_result['trend_score'])
                
                with col2:
                    st.subheader("Trading Signals")
                    st.metric("Entry Signal", "Yes" if analysis_result['entry_signal'] else "No")
                    st.metric("Entry Strength", f"{analysis_result['entry_strength']}/4")
                    st.metric("Recommendation", analysis_result['recommendation'], 
                             delta="Buy Now" if analysis_result['recommendation'] == "BUY" else "Hold")
                
                # Create plotly charts
                fig_price, fig_macd, fig_rsi = plot_analysis_plotly(data, ticker_symbol)
                
                # Display the charts
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Show MACD and RSI side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_macd, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analyzing stock: {e}")

elif app_mode == "Options Strategy" and ADVANCED_FEATURES:
    st.title("Options Strategy Analyzer")
    
    # Date selection
    today = datetime.today()
    default_date = today + timedelta(days=30)  # Default to 30 days in future
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        tickers_input = st.text_input(
            "Enter stock tickers to analyze (comma separated)", 
            "AAPL, MSFT, GOOGL"
        )
        tickers = [ticker.strip() for ticker in tickers_input.split(',')]
    
    with col2:
        expiration_date = st.date_input(
            "Option Expiration Date", 
            default_date,
            min_value=today
        )
    
    if st.button("Find Option Strategies"):
        expiration_datetime = datetime.combine(expiration_date, datetime.min.time())
        
        if tickers:
            results = audit_stocks(expiration_datetime, tickers)
            
            if not results.empty:
                # Score and sort the contracts
                scored_results = select_optimal_contract(results)
                
                # Display top strategies
                st.subheader("Recommended Option Strategies")
                st.dataframe(
                    scored_results[['ticker', 'strike', 'expiration_date', 'bid', 'ask', 
                                   'mid_price', 'ROI', 'profitability_likelihood', 
                                   'average_roi', 'sortino_ratio', 'score']]
                )
                
                # Show detailed analysis for selected contract
                st.subheader("Contract Details")
                selected_contract = st.selectbox(
                    "Select contract for detailed analysis",
                    options=scored_results.apply(
                        lambda x: f"{x['ticker']} - Strike: ${x['strike']} - Score: {x['score']:.2f}", 
                        axis=1
                    )
                )
                
                if selected_contract:
                    # Extract ticker from selection
                    selected_ticker = selected_contract.split(' - ')[0]
                    selected_strike = float(selected_contract.split('Strike: $')[1].split(' - ')[0])
                    
                    # Find the matching contract
                    contract_details = scored_results[
                        (scored_results['ticker'] == selected_ticker) & (scored_results['strike'] == selected_strike)
                    ].iloc[0]

                    # Display detailed contract information
                    st.write("### Contract Details")
                    st.json(contract_details.to_dict())

                    # Show GBM vs Real Price Comparison for the selected ticker
                    st.write("### GBM vs Real Price Comparison")
                    gbm_vs_real_fig = gbm_vs_real_graph(
                        symbol=selected_ticker,
                        mu=contract_details.get('optimized_mu', 0.0),
                        sigma=contract_details.get('optimized_sigma', 0.0),
                        period="3mo",
                        return_figure=True
                    )
                    st.pyplot(gbm_vs_real_fig)

                    # Show historical volatility and GARCH analysis
                    st.write("### Volatility Analysis")
                    with st.spinner("Calculating historical volatility..."):
                        historical_volatility = fit_garch(selected_ticker, period="3mo")
                        st.metric("Historical Volatility (GARCH)", f"{historical_volatility:.4f}")

                    # Show option chain for the selected ticker
                    st.write("### Option Chain")
                    option_chain = get_option_chain(selected_ticker, expiration_date=expiration_date)
                    if not option_chain.empty:
                        st.dataframe(option_chain)
                    else:
                        st.warning("No option chain data available for this ticker.")

            else:
                st.warning("No option strategies found for the given tickers and expiration date.")

elif app_mode == "Stock Screener" and ADVANCED_FEATURES:
    st.title("Stock Screener")
    
    # Run the stock screener
    if st.button("Run Screener"):
        with st.spinner("Screening stocks..."):
            screened_stocks = screen_stocks()
            
            if not screened_stocks.empty:
                st.subheader("Screened Stocks")
                st.dataframe(screened_stocks)
                
                # Allow user to select stocks for further analysis
                selected_tickers = st.multiselect(
                    "Select stocks for further analysis",
                    options=screened_stocks['ticker'].tolist()
                )
                
                if selected_tickers:
                    st.write("### Technical Analysis for Selected Stocks")
                    for ticker in selected_tickers:
                        with st.spinner(f"Analyzing {ticker}..."):
                            analysis_result, data = analyze_stock_trend(ticker, lookback_days=1)
                            
                            # Display results in a styled card
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.subheader(f"Analysis Results for {ticker}")
                                st.metric("Current Price", f"${analysis_result['last_price']:.2f}")
                                st.metric("Trend Direction", analysis_result['trend_direction'])
                                st.metric("Trend Score", analysis_result['trend_score'])
                            
                            with col2:
                                st.subheader("Trading Signals")
                                st.metric("Entry Signal", "Yes" if analysis_result['entry_signal'] else "No")
                                st.metric("Entry Strength", f"{analysis_result['entry_strength']}/4")
                                st.metric("Recommendation", analysis_result['recommendation'], 
                                         delta="Buy Now" if analysis_result['recommendation'] == "BUY" else "Hold")
                            
                            # Create plotly charts
                            fig_price, fig_macd, fig_rsi = plot_analysis_plotly(data, ticker)
                            
                            # Display the charts
                            st.plotly_chart(fig_price, use_container_width=True)
                            
                            # Show MACD and RSI side by side
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(fig_macd, use_container_width=True)
                            with col2:
                                st.plotly_chart(fig_rsi, use_container_width=True)
            else:
                st.warning("No stocks matched the screening criteria.")

else:
    st.warning("Advanced features are disabled. Install the required dependencies to enable this mode.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This app provides advanced stock analysis tools, including technical analysis, options strategy evaluation, and stock screening.")
st.sidebar.markdown("**Disclaimer:** This app is for educational purposes only. Do not use this information for making investment decisions.")