import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def correlation_function(series, max_lag=20):
    """
    Compute the auto-correlation function of a time series for lags 0..max_lag.
    """
    if len(series) < max_lag + 1:
        return np.ones(max_lag + 1)
    result = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            result[lag] = 1.0
        else:
            result[lag] = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            if np.isnan(result[lag]):
                result[lag] = 0.0
    return result

def response_function(series, max_lag=20):
    """
    Compute the response function (impulse response) from the auto-correlation.
    For a linear system, the response function is the derivative of the correlation.
    """
    corr = correlation_function(series, max_lag)
    if len(corr) < 2:
        return np.zeros(max_lag + 1)
    # Derivative of correlation
    response = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        if lag < len(corr) - 1:
            response[lag] = -(corr[lag+1] - corr[lag-1]) / 2.0
        else:
            response[lag] = -(corr[lag] - corr[lag-1])
    return response

def fluctuation_dissipation_ratio(series, max_lag=20):
    """
    Compute the fluctuation-dissipation ratio (FDR) = (fluctuation) / (dissipation).
    FDR = 1 in equilibrium; > 1 indicates out-of-equilibrium (larger fluctuations).
    """
    if len(series) < max_lag + 2:
        return 0.0
    corr = correlation_function(series, max_lag)
    resp = response_function(series, max_lag)
    # Fluctuation: integrated auto-correlation (area under correlation)
    fluct = np.sum(corr[1:]) / max_lag
    # Dissipation: integrated response (area under response)
    diss = np.sum(resp[1:]) / max_lag
    if diss == 0 or np.isnan(diss):
        return 1.0
    fdr = fluct / diss
    return float(fdr)

def fluctuation_dissipation_score(returns, macro_df, max_lag=20):
    """
    Compute per-ETF FDR score.
    Higher FDR = farther from equilibrium = more unstable regime.
    """
    if len(returns) < max_lag + 2 or macro_df is None or len(macro_df) < max_lag + 2:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < max_lag + 2:
        return 0.0
    # Compute FDR
    fdr = fluctuation_dissipation_ratio(returns, max_lag)
    # Use macro factor to modulate the score
    macro_factor = compute_composite_macro_factor(macro_df)[-1]
    # Higher macro factor amplifies the FDR signal
    fdr_adjusted = fdr * (1 + macro_factor * 0.5)
    return float(fdr_adjusted)
