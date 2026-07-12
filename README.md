# Fluctuation-Dissipation Ratio for ETFs

Measures the effective temperature and fluctuation-dissipation ratio to quantify how far from equilibrium the market is. The Fluctuation-Dissipation Ratio (FDR) = (fluctuation) / (dissipation). FDR = 1 in equilibrium; > 1 indicates out‑of‑equilibrium (larger fluctuations). Higher FDR → regime destabilization.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Auto‑correlation and response functions
- Integrated fluctuation and dissipation
- Macro‑modulated FDR score
- Score = FDR (higher = farther from equilibrium)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-fluctuation-dissipation-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High FDR → market is out of equilibrium → regime destabilization.
- Low FDR → near equilibrium → stable regime.

## Requirements

See `requirements.txt`.
