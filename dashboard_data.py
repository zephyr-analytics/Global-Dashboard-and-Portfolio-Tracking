import yfinance as yf
import pandas as pd
import numpy as np
import yaml
import os
import glob
from typing import List, Dict, Any
from pandas.tseries.offsets import MonthEnd


class DashboardProcessor:
    """Processor for stock and ETF metadata, momentum snapshots, and ETF performance."""

    # -------------------- Data Fetching --------------------
    @staticmethod
    def fetch_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch adjusted close prices for tickers and filter those with <10 years of data."""
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            adj_close = pd.concat(
                {ticker: data[ticker]["Adj Close"] for ticker in tickers if "Adj Close" in data[ticker]},
                axis=1
            )
        else:
            adj_close = data[["Adj Close"]]
            adj_close.columns = tickers if len(tickers) == 1 else adj_close.columns

        # --- Filter tickers with at least 10 years of data ---
        min_days = 252 * 10
        keep = [col for col in adj_close.columns if adj_close[col].dropna().shape[0] >= min_days]
        filtered = adj_close[keep]

        dropped = set(tickers) - set(keep)
        if dropped:
            print(f"⚠️ Dropped {len(dropped)} tickers with <10 years data: {', '.join(dropped)}")

        return filtered

    # -------------------- Momentum --------------------
    @staticmethod
    def calculate_momentum_series(prices: pd.Series) -> pd.Series:
        """Calculate rolling average compounded returns over multiple periods for one ticker."""
        periods = [21, 63, 126, 189, 252]
        momentum = pd.Series(index=prices.index, dtype=float)

        for i in range(len(prices)):
            if i < max(periods):
                momentum.iloc[i] = np.nan
                continue

            returns = []
            for p in periods:
                start_price = prices.iloc[i - p]
                end_price = prices.iloc[i]
                returns.append((end_price / start_price) - 1)

            momentum.iloc[i] = np.mean(returns)

        return momentum

    @staticmethod
    def get_momentum_snapshot(price_df: pd.DataFrame) -> pd.DataFrame:
        """Build snapshot: Ticker, latest momentum, momentum 1M ago."""
        results = []
        for ticker in price_df.columns:
            mom_series = DashboardProcessor.calculate_momentum_series(price_df[ticker]).dropna()

            if mom_series.empty:
                latest, one_month_ago = (None, None)
            else:
                latest_date = mom_series.index[-1]
                latest = mom_series.iloc[-1]

                # Approx "1 month ago" trading date
                one_month_date = (latest_date - MonthEnd(1))
                one_month_idx = mom_series.index.searchsorted(one_month_date, side="right") - 1
                one_month_ago = mom_series.iloc[one_month_idx] if one_month_idx >= 0 else None

            results.append({
                "Ticker": ticker,
                "Momentum_Latest": latest,
                "Momentum_1M_Ago": one_month_ago
            })

        return pd.DataFrame(results)

    # -------------------- Stock Metadata --------------------
    @staticmethod
    def fetch_stock_data(tickers: List[str]) -> pd.DataFrame:
        """Fetch stock sector/financial info."""
        records = []
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                trailing_pe = info.get("trailingPE", None)
                forward_pe = info.get("forwardPE", None)

                if trailing_pe and forward_pe and forward_pe > 0:
                    expected_growth = (trailing_pe / forward_pe) - 1
                else:
                    expected_growth = None

                records.append({
                    "Ticker": ticker,
                    "Name": info.get("shortName", ""),
                    "Sector": info.get("sector", ""),
                    "Industry": info.get("industry", ""),
                    "Country": info.get("country", ""),
                    "Region": info.get("region", ""),
                    "MarketCap": info.get("marketCap", None),
                    "TrailingPE": trailing_pe,
                    "ForwardPE": forward_pe,
                    "ExpectedEarningsGrowth": expected_growth,
                    "TotalRevenue": info.get("totalRevenue", None),
                    "FreeCashflow": info.get("freeCashflow", None)
                })
            except Exception as e:
                print(f"⚠️ Error fetching stock {ticker}: {e}")
        return pd.DataFrame(records)

    # -------------------- ETF Metadata --------------------
    @staticmethod
    def fetch_etf_data(tickers: List[str]) -> pd.DataFrame:
        """Fetch ETF metadata (fund family, category, inception, yield, market cap)."""
        records = []
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                records.append({
                    "Ticker": ticker,
                    "Name": info.get("shortName", ""),
                    "Category": info.get("category", ""),
                    "FundFamily": info.get("fundFamily", ""),
                    "FundInceptionDate": info.get("fundInceptionDate", None),
                    "Yield": info.get("yield", None),
                    "MarketCap": info.get("marketCap", None),
                    "Country": info.get("country", ""),
                    "Region": info.get("region", "")
                })
            except Exception as e:
                print(f"⚠️ Error fetching ETF {ticker}: {e}")
        return pd.DataFrame(records)

    # -------------------- ETF Performance --------------------
    @staticmethod
    def get_etf_performance(price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ETF past year, past quarter, and YTD performance as %."""
        results = []
        if price_df.empty:
            return pd.DataFrame()

        latest_date = price_df.index[-1]
        one_year_ago = latest_date - pd.DateOffset(years=1)
        one_quarter_ago = latest_date - pd.DateOffset(months=3)
        year_start = pd.Timestamp(year=latest_date.year, month=1, day=1)

        for ticker in price_df.columns:
            series = price_df[ticker].dropna()
            if series.empty:
                continue

            latest = series.iloc[-1]

            # Indices for lookbacks
            one_year_idx = series.index.searchsorted(one_year_ago, side="left")
            one_quarter_idx = series.index.searchsorted(one_quarter_ago, side="left")
            ytd_idx = series.index.searchsorted(year_start, side="left")

            one_year_return, one_quarter_return, ytd_return = None, None, None

            if one_year_idx < len(series):
                one_year_return = (latest / series.iloc[one_year_idx]) - 1
            if one_quarter_idx < len(series):
                one_quarter_return = (latest / series.iloc[one_quarter_idx]) - 1
            if ytd_idx < len(series):
                ytd_return = (latest / series.iloc[ytd_idx]) - 1

            results.append({
                "Ticker": ticker,
                "1Y Return (%)": one_year_return * 100 if one_year_return is not None else None,
                "1Q Return (%)": one_quarter_return * 100 if one_quarter_return is not None else None,
                "YTD Return (%)": ytd_return * 100 if ytd_return is not None else None,
            })

        return pd.DataFrame(results)

    @staticmethod
    def get_etf_growth_series(price_df: pd.DataFrame) -> pd.DataFrame:
        """Return YTD cumulative growth over time as % (time series)."""
        if price_df.empty:
            return pd.DataFrame()

        latest_date = price_df.index[-1]
        year_start = pd.Timestamp(year=latest_date.year, month=1, day=1)

        # Slice YTD
        ytd_df = price_df.loc[price_df.index >= year_start].copy()
        if ytd_df.empty:
            return pd.DataFrame()

        growth_df = (ytd_df / ytd_df.iloc[0] - 1) * 100
        return growth_df

    # -------------------- Processor --------------------
    @staticmethod
    def process(config_path: str = "config.yaml"):
        """Main process pipeline: load config, fetch stocks and ETFs separately."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        stock_dir = config.get("stock_dir")
        etf_dir = config.get("etf_dir")
        ticker_column = config.get("ticker_column", "Ticker")
        start_date = config.get("start_date", "2000-01-01")
        end_date = config.get("end_date", "2025-01-01")
        pipeline = config.get("pipeline", "both").lower()

        os.makedirs("Artifacts", exist_ok=True)

        # -------------------- Stocks --------------------
        if pipeline in ["stocks", "both"] and stock_dir and os.path.isdir(stock_dir):
            stock_files = glob.glob(os.path.join(stock_dir, "*.csv"))
            for file in stock_files:
                print(f"\n🚀 Processing STOCK file: {file}")
                tickers = pd.read_csv(file)[ticker_column].dropna().unique().tolist()
                base, _ = os.path.splitext(os.path.basename(file))

                stock_df = DashboardProcessor.fetch_stock_data(tickers)
                stock_path = f"Artifacts/{base}_stocks.xlsx"
                stock_df.to_excel(stock_path, index=False, engine="openpyxl")
                print(f"✅ Stock data saved to {stock_path}")

                price_df = DashboardProcessor.fetch_data(tickers, start_date, end_date)
                snapshot_df = DashboardProcessor.get_momentum_snapshot(price_df)
                snapshot_path = f"Artifacts/{base}_momentum_snapshot.csv"
                snapshot_df.to_csv(snapshot_path, index=False)
                print(f"✅ Stock momentum snapshot saved to {snapshot_path}")

        # -------------------- ETFs --------------------
        if pipeline in ["etfs", "both"] and etf_dir and os.path.isdir(etf_dir):
            etf_files = glob.glob(os.path.join(etf_dir, "*.csv"))
            for file in etf_files:
                print(f"\n🚀 Processing ETF file: {file}")
                tickers = pd.read_csv(file)[ticker_column].dropna().unique().tolist()
                base, _ = os.path.splitext(os.path.basename(file))

                # Metadata
                etf_df = DashboardProcessor.fetch_etf_data(tickers)
                etf_path = f"Artifacts/{base}_etfs.xlsx"
                etf_df.to_excel(etf_path, index=False, engine="openpyxl")
                print(f"✅ ETF metadata saved to {etf_path}")

                # Price data + momentum
                price_df = DashboardProcessor.fetch_data(tickers, start_date, end_date)
                snapshot_df = DashboardProcessor.get_momentum_snapshot(price_df)
                snapshot_path = f"Artifacts/{base}_momentum_snapshot.csv"
                snapshot_df.to_csv(snapshot_path, index=False)
                print(f"✅ ETF momentum snapshot saved to {snapshot_path}")

                # ETF performance (1Y, 1Q, YTD)
                perf_df = DashboardProcessor.get_etf_performance(price_df)
                perf_path = f"Artifacts/{base}_etf_performance.csv"
                perf_df.to_csv(perf_path, index=False)
                print(f"✅ ETF performance saved to {perf_path}")

                # ETF growth time series (YTD)
                growth_df = DashboardProcessor.get_etf_growth_series(price_df)
                growth_path = f"Artifacts/{base}_etf_growth_series.csv"
                growth_df.to_csv(growth_path)
                print(f"✅ ETF growth time series saved to {growth_path}")


if __name__ == "__main__":
    DashboardProcessor.process()
