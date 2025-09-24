import os
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import yaml

class YieldCurveProcessor:

    GLOBAL_10Y = {
        "Germany": "IRLTLT01DEM156N",
        "France": "IRLTLT01FRM156N",
        "Japan": "IRLTLT01JPM156N",
        "UK": "IRLTLT01GBM156N",
        "Switzerland": "IRLTLT01CHM156N",
        "Canada": "IRLTLT01CAM156N",
        "Australia": "IRLTLT01AUM156N",
    }

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.output_dir = self.config.get("fed_output_dir", "fed_files")
        os.makedirs(self.output_dir, exist_ok=True)

        today = datetime.strptime(self.config.get("end_date"), "%Y-%m-%d").date()
        self.date_labels = {
            "Today": today,
            "6M_Ago": today - timedelta(days=182),
            "1Y_Ago": today - timedelta(days=365),
        }

    def fetch_value(self, series_id, target_date, mode="nearest"):
        """
        Fetch a value for a FRED series around a date.
        mode="nearest"  -> pick closest available (for daily US yields)
        mode="backfill" -> pick last available before target (for global series)
        """
        # Always fetch at least 2 years of data
        start = target_date - timedelta(days=730)
        end = datetime.today() + timedelta(days=5)
        data = pdr.DataReader(series_id, "fred", start, end)
        if data.empty:
            return None

        if mode == "nearest":
            diffs = np.abs((data.index - pd.Timestamp(target_date)).days)
            nearest_idx = diffs.argmin()
            return data.iloc[nearest_idx, 0]

        elif mode == "backfill":
            subset = data.loc[:pd.Timestamp(target_date)]
            if subset.empty:
                return None
            return subset.iloc[-1, 0]

    def process_yield_curve(self):
        """Download US Treasury yield curve snapshots + monthly averages for Apr 2000 & Jan 2007."""
        end = datetime.today() + timedelta(days=5)
        start = "1999-01-01"  # enough history

        maturities = {
            "001-M": "DGS1MO",
            "003-M": "DGS3MO",
            "006-M": "DGS6MO",
            "01-Y": "DGS1",
            "02-Y": "DGS2",
            "03-Y": "DGS3",
            "05-Y": "DGS5",
            "07-Y": "DGS7",
            "10-Y": "DGS10",
            "20-Y": "DGS20",
            "30-Y": "DGS30",
        }

        # --- Snapshots (nearest available to target dates) ---
        target_dates = {
            "Today": datetime.today(),
            "6M_Ago": datetime.today() - timedelta(days=182),
            "1Y_Ago": datetime.today() - timedelta(days=365),
        }

        results = {mat: {} for mat in maturities}

        for label, target in target_dates.items():
            for mat, fred_id in maturities.items():
                try:
                    data = pdr.DataReader(fred_id, "fred", start, end)
                    idx = (np.abs((data.index - target).days)).argmin()
                    results[mat][label] = data.iloc[idx, 0]
                except Exception:
                    results[mat][label] = None

        # --- Monthly averages for Apr 2000 and Jan 2007 ---
        month_windows = {
            "Apr2000": ("2000-04-01", "2000-04-30"),
            "Jan2007": ("2007-01-01", "2007-01-31"),
            "Jul2023": ("2023-07-01", "2023-07-31"),
        }

        for label, (start_date, end_date) in month_windows.items():
            for mat, fred_id in maturities.items():
                try:
                    data = pdr.DataReader(fred_id, "fred", start_date, end_date)
                    results[mat][label] = data[fred_id].mean()
                except Exception:
                    results[mat][label] = None

        # --- Convert to DataFrame with maturities as rows ---
        df = pd.DataFrame(results).T
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Maturity"}, inplace=True)

        file_path = os.path.join(self.output_dir, "yield_curves.csv")
        df.to_csv(file_path, index=False)
        print(f"✅ Saved yield curve snapshots + monthly averages to {file_path}")


    def process_global_10y(self):
        """
        Process and save global 10-year bond yields.
        """
        rows = []
        for country, fred_id in self.GLOBAL_10Y.items():
            row = {"Country": country}
            for label, date in self.date_labels.items():
                try:
                    value = self.fetch_value(fred_id, date)
                    row[label] = value
                except Exception as e:
                    print(f"⚠️ Could not fetch {fred_id} on {date}: {e}")
                    row[label] = None
            rows.append(row)

        df = pd.DataFrame(rows)
        file_path = os.path.join(self.output_dir, "global_10y.csv")
        df.to_csv(file_path, index=False)
        print(f"✅ Saved global 10Y yields to {file_path}")

    def process_global_10y_history(self):
        """Download full historical global 10Y yields in wide monthly time series format."""
        end = datetime.today() + timedelta(days=5)

        # Only global 10Y series (no US)
        df_all = pd.DataFrame()

        for country, fred_id in self.GLOBAL_10Y.items():
            try:
                data = pdr.DataReader(fred_id, "fred", start="2000-01-01", end=end)

                # Resample to monthly (end-of-month) just in case
                data = data.resample("M").last()

                data = data.rename(columns={fred_id: country})

                if df_all.empty:
                    df_all = data
                else:
                    df_all = df_all.join(data, how="outer")

            except Exception as e:
                print(f"⚠️ Could not fetch {fred_id} for {country}: {e}")

        # Reset index so Date becomes a column
        df_all.reset_index(inplace=True)
        df_all.rename(columns={"index": "Date"}, inplace=True)

        file_path = os.path.join(self.output_dir, "global_10y_history.csv")
        df_all.to_csv(file_path, index=False)
        print(f"✅ Saved historical global 10Y monthly time series to {file_path}")


    def process_t10y2y(self):
        """Download 10Y-2Y Treasury spread and save as CSV."""
        end = datetime.today() + timedelta(days=5)
        start = "1990-01-01"   # plenty of history

        try:
            spread = pdr.DataReader("T10Y2Y", "fred", start, end)

            # Optional: resample to monthly averages for smoother series
            spread_monthly = spread.resample("M").mean()

            # Reset index for CSV
            spread_monthly.reset_index(inplace=True)
            spread_monthly.rename(columns={"index": "Date", "T10Y2Y": "Spread"}, inplace=True)

            file_path = os.path.join(self.output_dir, "t10y2y.csv")
            spread_monthly.to_csv(file_path, index=False)
            print(f"✅ Saved 10Y-2Y Treasury spread to {file_path}")

        except Exception as e:
            print(f"⚠️ Could not fetch T10Y2Y spread: {e}")

    def process_inflation(self):
        """Download inflation-related series (Oil + 5Y Breakeven) and save as CSV."""
        end = datetime.today() + timedelta(days=5)
        start = "2000-01-01"   # covers both series

        series_map = {
            "WTI_Crude": "DCOILWTICO",
            "Breakeven_5Y": "T5YIE"
        }

        df_all = pd.DataFrame()

        for name, fred_id in series_map.items():
            try:
                data = pdr.DataReader(fred_id, "fred", start, end)

                # Resample to monthly averages for consistency
                data = data.resample("M").mean()
                data = data.rename(columns={fred_id: name})

                if df_all.empty:
                    df_all = data
                else:
                    df_all = df_all.join(data, how="outer")

            except Exception as e:
                print(f"⚠️ Could not fetch {fred_id} ({name}): {e}")

        df_all.reset_index(inplace=True)
        df_all.rename(columns={"index": "Date"}, inplace=True)

        file_path = os.path.join(self.output_dir, "inflation.csv")
        df_all.to_csv(file_path, index=False)
        print(f"✅ Saved inflation series to {file_path}")


    def process_employment(self):
        """
        Download employment-related series from FRED (2000+):
        - Nonfarm Payrolls (PAYEMS)
        - Unemployment Rate (UNRATE)
        - ADP Employment (ADPWNUSNERSA)
        - Job Openings (JTSJOL)
        - Hires (JTSHIR)
        - Separations (JTSTSL)
        - Quits (JTSQUR)
        - Layoffs & Discharges (JTSLDL)
        """
        end = datetime.today() + timedelta(days=5)
        start = "2000-01-01"  # begin at 2000 for JOLTS alignment

        series = {
            "Nonfarm_Payrolls": "PAYEMS",
            "Unemployment_Rate": "UNRATE",
            "ADP_Employment": "ADPWNUSNERSA",
            "Job_Openings": "JTSJOL",
            "Hires": "JTSHIR",
            "Separations": "JTSTSL",
            "Quits": "JTSQUR",
            "Layoffs_Discharges": "JTSLDL",
        }

        data_frames = []
        for label, fred_id in series.items():
            try:
                df = pdr.DataReader(fred_id, "fred", start, end)
                df.rename(columns={fred_id: label}, inplace=True)
                data_frames.append(df)
            except Exception as e:
                print(f"⚠️ Could not fetch {label} ({fred_id}): {e}")

        if data_frames:
            combined = pd.concat(data_frames, axis=1)

            # Reset index for clean CSV
            combined.reset_index(inplace=True)
            combined.rename(columns={"index": "Date"}, inplace=True)

            file_path = os.path.join(self.output_dir, "employment.csv")
            combined.to_csv(file_path, index=False)
            print(f"✅ Saved employment data (2000+) to {file_path}")
        else:
            print("⚠️ No employment data fetched.")


    def process(self):
        self.process_yield_curve()
        self.process_global_10y()
        self.process_global_10y_history()
        self.process_t10y2y()
        self.process_inflation()
        self.process_employment()


if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    processor = YieldCurveProcessor(CONFIG_PATH)
    processor.process()
