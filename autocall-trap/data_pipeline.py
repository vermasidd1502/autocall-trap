#!/usr/bin/env python3
"""
data_pipeline.py — Complete Data Pipeline (No WRDS Required)
=============================================================
Takes your manually-entered EDGAR term sheets and does everything else:

1. Pulls stock prices from Yahoo Finance (free, no API key)
2. Pulls dividends from Yahoo Finance
3. Pulls risk-free rates from FRED (free, no API key)
4. Estimates ATM implied vol from realized vol
5. Reconstructs the realized outcome for each note
6. Assembles notes_universe.csv
7. Runs the full SCP backtest

USAGE:
    # Step 1: Create term_sheets.csv from EDGAR (manual, ~30 min)
    # Step 2: Run this script
    python data_pipeline.py --input term_sheets.csv

    # Or use the included sample to test the pipeline
    python data_pipeline.py --demo

DEPENDENCIES (install these first):
    pip install yfinance pandas requests

INPUT FORMAT (term_sheets.csv):
    note_id,issuer,underlying,issue_date,S0,maturity,n_obs,coupon_rate,
    autocall_trigger,coupon_barrier,ki_barrier,memory,issuer_estimated_value

    Example row:
    GS-ORCL-2022-01,Goldman Sachs,ORCL,2022-03-15,140.00,2.0,8,0.02625,
    1.0,0.70,0.60,True,962.50

Author: Siddharth Verma
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Install required packages: pip install pandas numpy")
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("Install yfinance: pip install yfinance")
    print("This is required for free stock price / dividend data.")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class NoteTermSheet:
    """Raw term sheet from EDGAR (your manual input)."""
    note_id: str
    issuer: str
    underlying: str       # Ticker symbol
    issue_date: str       # YYYY-MM-DD
    S0: float             # Initial stock price
    maturity: float       # Years
    n_obs: int            # Number of observation dates
    coupon_rate: float    # Per-period, decimal
    autocall_trigger: float  # Fraction of S0 (e.g. 1.0)
    coupon_barrier: float    # Fraction of S0 (e.g. 0.70)
    ki_barrier: float        # Fraction of S0 (e.g. 0.60)
    memory: bool
    issuer_estimated_value: Optional[float] = None
    first_autocall_obs: int = 2  # Default: autocall starts at obs 2


@dataclass
class EnrichedNote:
    """Term sheet + market data + realized outcome."""
    # From term sheet
    note_id: str = ""
    issuer: str = ""
    underlying: str = ""
    issue_date: str = ""
    S0: float = 0.0
    par: float = 1000.0
    maturity: float = 0.0
    n_obs: int = 0
    coupon_rate: float = 0.0
    autocall_trigger: float = 0.0
    coupon_barrier: float = 0.0
    ki_barrier: float = 0.0
    memory: bool = True
    first_autocall_obs: int = 2
    issuer_estimated_value: Optional[float] = None

    # From Yahoo Finance / FRED
    risk_free_rate: float = 0.0
    atm_iv: float = 0.0
    div_yield: float = 0.0

    # Reconstructed outcome
    outcome: str = ""                    # autocalled, matured_par, matured_loss, still_live
    autocall_date: str = ""
    realized_payoff: float = 0.0
    realized_return: float = 0.0
    holding_period_years: float = 0.0

    # Observation details (for debugging)
    obs_dates: List[str] = field(default_factory=list)
    obs_prices: List[float] = field(default_factory=list)
    coupons_received: int = 0
    ki_breached: bool = False


# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD TERM SHEETS
# ══════════════════════════════════════════════════════════════════

def load_term_sheets(filepath: str) -> List[NoteTermSheet]:
    """Load manually-entered term sheets from CSV."""
    notes = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            note = NoteTermSheet(
                note_id=row['note_id'].strip(),
                issuer=row['issuer'].strip(),
                underlying=row['underlying'].strip().upper(),
                issue_date=row['issue_date'].strip(),
                S0=float(row['S0']),
                maturity=float(row['maturity']),
                n_obs=int(row['n_obs']),
                coupon_rate=float(row['coupon_rate']),
                autocall_trigger=float(row['autocall_trigger']),
                coupon_barrier=float(row['coupon_barrier']),
                ki_barrier=float(row['ki_barrier']),
                memory=row.get('memory', 'True').strip().lower() in ('true', '1', 'yes'),
                issuer_estimated_value=_float_or_none(row.get('issuer_estimated_value', '')),
                first_autocall_obs=int(row.get('first_autocall_obs', 2)),
            )
            notes.append(note)
    print(f"  Loaded {len(notes)} term sheets from {filepath}")
    return notes


def _float_or_none(val):
    if val is None or val.strip() == '' or val.strip().upper() == 'NA':
        return None
    return float(val.strip())


# ══════════════════════════════════════════════════════════════════
# STEP 2: PULL YAHOO FINANCE DATA
# ══════════════════════════════════════════════════════════════════

def pull_yahoo_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Pull daily prices and dividends from Yahoo Finance.
    Returns dict of {ticker: DataFrame with columns [Close, Dividends]}.
    """
    data = {}
    for ticker in tickers:
        print(f"  Pulling {ticker} from Yahoo Finance...", end=" ")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=False)
            if len(hist) == 0:
                print(f"NO DATA")
                continue
            # Keep just Close and Dividends
            df = pd.DataFrame({
                'Close': hist['Close'],
                'Dividends': hist['Dividends'] if 'Dividends' in hist.columns else 0,
            })
            df.index = pd.to_datetime(df.index).tz_localize(None)
            data[ticker] = df
            print(f"{len(df)} days")
        except Exception as e:
            print(f"ERROR: {e}")
    return data


def get_price_on_date(df: pd.DataFrame, target_date: str) -> Optional[float]:
    """Get the closing price on or nearest before target_date."""
    target = pd.Timestamp(target_date)
    # Find nearest date at or before target
    valid = df[df.index <= target]
    if len(valid) == 0:
        # Try nearest after
        valid = df[df.index >= target]
        if len(valid) == 0:
            return None
        return float(valid.iloc[0]['Close'])
    return float(valid.iloc[-1]['Close'])


def estimate_realized_vol(df: pd.DataFrame, date: str, window: int = 60) -> float:
    """
    Estimate annualized realized vol using trailing daily returns.
    This is the free proxy for ATM implied vol.
    """
    target = pd.Timestamp(date)
    mask = df.index <= target
    recent = df[mask].tail(window)
    if len(recent) < 20:
        return 0.25  # Fallback
    log_returns = np.log(recent['Close'] / recent['Close'].shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))


def estimate_div_yield(df: pd.DataFrame, date: str, price: float) -> float:
    """Estimate trailing 12-month dividend yield."""
    target = pd.Timestamp(date)
    one_year_ago = target - timedelta(days=365)
    mask = (df.index >= one_year_ago) & (df.index <= target)
    annual_div = df[mask]['Dividends'].sum()
    if price <= 0:
        return 0.0
    return float(annual_div / price)


# ══════════════════════════════════════════════════════════════════
# STEP 3: PULL RISK-FREE RATES FROM FRED
# ══════════════════════════════════════════════════════════════════

def pull_fred_rates(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull 2-year Treasury yield from FRED (free, no API key needed).
    Falls back to a flat rate if the request fails.
    """
    print("  Pulling 2-year Treasury from FRED...", end=" ")
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans"
        f"&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on"
        f"&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0"
        f"&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes"
        f"&id=DGS2&scale=left&cosd={start_date}&coed={end_date}"
        f"&line_color=%234572a7&link_values=false&line_style=solid"
        f"&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0"
        f"&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01"
        f"&line_index=1&transformation=lin&vintage_date={end_date}"
        f"&revision_date={end_date}&nd=2000-01-01"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            data = []
            for line in lines[1:]:  # Skip header
                parts = line.split(',')
                if len(parts) == 2 and parts[1] != '.':
                    try:
                        data.append({
                            'date': pd.Timestamp(parts[0]),
                            'rate': float(parts[1]) / 100,  # Convert to decimal
                        })
                    except (ValueError, IndexError):
                        continue
            df = pd.DataFrame(data).set_index('date')
            print(f"{len(df)} observations")
            return df
    except Exception as e:
        print(f"FAILED ({e})")

    print("  Using fallback flat rate of 4.5%")
    return pd.DataFrame()


def get_rate_on_date(rates_df: pd.DataFrame, target_date: str, fallback: float = 0.045) -> float:
    """Get the 2-year Treasury yield on or nearest before target_date."""
    if rates_df is None or len(rates_df) == 0:
        return fallback
    target = pd.Timestamp(target_date)
    valid = rates_df[rates_df.index <= target]
    if len(valid) == 0:
        return fallback
    return float(valid.iloc[-1]['rate'])


# ══════════════════════════════════════════════════════════════════
# STEP 4: RECONSTRUCT OUTCOMES
# ══════════════════════════════════════════════════════════════════

def reconstruct_outcome(
    term_sheet: NoteTermSheet,
    price_data: pd.DataFrame,
    risk_free_rate: float,
    atm_iv: float,
    div_yield: float,
) -> EnrichedNote:
    """
    Given a term sheet and daily prices, walk through each observation
    date and determine exactly what happened.
    """
    note = EnrichedNote(
        note_id=term_sheet.note_id,
        issuer=term_sheet.issuer,
        underlying=term_sheet.underlying,
        issue_date=term_sheet.issue_date,
        S0=term_sheet.S0,
        maturity=term_sheet.maturity,
        n_obs=term_sheet.n_obs,
        coupon_rate=term_sheet.coupon_rate,
        autocall_trigger=term_sheet.autocall_trigger,
        coupon_barrier=term_sheet.coupon_barrier,
        ki_barrier=term_sheet.ki_barrier,
        memory=term_sheet.memory,
        first_autocall_obs=term_sheet.first_autocall_obs,
        issuer_estimated_value=term_sheet.issuer_estimated_value,
        risk_free_rate=risk_free_rate,
        atm_iv=atm_iv,
        div_yield=div_yield,
    )

    # Generate observation dates (evenly spaced)
    issue = pd.Timestamp(term_sheet.issue_date)
    dt_months = int(12 * term_sheet.maturity / term_sheet.n_obs)
    obs_dates = []
    for i in range(1, term_sheet.n_obs + 1):
        obs_date = issue + pd.DateOffset(months=dt_months * i)
        obs_dates.append(obs_date)

    maturity_date = obs_dates[-1]

    # Barrier levels in dollar terms
    autocall_level = term_sheet.autocall_trigger * term_sheet.S0
    coupon_level = term_sheet.coupon_barrier * term_sheet.S0
    ki_level = term_sheet.ki_barrier * term_sheet.S0
    coupon_dollar = term_sheet.coupon_rate * 1000.0

    # Walk through observation dates
    terminated = False
    ki_breached = False
    unpaid_coupons = 0
    total_coupons = 0.0
    obs_price_list = []

    for obs_idx, obs_date in enumerate(obs_dates):
        obs_num = obs_idx + 1  # 1-indexed

        # Get price on or nearest to observation date
        price = get_price_on_date(price_data, obs_date.strftime('%Y-%m-%d'))
        if price is None:
            # If we can't find a price, the note might still be live
            note.outcome = "still_live"
            note.obs_dates = [d.strftime('%Y-%m-%d') for d in obs_dates[:obs_idx]]
            note.obs_prices = obs_price_list
            return note

        obs_price_list.append(price)
        note.obs_dates.append(obs_date.strftime('%Y-%m-%d'))
        note.obs_prices.append(price)

        # Check knock-in
        if price < ki_level:
            ki_breached = True

        # Check coupon
        if price >= coupon_level:
            if term_sheet.memory:
                total_coupons += (1 + unpaid_coupons) * coupon_dollar
                note.coupons_received += 1 + int(unpaid_coupons)
                unpaid_coupons = 0
            else:
                total_coupons += coupon_dollar
                note.coupons_received += 1
        else:
            unpaid_coupons += 1

        # Check autocall (starting from first_autocall_obs)
        if obs_num >= term_sheet.first_autocall_obs and price >= autocall_level:
            note.outcome = "autocalled"
            note.autocall_date = obs_date.strftime('%Y-%m-%d')
            note.realized_payoff = 1000.0 + total_coupons
            note.holding_period_years = obs_num * (term_sheet.maturity / term_sheet.n_obs)
            note.realized_return = (note.realized_payoff - 1000.0) / 1000.0
            note.ki_breached = ki_breached
            return note

    # Reached maturity without autocall
    note.ki_breached = ki_breached
    note.holding_period_years = term_sheet.maturity

    final_price = obs_price_list[-1] if obs_price_list else term_sheet.S0

    if ki_breached:
        # Investor bears the loss
        terminal_value = 1000.0 * min(final_price / term_sheet.S0, 1.0)
        note.outcome = "matured_loss"
        note.realized_payoff = terminal_value + total_coupons
    else:
        note.outcome = "matured_par"
        note.realized_payoff = 1000.0 + total_coupons

    note.realized_return = (note.realized_payoff - 1000.0) / 1000.0
    return note


# ══════════════════════════════════════════════════════════════════
# STEP 5: ASSEMBLE AND EXPORT
# ══════════════════════════════════════════════════════════════════

def export_notes_csv(notes: List[EnrichedNote], filepath: str):
    """Export to the format expected by the backtest engine."""
    fieldnames = [
        'note_id', 'issuer', 'underlying', 'issue_date', 'S0', 'par',
        'maturity', 'n_obs', 'coupon_rate', 'autocall_trigger',
        'coupon_barrier', 'ki_barrier', 'memory', 'first_autocall_obs',
        'risk_free_rate', 'atm_iv', 'div_yield', 'issuer_estimated_value',
        'outcome', 'realized_payoff', 'realized_return', 'holding_period_years',
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n in notes:
            row = {
                'note_id': n.note_id,
                'issuer': n.issuer,
                'underlying': n.underlying,
                'issue_date': n.issue_date,
                'S0': n.S0,
                'par': n.par,
                'maturity': n.maturity,
                'n_obs': n.n_obs,
                'coupon_rate': n.coupon_rate,
                'autocall_trigger': n.autocall_trigger,
                'coupon_barrier': n.coupon_barrier,
                'ki_barrier': n.ki_barrier,
                'memory': n.memory,
                'first_autocall_obs': n.first_autocall_obs,
                'risk_free_rate': round(n.risk_free_rate, 5),
                'atm_iv': round(n.atm_iv, 4),
                'div_yield': round(n.div_yield, 4),
                'issuer_estimated_value': n.issuer_estimated_value or '',
                'outcome': n.outcome,
                'realized_payoff': round(n.realized_payoff, 2),
                'realized_return': round(n.realized_return, 4),
                'holding_period_years': round(n.holding_period_years, 2),
            }
            writer.writerow(row)
    print(f"\n  Exported {len(notes)} notes to {filepath}")


def print_summary(notes: List[EnrichedNote]):
    """Print a summary table of all notes and their outcomes."""
    print(f"\n{'OUTCOME SUMMARY':=^70}")
    print(f"  {'ID':<20} {'Ticker':<6} {'Outcome':<15} {'Payoff':>8} {'Return':>8} {'KI':>4}")
    print(f"  {'-'*20} {'-'*6} {'-'*15} {'-'*8} {'-'*8} {'-'*4}")
    for n in notes:
        ki_flag = "YES" if n.ki_breached else ""
        print(f"  {n.note_id:<20} {n.underlying:<6} {n.outcome:<15} "
              f"${n.realized_payoff:>7.0f} {n.realized_return*100:>7.1f}% {ki_flag:>4}")

    outcomes = [n.outcome for n in notes]
    print(f"\n  Autocalled:     {outcomes.count('autocalled')}")
    print(f"  Matured (par):  {outcomes.count('matured_par')}")
    print(f"  Matured (loss): {outcomes.count('matured_loss')}")
    print(f"  Still live:     {outcomes.count('still_live')}")
    avg_return = np.mean([n.realized_return for n in notes if n.outcome != 'still_live']) * 100
    print(f"  Avg return:     {avg_return:.1f}%")


# ══════════════════════════════════════════════════════════════════
# DEMO: SAMPLE TERM SHEETS
# ══════════════════════════════════════════════════════════════════

DEMO_CSV = """note_id,issuer,underlying,issue_date,S0,maturity,n_obs,coupon_rate,autocall_trigger,coupon_barrier,ki_barrier,memory,issuer_estimated_value,first_autocall_obs
GS-ORCL-2201,Goldman Sachs,ORCL,2022-01-15,86.50,2.0,8,0.025,1.0,0.70,0.60,True,965.00,2
GS-AAPL-2201,Goldman Sachs,AAPL,2022-01-20,164.50,1.5,6,0.018,1.0,0.75,0.65,True,970.00,2
HSBC-NVDA-2206,HSBC,NVDA,2022-06-15,165.00,2.0,8,0.035,1.0,0.70,0.55,True,945.00,2
JPM-MSFT-2203,JP Morgan,MSFT,2022-03-10,295.00,2.0,8,0.020,1.0,0.75,0.65,True,972.00,2
MS-TSLA-2204,Morgan Stanley,TSLA,2022-04-20,1005.00,1.5,6,0.045,1.0,0.65,0.55,True,930.00,2
GS-META-2209,Goldman Sachs,META,2022-09-15,149.00,2.0,8,0.038,1.0,0.70,0.60,True,940.00,2
HSBC-AMZN-2201,HSBC,AMZN,2022-01-10,164.00,2.0,8,0.022,1.0,0.75,0.65,True,968.00,2
JPM-JPM-2206,JP Morgan,JPM,2022-06-01,125.00,2.0,8,0.023,1.0,0.70,0.60,True,960.00,2
GS-GOOG-2203,Goldman Sachs,GOOG,2022-03-01,135.00,1.5,6,0.020,1.0,0.75,0.65,True,971.00,2
MS-ORCL-2209,Morgan Stanley,ORCL,2022-09-20,67.00,2.0,8,0.030,1.0,0.70,0.60,True,955.00,2
HSBC-AAPL-2206,HSBC,AAPL,2022-06-10,142.00,2.0,8,0.020,1.0,0.75,0.65,True,970.00,2
GS-NVDA-2203,Goldman Sachs,NVDA,2022-03-15,250.00,2.0,8,0.040,1.0,0.65,0.55,True,935.00,2
"""


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_pipeline(input_file: str, output_file: str = "notes_universe.csv"):
    print("=" * 60)
    print("AUTOCALL TRAP — DATA PIPELINE")
    print("=" * 60)

    # ── Load term sheets ──
    print("\n[1/5] Loading term sheets...")
    term_sheets = load_term_sheets(input_file)

    # ── Determine date range and unique tickers ──
    tickers = list(set(ts.underlying for ts in term_sheets))
    earliest = min(ts.issue_date for ts in term_sheets)
    latest_issue = max(ts.issue_date for ts in term_sheets)
    # Need prices from 60 days before earliest (for vol estimation)
    # through maturity of latest note
    start = (pd.Timestamp(earliest) - timedelta(days=90)).strftime('%Y-%m-%d')
    end = (pd.Timestamp(latest_issue) + timedelta(days=int(365 * 2.5))).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    if end > today:
        end = today

    print(f"  Tickers: {tickers}")
    print(f"  Date range: {start} to {end}")

    # ── Pull Yahoo Finance data ──
    print(f"\n[2/5] Pulling stock prices and dividends...")
    yahoo_data = pull_yahoo_data(tickers, start, end)

    # ── Pull risk-free rates ──
    print(f"\n[3/5] Pulling risk-free rates from FRED...")
    rates_df = pull_fred_rates(start, end)

    # ── Reconstruct outcomes ──
    print(f"\n[4/5] Reconstructing outcomes...")
    enriched_notes = []
    for ts in term_sheets:
        ticker = ts.underlying
        if ticker not in yahoo_data:
            print(f"  SKIP: {ts.note_id} — no price data for {ticker}")
            continue

        price_df = yahoo_data[ticker]

        # Get market data on issue date
        rfr = get_rate_on_date(rates_df, ts.issue_date)
        atm_iv = estimate_realized_vol(price_df, ts.issue_date, window=60)
        div_y = estimate_div_yield(price_df, ts.issue_date, ts.S0)

        # Reconstruct
        note = reconstruct_outcome(ts, price_df, rfr, atm_iv, div_y)
        enriched_notes.append(note)

        status = f"{note.outcome}"
        if note.outcome == 'autocalled':
            status += f" on {note.autocall_date}"
        print(f"  {note.note_id}: {status}, payoff=${note.realized_payoff:.0f} "
              f"({note.realized_return*100:+.1f}%)")

    # ── Export ──
    print(f"\n[5/5] Exporting...")
    export_notes_csv(enriched_notes, output_file)
    print_summary(enriched_notes)

    print(f"\n{'NEXT STEP':=^60}")
    print(f"  Run the backtest:")
    print(f"  python -m src.backtest --csv {output_file} --paths 100000")
    print(f"{'='*60}")

    return enriched_notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autocall Trap Data Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Path to term_sheets.csv")
    parser.add_argument("--output", type=str, default="notes_universe.csv", help="Output CSV path")
    parser.add_argument("--demo", action="store_true", help="Run with sample data")
    args = parser.parse_args()

    if args.demo:
        # Write demo CSV to temp file
        demo_path = "data/demo_term_sheets.csv"
        os.makedirs("data", exist_ok=True)
        with open(demo_path, 'w') as f:
            f.write(DEMO_CSV.strip())
        run_pipeline(demo_path, args.output)
    elif args.input:
        run_pipeline(args.input, args.output)
    else:
        print("Usage:")
        print("  python data_pipeline.py --demo              # Run with sample notes")
        print("  python data_pipeline.py --input my_notes.csv # Run with your EDGAR data")
        print()
        print("Create term_sheets.csv with columns:")
        print("  note_id, issuer, underlying, issue_date, S0, maturity, n_obs,")
        print("  coupon_rate, autocall_trigger, coupon_barrier, ki_barrier,")
        print("  memory, issuer_estimated_value, first_autocall_obs")
