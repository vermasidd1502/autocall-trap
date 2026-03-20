#!/usr/bin/env python3
"""
edgar_extractor.py — Paste a URL, get a CSV row
=================================================
Feed it EDGAR 424B2 filing URLs one at a time (or in batch).
It scrapes the HTML, extracts every term sheet field, and appends
to your growing CSV. You fill in S0 manually afterward.

USAGE:
    # Interactive mode — paste URLs one by one
    python edgar_extractor.py

    # Single URL
    python edgar_extractor.py --url "https://www.sec.gov/Archives/edgar/data/..."

    # Batch mode — file with one URL per line
    python edgar_extractor.py --batch urls.txt

    # Specify output file
    python edgar_extractor.py --output my_notes.csv

DEPENDENCIES:
    pip install requests beautifulsoup4 lxml

Author: Siddharth Verma
"""

import re
import os
import csv
import sys
import time
import argparse
from typing import Optional, List, Tuple
from dataclasses import dataclass, asdict, fields

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Install dependencies: pip install requests beautifulsoup4 lxml")
    sys.exit(1)


# SEC requires a user-agent with contact info
HEADERS = {
    "User-Agent": "SiddharthVerma research@university.edu",
    "Accept": "text/html",
}


@dataclass
class ExtractedNote:
    note_id: str = ""
    issuer: str = ""
    underlying: str = ""
    underlying_ticker: str = ""
    issue_date: str = ""          # pricing/trade date
    settlement_date: str = ""
    maturity_date: str = ""
    S0: str = "FILL_ME"           # you fill this in
    maturity: float = 0.0
    n_obs: int = 0
    coupon_rate: float = 0.0      # per period, decimal
    coupon_annual_pct: float = 0.0
    autocall_trigger: float = 1.0
    coupon_barrier: float = 0.0
    ki_barrier: float = 0.0
    memory: bool = False
    first_autocall_obs: int = 1
    issuer_estimated_value: float = 0.0
    par: float = 1000.0
    source_url: str = ""
    cusip: str = ""
    confidence: str = "low"
    notes: str = ""               # extraction notes / warnings


def fetch_filing(url: str) -> Optional[str]:
    """Download filing HTML."""
    print(f"  Fetching: {url[:80]}...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def extract_term_sheet(html: str, url: str) -> Optional[ExtractedNote]:
    """Parse a 424B2 filing and extract all term sheet fields."""
    soup = BeautifulSoup(html, 'lxml')
    text = soup.get_text(separator=' ', strip=True)
    text_lower = text.lower()

    # ── Quick filter: is this an autocallable? ──
    if not any(kw in text_lower for kw in ['autocall', 'auto-call', 'automatic call']):
        print("  SKIP: Not an autocallable note.")
        return None

    # ── Check for worst-of basket (skip these) ──
    multi_underlier_signals = [
        'least performing', 'lesser performing', 'worst performing',
        'each of the', 'each underlier', 'each index',
    ]
    # Count how many distinct index/stock names appear
    index_names = ['russell 2000', 's&p 500', 'dow jones', 'nasdaq',
                   'euro stoxx', 'nikkei', 'ftse']
    index_count = sum(1 for idx in index_names if idx in text_lower)

    is_basket = (index_count >= 2 or
                 any(sig in text_lower for sig in multi_underlier_signals))

    note = ExtractedNote(source_url=url)

    if is_basket:
        note.notes = "WARNING: Appears to be worst-of basket. Review manually."
        print(f"  WARNING: Worst-of basket detected ({index_count} indices). Extracting anyway but flagged.")

    # ── Issuer ──
    issuer_patterns = [
        (r'GS Finance Corp', 'Goldman Sachs'),
        (r'Goldman Sachs', 'Goldman Sachs'),
        (r'JPMorgan Chase Financial', 'JP Morgan'),
        (r'JPMorgan Financial', 'JP Morgan'),
        (r'J\.?P\.?\s*Morgan', 'JP Morgan'),
        (r'HSBC USA', 'HSBC'),
        (r'Morgan Stanley Finance', 'Morgan Stanley'),
        (r'Morgan Stanley', 'Morgan Stanley'),
        (r'Barclays Bank', 'Barclays'),
        (r'Citigroup Global Markets', 'Citigroup'),
        (r'Citigroup', 'Citigroup'),
        (r'Bank of America', 'Bank of America'),
        (r'Royal Bank of Canada', 'RBC'),
        (r'UBS AG', 'UBS'),
    ]
    for pattern, name in issuer_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            note.issuer = name
            break

    # ── Underlying ──
    # Try "common stock of [Company]"
    stock_match = re.search(
        r'(?:common\s+stock|shares)\s+of\s+([A-Z][A-Za-z\s&.,\']+?)(?:\s*\(|due|$)',
        text
    )
    if stock_match:
        note.underlying = stock_match.group(1).strip().rstrip('.,')

    # Try Bloomberg ticker
    ticker_match = re.search(
        r'(?:Bloomberg\s+ticker|ticker)[:\s]+["\']?([A-Z]{1,5})\b',
        text
    )
    if ticker_match:
        note.underlying_ticker = ticker_match.group(1)

    # Try to find ticker in parentheses after company name
    if not note.underlying_ticker:
        paren_ticker = re.search(
            r'(?:common stock of.*?)\((?:NYSE|NASDAQ)?[:\s]*([A-Z]{1,5})\)',
            text, re.IGNORECASE
        )
        if paren_ticker:
            note.underlying_ticker = paren_ticker.group(1)

    # Check for index underlyings
    if not note.underlying:
        for idx_name in ['S&P 500', 'Russell 2000', 'Dow Jones Industrial Average',
                         'Nasdaq-100', 'EURO STOXX 50']:
            if idx_name.lower() in text_lower:
                note.underlying = idx_name
                break

    # ── CUSIP ──
    cusip_match = re.search(r'CUSIP[:\s/]+([A-Z0-9]{9}|[A-Z0-9]{6,8})', text)
    if cusip_match:
        note.cusip = cusip_match.group(1)

    # ── Dates ──
    # Trade/pricing date
    trade_date = _extract_date(text, [
        r'[Tt]rade\s+[Dd]ate[:\s]*(?:expected\s+to\s+be\s+)?([A-Z][a-z]+ \d{1,2},?\s*\d{4})',
        r'[Pp]ricing\s+[Dd]ate[:\s]*(?:expected\s+to\s+be\s+)?([A-Z][a-z]+ \d{1,2},?\s*\d{4})',
        r'[Tt]rade\s+[Dd]ate[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
    ])
    if trade_date:
        note.issue_date = trade_date

    # Settlement/issue date
    settle_date = _extract_date(text, [
        r'[Oo]riginal\s+[Ii]ssue\s+[Dd]ate[:\s]*(?:expected\s+to\s+be\s+)?([A-Z][a-z]+ \d{1,2},?\s*\d{4})',
        r'[Ss]ettlement\s+[Dd]ate[:\s]*(?:expected\s+to\s+be\s+)?([A-Z][a-z]+ \d{1,2},?\s*\d{4})',
    ])
    if settle_date:
        note.settlement_date = settle_date
    if not note.issue_date and settle_date:
        note.issue_date = settle_date

    # Maturity date
    mat_date = _extract_date(text, [
        r'[Ss]tated\s+[Mm]aturity\s+[Dd]ate[:\s]*(?:expected\s+to\s+be\s+)?([A-Z][a-z]+ \d{1,2},?\s*\d{4})',
        r'[Mm]aturity\s+[Dd]ate[:\s]*(?:expected\s+to\s+be\s+)?([A-Z][a-z]+ \d{1,2},?\s*\d{4})',
        r'due\s+(?:on\s+or\s+about\s+)?([A-Z][a-z]+ \d{1,2},?\s*\d{4})',
    ])
    if mat_date:
        note.maturity_date = mat_date

    # Compute maturity in years from dates
    if note.issue_date and note.maturity_date:
        try:
            from datetime import datetime
            d1 = _parse_date(note.issue_date)
            d2 = _parse_date(note.maturity_date)
            if d1 and d2:
                note.maturity = round((d2 - d1).days / 365.25, 1)
        except:
            pass

    # Fallback: look for "approximately X year"
    if note.maturity == 0:
        yr_match = re.search(r'(?:[Aa]pproximately|[Tt]erm[:\s]+)\s*(\d+(?:\.\d+)?)\s*years?', text)
        if yr_match:
            note.maturity = float(yr_match.group(1))
        else:
            mo_match = re.search(r'(?:[Aa]pproximately|[Tt]erm[:\s]+)\s*(\d+)\s*months?', text)
            if mo_match:
                note.maturity = round(int(mo_match.group(1)) / 12, 2)

    # ── Coupon rate ──
    # Try "at least X% per annum"
    coupon_annual = _extract_pct(text, [
        r'(?:[Cc]ontingent\s+[Cc]oupon\s+[Rr]ate|coupon\s+rate)[:\s]+(?:at\s+least\s+)?(\d+\.?\d*)\s*%\s*per\s+annum',
        r'(?:at\s+least|up\s+to)\s+(\d+\.?\d*)\s*%\s*per\s+annum',
        r'(\d+\.?\d*)\s*%\s*per\s+annum',
    ])

    # Try per-period rate
    coupon_period = _extract_pct(text, [
        r'(\d+\.?\d*)\s*%\s*(?:per\s+)?(?:quarter|quarterly)',
        r'\$(\d+\.?\d*)\s*(?:per\s+)?\$1,?000.*?(?:quarterly|each\s+quarter)',
    ])

    # Try dollar amount per $1000
    coupon_dollar = re.search(
        r'coupon.*?\$(\d+\.?\d*)\s*(?:\(|per\s+\$1,?000)', text, re.IGNORECASE
    )

    if coupon_annual:
        note.coupon_annual_pct = coupon_annual
    if coupon_period:
        note.coupon_rate = coupon_period / 100
        if not coupon_annual:
            note.coupon_annual_pct = coupon_period * 4  # assume quarterly
    elif coupon_dollar:
        dollar_per_period = float(coupon_dollar.group(1))
        note.coupon_rate = dollar_per_period / 1000.0
        if not coupon_annual:
            note.coupon_annual_pct = (dollar_per_period / 1000.0) * 4 * 100
    elif coupon_annual:
        # Guess quarterly
        note.coupon_rate = coupon_annual / 100 / 4

    # ── Observation dates / count ──
    # Count listed dates
    obs_dates_section = re.search(
        r'[Oo]bservation\s+[Dd]ates?.*?(?:quarterly|monthly|semi)',
        text, re.IGNORECASE | re.DOTALL
    )

    if 'quarterly' in text_lower:
        periods_per_year = 4
    elif 'monthly' in text_lower:
        periods_per_year = 12
    elif 'semi-annual' in text_lower or 'semiannual' in text_lower:
        periods_per_year = 2
    else:
        periods_per_year = 4  # default

    if note.maturity > 0:
        note.n_obs = max(1, int(round(note.maturity * periods_per_year)))

    # ── Barriers ──
    # Autocall trigger
    autocall_pct = _extract_pct(text, [
        r'(?:autocall|auto-call|automatic\s*call).*?(\d+\.?\d*)\s*%\s*of\s+(?:the\s+)?(?:initial|starting)',
        r'(?:greater\s+than\s+or\s+equal\s+to)\s+(?:the\s+|its\s+)?initial\s+(?:level|value|price)',
    ])
    if autocall_pct:
        note.autocall_trigger = autocall_pct / 100
    else:
        # Default: if it says "greater than or equal to its initial level" → 100%
        if re.search(r'(?:equal\s+to|at\s+least)\s+(?:the\s+|its\s+)?initial\s+(?:level|value|price)', text, re.IGNORECASE):
            note.autocall_trigger = 1.0

    # Coupon barrier
    coupon_bar = _extract_pct(text, [
        r'[Cc]oupon\s+(?:[Bb]arrier|[Tt]hreshold|[Tt]rigger)\s*(?:[Ll]evel)?[:\s]+(?:\$[\d,.]+[,\s]+which\s+is\s+)?(\d+\.?\d*)\s*%',
        r'[Cc]oupon\s+(?:[Bb]arrier|[Tt]hreshold|[Tt]rigger)[:\s]+(\d+\.?\d*)\s*%',
        r'(\d+\.?\d*)\s*%\s*of\s+(?:the\s+|its\s+)?(?:initial|starting).*?coupon',
    ])
    if coupon_bar:
        note.coupon_barrier = coupon_bar / 100

    # Knock-in / downside barrier
    ki_bar = _extract_pct(text, [
        r'[Dd]ownside\s+[Tt]hreshold[:\s]+(?:\$[\d,.]+[,\s]+which\s+is\s+)?(\d+\.?\d*)\s*%',
        r'[Kk]nock-?\s*[Ii]n\s+(?:[Bb]arrier\s+)?(?:[Ll]evel)?[:\s]+(\d+\.?\d*)\s*%',
        r'[Tt]rigger\s+(?:[Bb]uffer\s+)?[Ll]evel[:\s]+(?:for\s+each\s+underlier,?\s+)?(\d+\.?\d*)\s*%',
        r'[Dd]ownside\s+[Tt]hreshold.*?(\d+\.?\d*)\s*%\s*of\s+(?:the\s+)?[Ii]nitial',
    ])
    if ki_bar:
        note.ki_barrier = ki_bar / 100

    # If coupon barrier not found but KI found, they might be the same
    if note.ki_barrier > 0 and note.coupon_barrier == 0:
        # Check if coupon barrier equals KI
        if 'coupon barrier' not in text_lower and 'coupon threshold' not in text_lower:
            note.coupon_barrier = note.ki_barrier

    # ── Memory coupon ──
    memory_signals = ['memory', 'accumulated coupon', 'cumulative coupon',
                      'catch-up', 'catch up', 'previously unpaid']
    note.memory = any(sig in text_lower for sig in memory_signals)

    # ── First autocall observation ──
    # Look for "will not be subject to automatic call until [date]"
    no_call_until = re.search(
        r'(?:will\s+not\s+be\s+(?:subject\s+to\s+)?(?:automatic\s+call|called))\s+until\s+(?:the\s+)?([A-Z][a-z]+ \d{4})',
        text, re.IGNORECASE
    )
    if no_call_until:
        # Try to figure out which obs number this is
        note.first_autocall_obs = 2  # safe default if there's a delay
    else:
        # Check if call starts from first observation or later
        call_commence = re.search(
            r'[Cc]all\s+[Oo]bservation\s+[Dd]ates?.*?commencing\s+(?:in\s+)?([A-Z][a-z]+ \d{4})',
            text
        )
        coupon_commence = re.search(
            r'[Cc]oupon\s+[Oo]bservation\s+[Dd]ates?.*?commencing\s+(?:in\s+)?([A-Z][a-z]+ \d{4})',
            text
        )
        if call_commence and coupon_commence:
            # If call starts later than coupon, compute the offset
            call_start = call_commence.group(1)
            coupon_start = coupon_commence.group(1)
            if call_start != coupon_start:
                note.first_autocall_obs = 2  # at least a 1-period delay
            else:
                note.first_autocall_obs = 1

    # ── Estimated value ──
    ev_match = re.search(
        r'[Ee]stimated\s+[Vv]alue.*?\$(\d+[\d,.]*)\s*per\s+\$[\d,]+(?:\.\d+)?\s*(?:face|principal|note)',
        text
    )
    if ev_match:
        val = ev_match.group(1).replace(',', '')
        note.issuer_estimated_value = float(val)
        # Normalize to per-$1000 basis
        if note.issuer_estimated_value < 100:
            # It's per $10 (JPMorgan style)
            note.issuer_estimated_value *= 100

    # Also try range: "between $925 and $955"
    if note.issuer_estimated_value == 0:
        ev_range = re.search(
            r'[Ee]stimated\s+[Vv]alue.*?between\s+\$(\d+[\d,.]*)\s+and\s+\$(\d+[\d,.]*)',
            text
        )
        if ev_range:
            low = float(ev_range.group(1).replace(',', ''))
            high = float(ev_range.group(2).replace(',', ''))
            note.issuer_estimated_value = (low + high) / 2

    # Also try "approximately $XXX per $1,000"
    if note.issuer_estimated_value == 0:
        ev_approx = re.search(
            r'[Ee]stimated\s+[Vv]alue.*?approximately\s+\$(\d+[\d,.]*)\s*per\s+\$1,?000',
            text
        )
        if ev_approx:
            note.issuer_estimated_value = float(ev_approx.group(1).replace(',', ''))

    # ── Generate note ID ──
    ticker = note.underlying_ticker or note.underlying[:4].upper().replace(' ', '')
    date_short = note.issue_date.replace('-', '')[:6] if note.issue_date else 'XXXX'
    issuer_short = note.issuer[:2].upper() if note.issuer else 'XX'
    note.note_id = f"{issuer_short}-{ticker}-{date_short}"

    # ── Confidence score ──
    filled = sum([
        bool(note.issuer),
        bool(note.underlying),
        bool(note.issue_date),
        note.maturity > 0,
        note.coupon_rate > 0,
        note.ki_barrier > 0,
        note.issuer_estimated_value > 0,
    ])
    if filled >= 6:
        note.confidence = "high"
    elif filled >= 4:
        note.confidence = "medium"
    else:
        note.confidence = "low"

    return note


# ── Helper functions ──────────────────────────────────────────────

def _extract_pct(text: str, patterns: List[str]) -> Optional[float]:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except (ValueError, IndexError):
                continue
    return None


def _extract_date(text: str, patterns: List[str]) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(1).strip()
            parsed = _parse_date(date_str)
            if parsed:
                return parsed.strftime('%Y-%m-%d')
    return None


def _parse_date(date_str: str):
    from datetime import datetime
    for fmt in ['%B %d, %Y', '%B %d %Y', '%b %d, %Y', '%b %d %Y',
                '%m/%d/%Y', '%Y-%m-%d']:
        try:
            return datetime.strptime(date_str.replace(',', ',').strip(), fmt)
        except ValueError:
            continue
    return None


# ── Output ────────────────────────────────────────────────────────

def print_extracted(note: ExtractedNote):
    """Pretty-print the extracted fields."""
    print(f"\n  {'EXTRACTED TERM SHEET':=^55}")
    print(f"  Note ID:              {note.note_id}")
    print(f"  Issuer:               {note.issuer}")
    print(f"  Underlying:           {note.underlying} ({note.underlying_ticker})")
    print(f"  Trade/Pricing Date:   {note.issue_date}")
    print(f"  Settlement Date:      {note.settlement_date}")
    print(f"  Maturity Date:        {note.maturity_date}")
    print(f"  S0 (Initial Price):   {note.S0}")
    print(f"  Maturity:             {note.maturity} years")
    print(f"  Observations:         {note.n_obs}")
    print(f"  Coupon Rate:          {note.coupon_rate:.4f} per period ({note.coupon_annual_pct:.2f}% p.a.)")
    print(f"  Autocall Trigger:     {note.autocall_trigger:.0%}")
    print(f"  Coupon Barrier:       {note.coupon_barrier:.0%}")
    print(f"  KI Barrier:           {note.ki_barrier:.0%}")
    print(f"  Memory Coupon:        {note.memory}")
    print(f"  First Autocall Obs:   {note.first_autocall_obs}")
    print(f"  Estimated Value:      ${note.issuer_estimated_value:.2f}")
    print(f"  CUSIP:                {note.cusip}")
    print(f"  Confidence:           {note.confidence}")
    if note.notes:
        print(f"  NOTES:                {note.notes}")
    print(f"  {'='*55}")

    margin = (1000 - note.issuer_estimated_value) / 1000 * 100 if note.issuer_estimated_value > 0 else 0
    print(f"\n  Issuer's admitted margin: {margin:.1f}%")


def to_csv_row(note: ExtractedNote) -> dict:
    """Convert to the CSV format expected by the backtest engine."""
    return {
        'note_id': note.note_id,
        'issuer': note.issuer,
        'underlying': note.underlying_ticker or note.underlying,
        'issue_date': note.issue_date,
        'S0': note.S0,
        'maturity': note.maturity,
        'n_obs': note.n_obs,
        'coupon_rate': round(note.coupon_rate, 6),
        'autocall_trigger': note.autocall_trigger,
        'coupon_barrier': note.coupon_barrier,
        'ki_barrier': note.ki_barrier,
        'memory': note.memory,
        'issuer_estimated_value': note.issuer_estimated_value,
        'first_autocall_obs': note.first_autocall_obs,
    }


CSV_COLUMNS = [
    'note_id', 'issuer', 'underlying', 'issue_date', 'S0',
    'maturity', 'n_obs', 'coupon_rate', 'autocall_trigger',
    'coupon_barrier', 'ki_barrier', 'memory', 'issuer_estimated_value',
    'first_autocall_obs',
]


def append_to_csv(note: ExtractedNote, filepath: str):
    """Append extracted note to CSV file."""
    file_exists = os.path.exists(filepath)
    row = to_csv_row(note)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n  Appended to {filepath}")


def process_url(url: str, output_file: str) -> Optional[ExtractedNote]:
    """Full pipeline: fetch → extract → print → save."""
    html = fetch_filing(url)
    if not html:
        return None

    note = extract_term_sheet(html, url)
    if not note:
        return None

    print_extracted(note)

    # Print the CSV row for copy-paste
    row = to_csv_row(note)
    csv_line = ','.join(str(row[c]) for c in CSV_COLUMNS)
    print(f"\n  CSV row (copy-paste ready):")
    print(f"  {csv_line}")

    append_to_csv(note, output_file)
    return note


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDGAR Autocallable Note Extractor")
    parser.add_argument("--url", type=str, help="Single filing URL")
    parser.add_argument("--batch", type=str, help="File with one URL per line")
    parser.add_argument("--output", type=str, default="term_sheets.csv", help="Output CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("EDGAR AUTOCALLABLE NOTE EXTRACTOR")
    print("=" * 60)

    if args.url:
        process_url(args.url, args.output)

    elif args.batch:
        with open(args.batch, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and line.strip().startswith('http')]
        print(f"  Found {len(urls)} URLs in {args.batch}")
        for i, url in enumerate(urls):
            print(f"\n[{i+1}/{len(urls)}]")
            process_url(url, args.output)
            time.sleep(0.2)  # Be nice to SEC servers

    else:
        # Interactive mode
        print("\nPaste EDGAR 424B2 URLs one at a time. Type 'done' to finish.\n")
        count = 0
        while True:
            url = input("URL (or 'done'): ").strip()
            if url.lower() in ('done', 'quit', 'exit', 'q'):
                break
            if not url.startswith('http'):
                print("  Not a valid URL. Try again.")
                continue
            process_url(url, args.output)
            count += 1
            time.sleep(0.2)

        print(f"\n  Extracted {count} notes → {args.output}")
        print(f"  Remember to fill in S0 for each note!")
        print(f"  Then run: python data_pipeline.py --input {args.output}")


if __name__ == "__main__":
    main()
