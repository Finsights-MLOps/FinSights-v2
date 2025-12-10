from edgar import get_filings, set_identity, Company
from edgar.reference.tickers import find_ticker_safe
import pandas as pd
import os
import nltk
from datetime import datetime
import re
import hashlib
from bs4 import BeautifulSoup
from pathlib import Path

# Download sentence tokenizer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# --------------------------
# 1️⃣ Configuration
# --------------------------
set_identity("karthiks233@email.com")

# companies_list = ["AAPL"]

companies_list = ['AAPL', 'MSFT', 'LLY', 'AMZN', 'JNJ', 'COST', 'NFLX', 'GOOGL',
                  'MA', 'TSLA', 'XOM', 'GNW', 'ORCL', 'GOOG', 'V', 'NVDA', 'IEP', 
                  'WMT', 'MBI', 'RDN', 'AGO', 'META']
form_type = "10-K"
start_date_range = "2024-01-01"
end_date_range = "2024-12-31"

start_year = int(start_date_range[:4])
end_year = int(end_date_range[:4])
years = list(range(start_year, end_year + 1))

output_dir = Path(__file__).parent / "sentence_data"
os.makedirs(output_dir, exist_ok=True)
output_file_parquet = os.path.join(output_dir, "10k_sentences_extracted.parquet")
output_file_csv = os.path.join(output_dir, "10k_sentences_extracted.csv")

SECTIONS_TO_EXTRACT = ["Item 1", "Item 1A", "Item 7", "Item 7A"]

# --------------------------------------------------------
# Extract report date from SEC filing header (MOST RELIABLE)
# --------------------------------------------------------
def extract_report_date_from_header(filing):
    """
    Extract CONFORMED PERIOD OF REPORT from the SEC filing header.
    This is the most reliable source for the actual report date.
    """
    try:
        # Get the raw filing text which includes the SGML header
        filing_text = filing.text()
        if not filing_text:
            return None
        
        # Look for CONFORMED PERIOD OF REPORT in the header (first 2000 chars)
        header = filing_text[:2000]
        period_match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", header)
        
        if period_match:
            date_str = period_match.group(1)  # Format: YYYYMMDD
            parsed = datetime.strptime(date_str, "%Y%m%d").date()
            return str(parsed)
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error extracting from header: {e}")
        return None

# --------------------------------------------------------
# Extract "For the fiscal year ended ..." from HTML
# --------------------------------------------------------
def extract_report_date_from_html(html_content):
    """
    Extract the fiscal year end date from the 10-K cover page.
    Tries multiple patterns and locations in the HTML.
    """
    try:
        if not html_content:
            return None
        
        # Try parsing with BeautifulSoup for better text extraction
        soup = BeautifulSoup(html_content[:5000], 'html.parser')
        text = soup.get_text()
        
        # Multiple regex patterns to catch different formats
        patterns = [
            r"For the (?:fiscal )?year ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            r"fiscal year ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            r"year ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            r"FISCAL YEAR ENDED\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            r"YEAR ENDED\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_date = match.group(1).strip()
                try:
                    parsed = datetime.strptime(raw_date, "%B %d, %Y").date()
                    return str(parsed)
                except ValueError:
                    continue
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error extracting report date from HTML: {e}")
        return None

# --------------------------------------------------------
# Alternative: Extract from filing metadata directly
# --------------------------------------------------------
def extract_report_date_from_filing(filing):
    """
    Extract report date from the filing object's metadata.
    IMPORTANT: Does NOT fallback to filing_date as they can be different years.
    """
    try:
        # Try to get the period of report from filing metadata
        if hasattr(filing, 'report_date') and filing.report_date:
            return str(filing.report_date)
        
        # Check for period_of_report attribute
        if hasattr(filing, 'period_of_report') and filing.period_of_report:
            return str(filing.period_of_report)
        
        return None
    except Exception as e:
        print(f"⚠️ Error extracting from filing metadata: {e}")
        return None

# --------------------------------------------------------
# Basic NLP tags
# --------------------------------------------------------
def likely_kpi(sentence):
    KPI_KEYWORDS = ["growth", "margin", "revenue", "cost", "profit", "guidance",
                    "expenses", "cash flow", "operating", "capital", "EBITDA"]
    return any(k.lower() in sentence.lower() for k in KPI_KEYWORDS)

def contains_number(sentence):
    return bool(re.search(r"\d", sentence))

def contains_comparison(sentence):
    return any(word in sentence.lower() for word in ["than", "vs", "compared", "greater", "less"])

def normalize_section_id(section_name: str) -> int:
    """
    Converts 'Item 1', 'Item 1A', 'Item 7', 'Item 7A' → 1, 1, 7, 7
    """
    num = "".join(ch for ch in section_name if ch.isdigit())
    return int(num) if num else 0

# --------------------------------------------------------
# Hash row
# --------------------------------------------------------
def generate_row_hash(row):
    concat = (
        str(row["cik"])
        + str(row["docID"])
        + str(row["section_ID"])
        + row["sentence"]
    )
    return hashlib.md5(concat.encode()).hexdigest()

# --------------------------------------------------------
# Metadata columns
# --------------------------------------------------------
def add_metadata(df, source_identifier="SEC/EDGAR"):
    now = datetime.now()
    df["sample_created_at"] = now
    df["last_modified_date"] = now
    df["sample_version"] = "1.0.0"
    df["load_method"] = "manual_python_script"
    df["source_file_path"] = df.apply(
        lambda r: f"{source_identifier}/{r['cik']}/{r['docID']}", axis=1
    )
    df["temporal_bin"] = df["filingDate"].astype(str)
    df["row_hash"] = df.apply(generate_row_hash, axis=1)
    return df

# --------------------------------------------------------
# MAIN EXTRACTION LOOP
# --------------------------------------------------------
all_rows = []
count = 0

for ticker in companies_list:
    print(f"\n🔍 Fetching filings for {ticker}")
    
    filings = get_filings(year=years, form=form_type, amendments=False)
    ticker_filings = filings.filter(
        ticker=[ticker],
        filing_date=f"{start_date_range}:{end_date_range}"
    )
    
    if len(ticker_filings) == 0:
        print(f"❌ No filings found for {ticker}")
        continue
    
    for filing in ticker_filings:
        company_name = filing.company
        filing_date = filing.filing_date
        accession = filing.accession_no
        cik = filing.cik
        cik_int = int(cik)
        tickers = ticker
        report_year = filing_date.year if filing_date else ""
        docID = f"{cik}_{form_type}_{report_year}"
        
        try:
            company_obj = Company(ticker)
            sic = company_obj.sic or ""
        except:
            sic = ""
        
        # Get the 10-K document object
        doc = filing.obj()
        
        # Get raw HTML for report date extraction (if needed)
        try:
            raw_html = filing.html()
        except:
            raw_html = None
        
        # Try multiple methods to get report date
        report_date = extract_report_date_from_header(filing)
        if not report_date:
            report_date = extract_report_date_from_filing(filing)
        if not report_date and raw_html:
            report_date = extract_report_date_from_html(raw_html)
        
        if report_date:
            print(f"📅 {ticker} - Filing: {filing_date}, Report Date: {report_date}")
        else:
            print(f"⚠️ {ticker} - Filing: {filing_date}, Report Date: NOT FOUND")
        
        for section in SECTIONS_TO_EXTRACT:
            try:
                # Access section directly using bracket notation
                text = doc[section]
                
                if text is None:
                    print(f"⚠️ {ticker} - Section '{section}' returned None")
                    continue
                    
                if not text.strip():
                    print(f"⚠️ {ticker} - Section '{section}' is empty")
                    continue
                    
                print(f"✅ {ticker} - Extracted '{section}': {len(text)} characters")
                
            except KeyError:
                print(f"⚠️ {ticker} - Section '{section}' not found")
                continue
            except Exception as e:
                print(f"⚠️ {ticker} - Section '{section}' error: {type(e).__name__}: {e}")
                continue
            
            sentences = sent_tokenize(text.strip())
            section_id = normalize_section_id(section)
            
            print(f"  📝 Tokenized into {len(sentences)} sentences")
            
            for idx, sentence in enumerate(sentences):
                row = {
                    "cik": str(cik).zfill(10),
                    "cik_int": cik_int,
                    "name": company_name,
                    "tickers": ticker,
                    "docID": docID,
                    "sentenceID": f"{docID}_section_{section_id}_{idx}",
                    "section_ID": section_id,
                    "section_name": section,
                    "form": form_type,
                    "sic": sic,
                    "sentence": sentence,
                    "filingDate": filing_date,
                    "report_year": report_year,
                    "reportDate": report_date,
                    "temporal_bin": f'bin_{start_year}_{end_year}',
                    "likely_kpi": likely_kpi(sentence),
                    "has_numbers": contains_number(sentence),
                    "has_comparison": contains_comparison(sentence),
                }
                
                all_rows.append(row)
                count += 1

print(f"\n📊 Total sentences extracted: {count}")

# --------------------------------------------------------
# Save final combined dataset
# --------------------------------------------------------
if all_rows:
    df = pd.DataFrame(all_rows)
    df = add_metadata(df)
    df.to_parquet(output_file_parquet, index=False)
    df.to_csv(output_file_csv, index=False)
    print(f"\n✅ Extracted {len(df)} sentences")
    print(f"💾 Saved parquet → {output_file_parquet}")
    print(f"\n📊 Report Date Coverage:")
    print(df['reportDate'].value_counts())
    print(f"\n📊 Section Distribution:")
    print(df['section_name'].value_counts())
else:
    print("\n❌ No rows extracted")