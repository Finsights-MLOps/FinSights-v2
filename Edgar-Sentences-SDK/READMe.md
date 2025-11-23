## Steps to run and modify sentence-level SDK ingestion

# 1. Sentence Ingestion

    Here, we define the tickers of the companies we want to obtain the 10K report of, form type, and we also control the date ranges here

    companies_list = ['AAPL', 'MSFT', 'LLY', 'AMZN', 'JNJ', 'COST', 'NFLX', 'GOOGL',
                  'MA', 'TSLA', 'XOM', 'GNW', 'ORCL', 'GOOG', 'V', 'NVDA', 'IEP', 
                  'WMT', 'MBI', 'RDN', 'AGO', 'META']
    form_type = "10-K"
    start_date_range = "2024-01-01"
    end_date_range = "2024-12-31"

# 2. Sentence Cleaning

    Here, we clean the extracted sentences, and then we do some feature modifications.

# 3. Sentence Validation


    Here, we check for dataframe sanity by running duplicate checks and null checks.

# 4. Data Loading

    Here, we upload the file to the incremental S3 bucket folder.
