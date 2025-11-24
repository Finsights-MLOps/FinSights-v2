"""
FinRAG-Insights: metric + section + risk keyword configuration

Grounded on:
- analysis_keywords_kpi_by_label.json
- analysis_keywords_kpi_tokens.json
- analysis_keywords_risk_by_topic.json
- analysis_keywords_risk_tokens.json
- analysis_keywords_section_by_section.json
- analysis_keywords_section_global.json
"""

# ---------------------------------------------------------------------
# 1) Natural language -> canonical metric name
# ---------------------------------------------------------------------

"""
tightly focused on semantic aliases - 
    all the ways humans talk about revenue, net income, operating income, COGS, tax, cash flow, EPS etc 
    -  normalized into canonical metric IDs like income_stmt_Revenue, cash_flow_Operating Cash Flow, 
    Return on Assets (ROA) %, EPS, etc.
"""


# ---------------------------------------------------------------------
# Natural language -> metric_label (97 standardized labels)
# ---------------------------------------------------------------------

"""
Maps natural language queries to exact metric_label values in KPI_FACT_DATA_EDGAR.parquet
Covers all 97 available metrics including derived ratios and GAAP line items
"""

METRIC_MAPPINGS = {
    # ===================================================================
    # CORE INCOME STATEMENT METRICS
    # ===================================================================
    
    # Revenue (primary metric)
    'revenue': 'Revenue',
    'revenues': 'Revenue',
    'total revenue': 'Revenue',
    'total revenues': 'Revenue',
    'sales': 'Revenue',
    'total sales': 'Revenue',
    'top line': 'Revenue',
    'sales revenue': 'Revenue',
    
    # Net Revenue variant
    'net revenue': 'Net Revenue',
    'net revenues': 'Net Revenue',
    'net sales': 'Net Revenue',
    
    # Gross Revenue variant
    'gross revenue': 'Gross Revenue',
    'gross revenues': 'Gross Revenue',
    'gross sales': 'Gross Revenue',
    
    # Net Income (primary profit metric)
    'net income': 'Net Income',
    'net earnings': 'Net Income',
    'earnings': 'Net Income',
    'profit': 'Net Income',
    'net profit': 'Net Income',
    'bottom line': 'Net Income',
    'income': 'Net Income',
    'net loss': 'Net Income',  # Same metric, negative value
    
    # Net Income variants (specialized)
    'net income common basic': 'Net Income Available to Common - Basic',
    'net income common': 'Net Income Available to Common - Basic',
    'net income basic': 'Net Income Available to Common - Basic',
    'earnings per common basic': 'Net Income Available to Common - Basic',
    
    'net income common diluted': 'Net Income Available to Common - Diluted',
    'net income diluted': 'Net Income Available to Common - Diluted',
    'diluted net income': 'Net Income Available to Common - Diluted',
    
    'net income parent diluted': 'Net Income Attributable to Parent - Diluted',
    'parent net income': 'Net Income Attributable to Parent - Diluted',
    
    'net income nci': 'Net Income - Noncontrolling Interest',
    'noncontrolling interest income': 'Net Income - Noncontrolling Interest',
    
    'net income nonredeemable nci': 'Net Income - Nonredeemable NCI',
    'net income redeemable nci': 'Net Income - Redeemable NCI',
    'net income including nci': 'Net Income - Including Nonredeemable NCI',
    
    # LP Unit metrics
    'net income lp': 'Net Income per LP Unit (Basic)',
    'income per lp unit': 'Net Income per LP Unit (Basic)',
    'net income lp basic diluted': 'Net Income per LP Unit (Basic & Diluted)',
    
    # Gross Profit
    'gross profit': 'Gross Profit',
    'gross income': 'Gross Profit',
    'gross margin dollars': 'Gross Profit',
    'gross margins': 'Gross Profit',  
    'gross profit margin': 'Gross Profit',
    'gross profit margins': 'Gross Profit',  
    'gp': 'Gross Profit',
    
    'gross profit disposal': 'Gross Profit (Disposal Group)',
    
    # ===================================================================
    # CASH FLOW METRICS
    # ===================================================================
    
    # Operating Cash Flow (primary)
    'operating cash flow': 'Operating Cash Flow',
    'operating cash flows': 'Operating Cash Flow',  
    'cash flows': 'Operating Cash Flow',            
    'cash flow from operations': 'Operating Cash Flow',
    'cash flows from operations': 'Operating Cash Flow',  
    'cash flow operations': 'Operating Cash Flow',
    'ocf': 'Operating Cash Flow',
    'cfo': 'Operating Cash Flow',
    'cash from operations': 'Operating Cash Flow',
    'operating activities': 'Operating Cash Flow',
    'cash provided by operating activities': 'Operating Cash Flow',
    'net cash operating': 'Operating Cash Flow',
    'cashflow': 'Operating Cash Flow',
    'cash flow': 'Operating Cash Flow',
    
    # Continuing Ops variant
    'operating cash flow continuing': 'Operating Cash Flow - Continuing Ops',
    'ocf continuing operations': 'Operating Cash Flow - Continuing Ops',
    
    # ===================================================================
    # BALANCE SHEET - EQUITY
    # ===================================================================
    
    # Stockholders Equity (primary)
    'stockholders equity': 'Stockholders\' Equity',
    'shareholders equity': 'Stockholders\' Equity',
    'stockholder equity': 'Stockholders\' Equity',
    'shareholder equity': 'Stockholders\' Equity',
    'total equity': 'Stockholders\' Equity',
    'equity': 'Stockholders\' Equity',
    'book value': 'Stockholders\' Equity',
    'net worth': 'Stockholders\' Equity',
    
    # Equity variants
    'equity including nci': 'Equity Including NCI',
    'equity with nci': 'Equity Including NCI',
    'equity noncontrolling': 'Equity Including NCI',
    
    'other equity': 'Other Stockholders\' Equity',
    'other stockholders equity': 'Other Stockholders\' Equity',
    
    'equity shares': 'Other Equity Shares',
    'other equity shares': 'Other Equity Shares',
    
    'stock split': 'Equity - Stock Split Conversion',
    'stock split conversion': 'Equity - Stock Split Conversion',
    
    # ===================================================================
    # BALANCE SHEET - LIABILITIES
    # ===================================================================
    
    # Generic liabilities
    'liabilities': 'Other Liabilities',
    'total liabilities': 'Liabilities & Stockholders\' Equity',
    'liabilities equity': 'Liabilities & Stockholders\' Equity',
    'liabilities and equity': 'Liabilities & Stockholders\' Equity',
    
    'other liabilities': 'Other Liabilities',
    'other liabilities noncurrent': 'Other Liabilities (Noncurrent)',
    'other liabilities current': 'Other Liabilities Current',
    'other liabilities fair value': 'Other Liabilities - Fair Value',
    'other liabilities disposal': 'Other Liabilities - Disposal Group',
    
    'sundry liabilities': 'Other Sundry Liabilities',
    'sundry liabilities noncurrent': 'Other Sundry Liabilities (Noncurrent)',
    
    'accrued liabilities': 'Other Accrued Liabilities (Current)',
    'accrued liabilities current': 'Other Accrued Liabilities (Current)',
    
    # Derivative liabilities
    'derivative liabilities': 'Derivative Liabilities (Current)',
    'derivative liabilities current': 'Derivative Liabilities (Current)',
    
    # Employee liabilities
    'employee liabilities': 'Employee-Related Liabilities (Total)',
    'employee liabilities total': 'Employee-Related Liabilities (Total)',
    'employee liabilities current': 'Employee-Related Liabilities (Current)',
    
    # Pension liabilities
    'pension liabilities': 'Pension & Benefit Liabilities (Total)',
    'pension benefits': 'Pension & Benefit Liabilities (Total)',
    'pension liabilities noncurrent': 'Pension & Postretirement Liabilities - Noncurrent',
    'postretirement liabilities': 'Pension & Postretirement Liabilities - Noncurrent',
    
    # Business combination
    'liabilities assumed': 'Business Combination - Liabilities Assumed',
    'acquisition liabilities': 'Business Combination - Liabilities Assumed',
    'noncash liabilities assumed': 'Non-cash Liabilities Assumed',
    
    # Disposal group liabilities
    'disposal liabilities current': 'Current Liabilities - Disposal Group',
    
    # ===================================================================
    # BALANCE SHEET - ASSETS
    # ===================================================================
    
    # Other assets
    'other assets noncurrent': 'Other Assets (Noncurrent)',
    'other noncurrent assets': 'Other Assets (Noncurrent)',
    'other assets current': 'Other Current Assets',
    'other current assets': 'Other Current Assets',
    'other assets fair value': 'Other Assets - Fair Value',
    'other assets disposal': 'Other Assets - Disposal Group',
    
    'miscellaneous assets': 'Miscellaneous Assets (Noncurrent)',
    'misc assets noncurrent': 'Miscellaneous Assets (Noncurrent)',
    
    # Prepaid expenses
    'prepaid expenses': 'Prepaid Expenses (Current)',
    'prepaid': 'Prepaid Expenses (Current)',
    
    # Intangible assets
    'intangible assets gross': 'Intangible Assets (Gross, Excluding Goodwill)',
    'intangibles gross': 'Intangible Assets (Gross, Excluding Goodwill)',
    'intangible assets': 'Intangible Assets (Gross, Excluding Goodwill)',
    
    # Impairments
    'impairment intangibles': 'Impairment - Indefinite-Lived Intangibles',
    'intangible impairment': 'Impairment - Indefinite-Lived Intangibles',
    'impairment long lived': 'Impairment - Long-Lived Assets',
    'asset impairment': 'Impairment - Long-Lived Assets',
    
    # Amortization
    'amortization year 5': 'Amortization Expense - Year 5',
    
    # Change in assets
    'change other assets': 'Change in Other Assets',
    
    # Disposal group assets
    'disposal assets current': 'Current Assets - Disposal Group',
    
    # Securities
    'securities loaned': 'Securities Loaned',
    'loaned securities': 'Securities Loaned',
    
    # Separate accounts
    'separate account assets': 'Separate Account Assets',
    
    # ===================================================================
    # DERIVED RATIOS & MARGINS
    # ===================================================================
    
    # Return ratios
    'return on assets': 'ROA % (Avg Assets)',
    'roa': 'ROA % (Avg Assets)',
    'roa percent': 'ROA % (Avg Assets)',
    'return on average assets': 'ROA % (Avg Assets)',
    
    'return on equity': 'ROE % (Avg Equity)',
    'roe': 'ROE % (Avg Equity)',
    'roe percent': 'ROE % (Avg Equity)',
    'return on average equity': 'ROE % (Avg Equity)',
    
    # Margin ratios
    'operating margin': 'Operating Margin %',
    'operating margin percent': 'Operating Margin %',
    'ebit margin': 'Operating Margin %',
    'op margin': 'Operating Margin %',
    
    'net profit margin': 'Net Profit Margin %',
    'net margin': 'Net Profit Margin %',
    'profit margin': 'Net Profit Margin %',
    'npm': 'Net Profit Margin %',
    
    'gross profit margin': 'Gross Profit',  # Note: Use dollars, not % in data
    'gross margin': 'Gross Profit',
    
    # Debt ratios
    'debt to assets': 'Debt-to-Assets',
    'debt to asset ratio': 'Debt-to-Assets',
    'debt assets': 'Debt-to-Assets',
    'total debt levels': 'Debt-to-Assets',        
    'debt levels': 'Debt-to-Assets',               
    'total debt': 'Debt-to-Assets',                
    'debt': 'Debt-to-Assets',                      

    'debt to equity': 'Debt-to-Equity',
    'debt to equity ratio': 'Debt-to-Equity',
    'debt equity': 'Debt-to-Equity',
    'leverage ratio': 'Debt-to-Equity',
    'leverage': 'Debt-to-Equity',
    
    # Cost of Goods Sold - NOT AVAILABLE, map to closest proxy
    'cost of goods sold': 'Gross Profit',         
    'cogs': 'Gross Profit',                        
    'cost of sales': 'Gross Profit',               
    'cost of revenue': 'Gross Profit',

    # ===================================================================
    # TAX-RELATED METRICS (Deferred Tax Assets/Liabilities)
    # ===================================================================
    
    # DTA - Net
    'deferred tax assets net': 'Deferred Income Tax Assets - Net',
    'dta net': 'Deferred Income Tax Assets - Net',
    'net deferred tax assets': 'Deferred Income Tax Assets - Net',
    
    # DTA - NOL (Net Operating Loss)
    'deferred tax nol': 'Deferred Tax Assets - NOL',
    'dta nol': 'Deferred Tax Assets - NOL',
    'nol deferred tax': 'Deferred Tax Assets - NOL',
    
    'dta nol domestic': 'Deferred Tax Assets - NOL Domestic',
    'deferred tax nol domestic': 'Deferred Tax Assets - NOL Domestic',
    
    'dta nol foreign': 'Deferred Tax Assets - NOL Foreign',
    'deferred tax nol foreign': 'Deferred Tax Assets - NOL Foreign',
    
    'dta nol state local': 'Deferred Tax Assets - NOL State/Local',
    'deferred tax nol state': 'Deferred Tax Assets - NOL State/Local',
    
    # DTA - Other categories
    'dta derivatives': 'Deferred Tax Assets - Derivatives',
    'deferred tax assets derivatives': 'Deferred Tax Assets - Derivatives',
    
    'dta ppe': 'Deferred Tax Assets - PPE',
    'deferred tax assets ppe': 'Deferred Tax Assets - PPE',
    
    'dta deferred income': 'Deferred Tax Assets - Deferred Income',
    
    'dta iprd': 'Deferred Tax Assets - IPR&D',
    'dta research development': 'Deferred Tax Assets - IPR&D',
    
    'dta oci loss': 'Deferred Tax Assets - OCI Loss',
    'deferred tax oci': 'Deferred Tax Assets - OCI Loss',
    
    'dta unrealized losses': 'Deferred Tax Assets - Unrealized Losses (AFS)',
    
    'dta capital loss': 'Deferred Tax Assets - Capital Loss CF',
    
    'dta other': 'Deferred Tax Assets - Other',
    
    # DTL - Deferred Tax Liabilities
    'dtl derivatives': 'Deferred Tax Liabilities - Derivatives',
    'deferred tax liabilities derivatives': 'Deferred Tax Liabilities - Derivatives',
    
    'dtl ppe': 'Deferred Tax Liabilities - PPE',
    'deferred tax liabilities ppe': 'Deferred Tax Liabilities - PPE',
    
    'dtl goodwill intangibles': 'Deferred Tax Liabilities - Goodwill & Intangibles',
    'deferred tax goodwill': 'Deferred Tax Liabilities - Goodwill & Intangibles',
    
    'dtl noncontrolled affiliates': 'Deferred Tax Liabilities - Noncontrolled Affiliates',
    
    'dtl undistributed foreign': 'Deferred Tax Liabilities - Undistributed Foreign Earnings',
    'deferred tax foreign earnings': 'Deferred Tax Liabilities - Undistributed Foreign Earnings',
    
    'dtl unrealized gains': 'Deferred Tax Liabilities - Unrealized Gains (Trading)',
    
    # Combined deferred tax line items
    'deferred taxes other assets current': 'Deferred Taxes & Other Assets (Current)',
    'deferred taxes other liabilities': 'Deferred Tax And Other Liabilities (Noncurrent)',
    'deferred income taxes liabilities': 'Deferred Income Taxes And Other Liabilities (Noncurrent)',
    
    # Tax valuation allowance
    'tax valuation allowance': 'Deferred Tax Asset Valuation Allowance Change',
    'valuation allowance change': 'Deferred Tax Asset Valuation Allowance Change',
    'tax rate valuation allowance': 'Tax Rate - Change in Valuation Allowance',
    
    # ===================================================================
    # CAPITAL EXPENDITURES & ASSET TRANSACTIONS
    # ===================================================================
    
    # Payments for assets
    'capex productive': 'Payments to Acquire Productive Assets',
    'capital expenditures': 'Payments to Acquire Productive Assets',
    'capex': 'Payments to Acquire Productive Assets',
    
    'capex intangibles': 'Payments to Acquire Intangible Assets',
    'intangibles': 'Payments to Acquire Intangible Assets',
    'capex': 'Payments to Acquire Intangible Assets',
    'intangible capex': 'Payments to Acquire Intangible Assets',
    
    'net payments productive': 'Net Payments/Proceeds - Productive Assets',
    'net capex': 'Net Payments/Proceeds - Productive Assets',
    
    # Proceeds from sales
    'proceeds productive': 'Proceeds from Productive Asset Sales',
    'asset sale proceeds': 'Proceeds from Productive Asset Sales',
    
    'proceeds other assets': 'Proceeds from Other Asset Sales (Investing)',
    
    'proceeds businesses': 'Proceeds from Sale of Businesses / Assets',
    'business sale proceeds': 'Proceeds from Sale of Businesses / Assets',
    
    # Segment capex
    'segment capex': 'Segment Additions to Long-Lived Assets',
    'segment additions': 'Segment Additions to Long-Lived Assets',
    
    # ===================================================================
    # VARIABLE INTEREST ENTITIES (VIE)
    # ===================================================================
    
    'vie assets': 'VIE Consolidated Assets',
    'vie consolidated assets': 'VIE Consolidated Assets',
    
    'vie liabilities': 'VIE Consolidated Liabilities',
    'vie consolidated liabilities': 'VIE Consolidated Liabilities',
    
    'vie liabilities no recourse': 'VIE Liabilities (No Recourse)',
    
    'vie assets pledged': 'VIE Assets Pledged',
    'vie pledged assets': 'VIE Assets Pledged',
    
    'vie nonconsolidated assets': 'VIE Non-consolidated Assets',
    
    # ===================================================================
    # OTHER SPECIALIZED METRICS
    # ===================================================================
    
    # Financial transfers
    'financial asset transfers': 'Financial Asset Transfers Derecognized',
    'asset transfers derecognized': 'Financial Asset Transfers Derecognized',
    
    # Share-based payments
    'share based liabilities paid': 'Share-based Payment Liabilities Paid',
    'stock compensation paid': 'Share-based Payment Liabilities Paid',
}


# ---------------------------------------------------------------------
# 2) Metric keywords for quick detection (no mapping needed)
# ---------------------------------------------------------------------

METRIC_KEYWORDS = list(METRIC_MAPPINGS.keys())

# ---------------------------------------------------------------------
# 3) Quantitative indicators (triggers metric layer)
# ---------------------------------------------------------------------

QUANTITATIVE_INDICATORS = [
    'how much', 'what was', 'what were', 'show me', 'compare',
    'trend', 'growth', 'increase', 'decrease', 'change',
    'performance', 'financial', 'report', 'results'
]



# ---------------------------------------------------------------------
# 4) Section keywords: NL phrases -> sec_item_canonical (ITEM_x / ITEM_xA)
#    This is designed to drive S3 Vectors filters on `sec_item_canonical`
# ---------------------------------------------------------------------

SECTION_KEYWORDS = {
    # Core business (Item 1)
    "business description": "ITEM_1",
    "business overview": "ITEM_1",
    "company overview": "ITEM_1",
    "products and services": "ITEM_1",
    "market segments": "ITEM_1",
    "business strategy": "ITEM_1",

    # Risk factors (Item 1A)
    "risk factors": "ITEM_1A",
    "key risks": "ITEM_1A",
    "risk factor section": "ITEM_1A",
    "forward looking statements": "ITEM_1A",
    "forward-looking statements": "ITEM_1A",
    "uncertainties": "ITEM_1A",

    # Properties (Item 2)
    "properties": "ITEM_2",
    "facilities": "ITEM_2",
    "real estate": "ITEM_2",

    # Legal proceedings (Item 3)
    "legal proceedings": "ITEM_3",
    "litigation": "ITEM_3",
    "legal matters": "ITEM_3",

    # Mine safety (Item 4)
    "mine safety": "ITEM_4",
    "mine safety disclosures": "ITEM_4",

    # Market / trading info (Item 5)
    "market for registrant common equity": "ITEM_5",
    "stock market data": "ITEM_5",
    "share repurchases": "ITEM_5",
    "dividends": "ITEM_5",
    "stock performance": "ITEM_5",

    # Selected financial data (Item 6)
    "selected financial data": "ITEM_6",
    "multi-year financial summary": "ITEM_6",

    # MD&A core (Item 7)
    "md&a": "ITEM_7",
    "mda": "ITEM_7",
    "mdna": "ITEM_7",
    "management discussion and analysis": "ITEM_7",
    "management's discussion and analysis": "ITEM_7",
    "results of operations": "ITEM_7",
    "operating results": "ITEM_7",
    "liquidity and capital resources": "ITEM_7",
    "capital resources": "ITEM_7",
    "liquidity": "ITEM_7",
    "outlook": "ITEM_7",

    # Market risk (Item 7A)
    "market risk": "ITEM_7A",
    "quantitative and qualitative disclosures about market risk": "ITEM_7A",
    "interest rate risk": "ITEM_7A",
    "foreign currency risk": "ITEM_7A",
    "fx risk": "ITEM_7A",
    "commodity price risk": "ITEM_7A",

    # Financial statements (Item 8)
    "financial statements": "ITEM_8",
    "audited financial statements": "ITEM_8",
    "balance sheet": "ITEM_8",
    "income statement": "ITEM_8",
    "statement of operations": "ITEM_8",
    "statement of cash flows": "ITEM_8",
    "cash flow statement": "ITEM_8",
    "supplementary data": "ITEM_8",

    # Changes in accountants (Item 9)
    "changes in and disagreements with accountants": "ITEM_9",
    "changes in accountants": "ITEM_9",

    # Controls & procedures (Item 9A)
    "controls and procedures": "ITEM_9A",
    "internal controls": "ITEM_9A",
    "icfr": "ITEM_9A",
    "sox 404": "ITEM_9A",
    "internal control over financial reporting": "ITEM_9A",

    # Other information (Item 9B)
    "other information": "ITEM_9B",

    # Governance & ownership (Part III)
    "directors and executive officers": "ITEM_10",
    "corporate governance": "ITEM_10",
    "board of directors": "ITEM_10",

    "executive compensation": "ITEM_11",
    "compensation discussion and analysis": "ITEM_11",

    "security ownership": "ITEM_12",
    "beneficial ownership": "ITEM_12",

    "related party transactions": "ITEM_13",
    "related transactions": "ITEM_13",

    "auditor fees": "ITEM_14",
    "principal accountant fees": "ITEM_14",

    # Exhibits / summary (Part IV)
    "exhibits and financial statement schedules": "ITEM_15",
    "exhibit index": "ITEM_15",

    "10-k summary": "ITEM_16",
    "form 10-k summary": "ITEM_16",
}

# ---------------------------------------------------------------------
# 4b) Explicit "Item X" patterns -> sec_item_canonical
#
# This is *in addition* to SECTION_KEYWORDS and will be used by the
# SectionExtractor as regexes on the normalized query string.
# ---------------------------------------------------------------------

SECTION_ITEM_PATTERNS = {
    # Generic "item N" / "itemN" / "item-N" etc.
    r"\bitem\s*[-_ ]?\s*1\b": "ITEM_1",
    r"\bitem\s*[-_ ]?\s*1a\b": "ITEM_1A",
    r"\bitem\s*[-_ ]?\s*2\b": "ITEM_2",
    r"\bitem\s*[-_ ]?\s*3\b": "ITEM_3",
    r"\bitem\s*[-_ ]?\s*4\b": "ITEM_4",
    r"\bitem\s*[-_ ]?\s*5\b": "ITEM_5",
    r"\bitem\s*[-_ ]?\s*6\b": "ITEM_6",
    r"\bitem\s*[-_ ]?\s*7\b": "ITEM_7",
    r"\bitem\s*[-_ ]?\s*7a\b": "ITEM_7A",
    r"\bitem\s*[-_ ]?\s*8\b": "ITEM_8",
    r"\bitem\s*[-_ ]?\s*9\b": "ITEM_9",
    r"\bitem\s*[-_ ]?\s*9a\b": "ITEM_9A",
    r"\bitem\s*[-_ ]?\s*9b\b": "ITEM_9B",
    r"\bitem\s*[-_ ]?\s*10\b": "ITEM_10",
    r"\bitem\s*[-_ ]?\s*11\b": "ITEM_11",
    r"\bitem\s*[-_ ]?\s*12\b": "ITEM_12",
    r"\bitem\s*[-_ ]?\s*13\b": "ITEM_13",
    r"\bitem\s*[-_ ]?\s*14\b": "ITEM_14",
    r"\bitem\s*[-_ ]?\s*15\b": "ITEM_15",
    r"\bitem\s*[-_ ]?\s*16\b": "ITEM_16",

    # Bare numeric references that often appear as "see Item 7A above"
    r"\b7a\b": "ITEM_7A",
    r"\b1a\b": "ITEM_1A",
}




# ---------------------------------------------------------------------
# 5) Risk-topic keywords (natural phrases -> risk_topic buckets)
#    Keys are your topic labels from view2_risk_atlas
# ---------------------------------------------------------------------

RISK_TOPIC_KEYWORDS = {
    'liquidity_credit': [
        'liquidity', 'cash flow', 'cash flows', 'cash position', 'capital resources', 'capital structure', 'refinancing',
        'refinance', 'credit facility', 'credit facilities', 'revolving credit', 'covenant', 'covenants', 'default',
          'debt maturity', 'going concern', 'solvency', ],
    'regulatory': [
        'regulatory', 'regulation', 'regulations', 'regulatory approval', 'regulatory changes', 'laws and regulations',
          'legal and regulatory', 'compliance', 'government investigations', 'regulatory investigations', 'fines',
            'penalties', 'sanctions', 'enforcement actions', ],
    'market_competitive': [
        'competition', 'competitive', 'competitive pressures', 'market share', 'pricing pressure', 'pricing pressures',
          'demand', 'customer demand', 'macroeconomic', 'economic conditions', 'recession', 'downturn',
            'market volatility', 'volatility', ],
    'operational_supply_chain': [
        'operations', 'operational', 'operating disruptions', 'supply chain', 'supply-chain', 'supply disruptions',
          'logistics', 'manufacturing', 'production facilities', 'plant closures', 'outages', 'business interruption',
            'natural disaster', 'catastrophic events', ],
    'cybersecurity_tech': [
        'cybersecurity', 'cyber security', 'cyber', 'information security', 'data security', 'data breach',
          'breach of data', 'security incident', 'security incidents', 'ransomware', 'malware', 'hacking', 'unauthorized access', 'privacy', 'personal data', 'confidential information', ],
    'legal_ip_litigation': [
        'litigation', 'lawsuits', 'class action', 'legal proceedings', 'legal claims', 'claims and proceedings',
          'disputes', 'arbitration', 'settlement', 'judgment', 'patent', 'patent infringement', 'intellectual property',
            'ip rights', ],
    'general_risk': [
        'risk factors', 'risks and uncertainties', 'material adverse effect', 'adverse effect', 'adverse impacts',
          'economic conditions', 'market conditions', 'pandemic', 'covid-19', 'downturn', 'volatility', ],
}
