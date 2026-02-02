#!/usr/bin/env python3
"""
Multilingual invoice normalization.

Architecture:
- Business logic = language-agnostic (this file)
- Business terms = per-language dictionaries (dictionaries/)
- Extensible and startup-ready
"""

import re
from typing import Optional, Dict


def normalize_text(text: str, dictionary: Optional[Dict] = None) -> str:
    """
    Clean OCR text in a universal way.

    Transformations:
    - Non-breaking spaces → normal spaces
    - Multiple spaces → single space
    - Optional: replace business terms using a dictionary

    Args:
        text: raw OCR text
        dictionary: mapping of business terms to standardized fields
                    (e.g., {"qty": "quantity"})

    Returns:
        cleaned text
    """
    if not text:
        return ""

    # 1. Clean special spaces (universal, all languages)
    text = text.replace('\xa0', ' ').replace('\u202f', ' ')
    text = re.sub(r'\s+', ' ', text)

    # 2. Optional: normalize business terms
    if dictionary:
        text_lower = text.lower()
        for term, standardized in dictionary.items():
            # Replace term by standardized field
            text_lower = re.sub(
                rf'\b{re.escape(term)}\b',
                standardized,
                text_lower
            )
        return text_lower.strip()

    return text.strip()


def normalize_amount(value: str) -> Optional[float]:
    """
    Convert a monetary amount to float (multilingual).

    Supports:
    - FR/DE/IT/ES: '1 829,17 €' → 1829.17
    - EN/US: '1,829.17 $' → 1829.17
    - Universal: '1829.17 EUR' → 1829.17

    Smart detection:
    - Comma as decimal → FR/DE style (1.200,50)
    - Dot as decimal → EN/US style (1,200.50)

    Args:
        value: amount as string

    Returns:
        float or None if conversion fails
    """
    if not value or not isinstance(value, str):
        return None

    # Remove currency symbols and spaces
    v = re.sub(r'[€$£¥\sA-Z]', '', value.strip())
    if not v:
        return None

    has_comma = ',' in v
    has_dot = '.' in v

    if has_comma and has_dot:
        # Both present → last one is decimal
        last_comma = v.rfind(',')
        last_dot = v.rfind('.')
        if last_comma > last_dot:
            # FR style: "1.200,50"
            v = v.replace('.', '').replace(',', '.')
        else:
            # EN style: "1,200.50"
            v = v.replace(',', '')
    elif has_comma and not has_dot:
        parts = v.split(',')
        if len(parts[-1]) <= 2:
            # Decimal comma: "100,50"
            v = v.replace(',', '.')
        else:
            # Thousands separator: "1,200"
            v = v.replace(',', '')

    # Dot alone → already correct
    try:
        return float(v)
    except ValueError:
        return None


def normalize_percentage(value: str) -> Optional[float]:
    """
    Convert a percentage to float (universal).

    Examples:
        '20,00%' → 20.0
        '5.5%' → 5.5
        '19 %' → 19.0
    """
    if not value:
        return None

    v = re.sub(r'[%\s]', '', value)
    v = v.replace(',', '.')

    try:
        return float(v)
    except ValueError:
        return None


def normalize_date(value: str) -> Optional[str]:
    """
    Normalize dates to ISO format (YYYY-MM-DD).

    Supports:
    - EU (FR/DE/IT/ES): '15/01/2024' → '2024-01-15'
    - UK: '15-01-2024' → '2024-01-15'
    - US: '01/15/2024' → '2024-01-15' (TODO)

    Note: US format ambiguous, needs language context.
    """
    if not value:
        return None

    # EU format: DD/MM/YYYY or DD-MM-YYYY
    match = re.match(r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})', value)
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = '20' + year
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    return value


def detect_currency(value: str) -> str:
    """
    Detect the currency of an amount.

    Returns:
        'EUR', 'USD', 'GBP', 'CHF', or 'UNKNOWN'

    Useful for:
    - Multi-currency validation
    - Currency-specific reporting
    - Detecting invoice errors (FR invoice in $)
    """
    if not value:
        return 'UNKNOWN'

    v = value.upper()
    if '€' in value or 'EUR' in v:
        return 'EUR'
    if '$' in value or 'USD' in v:
        return 'USD'
    if '£' in value or 'GBP' in v:
        return 'GBP'
    if 'CHF' in v:
        return 'CHF'

    return 'UNKNOWN'


