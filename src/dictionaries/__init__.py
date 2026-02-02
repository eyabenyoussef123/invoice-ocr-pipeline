import operator

def detect_language(text: str) -> str:
    import unicodedata

    # Normalize text (lowercase + remove accents)
    normalized_text = ''.join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    )

    scores: dict[str, int] = {'fr': 0, 'en': 0}

    french_keywords = ['facture', 'designation', 'montant', 'ttc', 'tva', 'qte']
    english_keywords = ['invoice', 'description', 'amount', 'total', 'vat', 'qty']

    scores['fr'] = sum(1 for word in french_keywords if word in normalized_text)
    scores['en'] = sum(1 for word in english_keywords if word in normalized_text)

    # Pylance-friendly max
    language, max_score = max(scores.items(), key=operator.itemgetter(1))

    # fallback to generic if score = 0
    return language if max_score > 0 else 'generic'
