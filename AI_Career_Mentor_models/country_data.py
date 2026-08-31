"""
country_data.py
----------------
Real-world country economic data used to enrich the AI Career Mentor
salary-prediction dataset, and to convert predicted USD salaries into
each country's local currency.

Sources (fetched live, Aug 2026):
  - Cost of Living Index & Local Purchasing Power Index: Numbeo
    (https://www.numbeo.com/cost-of-living/rankings_by_country.jsp)
  - USD exchange rates: x-rates.com / OFX (interbank, Aug 2026 snapshot)

NOTE: FX rates move daily. For a production app, refresh `EXCHANGE_RATE_TO_USD`
from a live FX API (e.g. exchangerate.host, Xe, Wise) instead of hardcoding.
"""

# Cost of Living Index (100 = NYC baseline) and Local Purchasing Power Index
# (higher = salary goes further locally), per Numbeo, 2026 mid-year data.
COUNTRY_ECONOMIC_DATA = {
    "Australia":      {"cost_of_living_index": 71.4, "purchasing_power_index": 134.8, "currency_code": "AUD"},
    "Brazil":         {"cost_of_living_index": 33.1, "purchasing_power_index": 44.3,  "currency_code": "BRL"},
    "Canada":         {"cost_of_living_index": 61.3, "purchasing_power_index": 114.8, "currency_code": "CAD"},
    "Germany":        {"cost_of_living_index": 68.0, "purchasing_power_index": 130.0, "currency_code": "EUR"},
    "India":          {"cost_of_living_index": 18.1, "purchasing_power_index": 69.6,  "currency_code": "INR"},
    "Japan":          {"cost_of_living_index": 47.6, "purchasing_power_index": 107.3, "currency_code": "JPY"},
    "Nigeria":        {"cost_of_living_index": 20.2, "purchasing_power_index": 8.8,   "currency_code": "NGN"},
    "Pakistan":       {"cost_of_living_index": 20.3, "purchasing_power_index": 27.9,  "currency_code": "PKR"},
    "Philippines":    {"cost_of_living_index": 29.1, "purchasing_power_index": 32.1,  "currency_code": "PHP"},
    "Poland":         {"cost_of_living_index": 46.2, "purchasing_power_index": 94.7,  "currency_code": "PLN"},
    "Singapore":      {"cost_of_living_index": 90.8, "purchasing_power_index": 91.3,  "currency_code": "SGD"},
    "South Africa":   {"cost_of_living_index": 38.9, "purchasing_power_index": 105.5, "currency_code": "ZAR"},
    "UAE":            {"cost_of_living_index": 55.6, "purchasing_power_index": 113.4, "currency_code": "AED"},
    "United Kingdom": {"cost_of_living_index": 68.2, "purchasing_power_index": 118.2, "currency_code": "GBP"},
    "United States":  {"cost_of_living_index": 69.7, "purchasing_power_index": 144.5, "currency_code": "USD"},
}

# 1 USD = X local currency (interbank/market rate snapshot, Aug 2026)
EXCHANGE_RATE_TO_USD = {
    "AUD": 1.3964,
    "BRL": 5.1906,
    "CAD": 1.3898,
    "EUR": 0.8630,
    "INR": 95.40,
    "JPY": 159.79,
    "NGN": 1400.0,
    "PKR": 278.05,
    "PHP": 62.46,
    "PLN": 3.7450,
    "SGD": 1.2730,
    "ZAR": 16.15,
    "AED": 3.6725,
    "GBP": 0.7386,
    "USD": 1.0,
}

CURRENCY_SYMBOL = {
    "AUD": "A$", "BRL": "R$", "CAD": "C$", "EUR": "€", "INR": "₹",
    "JPY": "¥", "NGN": "₦", "PKR": "Rs", "PHP": "₱", "PLN": "zł",
    "SGD": "S$", "ZAR": "R", "AED": "AED", "GBP": "£", "USD": "$",
}


def get_country_info(country: str) -> dict:
    """Return {cost_of_living_index, purchasing_power_index, currency_code} for a country."""
    if country not in COUNTRY_ECONOMIC_DATA:
        raise KeyError(f"No economic data for '{country}'. Known countries: {list(COUNTRY_ECONOMIC_DATA)}")
    return COUNTRY_ECONOMIC_DATA[country]


def to_local_currency(usd_amount: float, country: str) -> float:
    """Convert a USD salary figure into the given country's local currency."""
    info = get_country_info(country)
    rate = EXCHANGE_RATE_TO_USD[info["currency_code"]]
    return round(usd_amount * rate, 2)


def format_local_currency(usd_amount: float, country: str) -> str:
    info = get_country_info(country)
    code = info["currency_code"]
    symbol = CURRENCY_SYMBOL.get(code, code)
    local_amount = to_local_currency(usd_amount, country)
    return f"{symbol}{local_amount:,.0f} {code}"


# Backwards-compatible aliases used elsewhere in the notebook
CURRENCY_CODE = {c: v["currency_code"] for c, v in COUNTRY_ECONOMIC_DATA.items()}

if __name__ == "__main__":
    for c in COUNTRY_ECONOMIC_DATA:
        print(c, "->", format_local_currency(60000, c))
