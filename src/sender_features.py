# sender_features.py (example code)
import dns.resolver

def check_spf(domain):
    try:
        result = dns.resolver.resolve(domain, 'TXT')
        spf_record = [str(record) for record in result if 'v=spf1' in str(record).lower()]
        return 1 if spf_record else 0
    except Exception:
        return 0

def check_dkim(domain):
    try:
        result = dns.resolver.resolve(f'_domainkey.{domain}', 'TXT')
        dkim_record = [str(record) for record in result]
        return 1 if dkim_record else 0
    except Exception:
        return 0

def check_dmarc(domain):
    try:
        result = dns.resolver.resolve(f'_dmarc.{domain}', 'TXT')
        dmarc_record = [str(record) for record in result]
        return 1 if dmarc_record else 0
    except Exception:
        return 0

def check_sender(domain: str) -> dict:
    """Run all sender-authentication checks for a domain in one call."""
    return {
        "spf": check_spf(domain),
        "dkim": check_dkim(domain),
        "dmarc": check_dmarc(domain),
    }


# IMPORTANT: don't run network calls on import - only when executed directly.
# Previously this ran a live DNS lookup against 'example.com' every time the
# module was imported (e.g. every app startup), which slowed things down
# and could raise on machines without network access.
if __name__ == "__main__":
    sender_domain = "example.com"
    result = check_sender(sender_domain)
    print(f"SPF: {result['spf']}, DKIM: {result['dkim']}, DMARC: {result['dmarc']}")
