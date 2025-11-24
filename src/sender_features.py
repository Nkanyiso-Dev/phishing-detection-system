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

# Example usage:
sender_domain = 'example.com'
spf = check_spf(sender_domain)
dkim = check_dkim(sender_domain)
dmarc = check_dmarc(sender_domain)

print(f"SPF: {spf}, DKIM: {dkim}, DMARC: {dmarc}")
