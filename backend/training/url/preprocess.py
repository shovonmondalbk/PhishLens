"""
PhishLens - URL Preprocessor
Prepares URLs for DeBERTa v3 training.

Key Features:
- Extracts URL components (protocol, subdomain, domain, tld, path, params)
- Creates structured representation for transformer learning
- Handles edge cases (IP addresses, unicode, punycode)
"""

import re
import os
import json
import pandas as pd
from urllib.parse import urlparse, unquote
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class URLPreprocessor:
    """
    Preprocesses URLs for phishing detection model training.
    
    The preprocessing creates a structured text representation that helps
    DeBERTa understand URL patterns without relying on simple heuristics.
    """
    
    # Common legitimate TLDs (not exhaustive, just for feature extraction)
    COMMON_TLDS = {
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
        'co', 'io', 'ai', 'app', 'dev', 'me', 'info', 'biz',
        'uk', 'us', 'ca', 'au', 'de', 'fr', 'jp', 'cn', 'in', 'br',
        'ac', 'bd'  # Added for Bangladesh domains like du.ac.bd
    }
    
    # Known URL shorteners
    SHORTENERS = {
        'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly',
        'is.gd', 'buff.ly', 'short.link', 'tiny.cc', 'bc.vc',
        'j.mp', 'rb.gy', 'cutt.ly', 's.id'
    }
    
    # Suspicious keywords often found in phishing URLs
    SUSPICIOUS_KEYWORDS = {
        'login', 'signin', 'sign-in', 'verify', 'verification',
        'secure', 'security', 'account', 'update', 'confirm',
        'password', 'credential', 'authenticate', 'banking',
        'paypal', 'amazon', 'microsoft', 'apple', 'google',
        'facebook', 'instagram', 'netflix', 'walmart', 'ebay'
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config_path: Path to config.json (optional)
        """
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from config.json"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def extract_url_components(self, url: str) -> Dict[str, str]:
        """
        Extract all components from a URL.
        
        Args:
            url: Raw URL string
            
        Returns:
            Dictionary with URL components
        """
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        try:
            # Decode URL-encoded characters
            url = unquote(url)
            parsed = urlparse(url)
            
            # Extract components
            protocol = parsed.scheme or 'http'
            full_host = parsed.netloc.lower()
            path = parsed.path or '/'
            query = parsed.query or ''
            fragment = parsed.fragment or ''
            
            # Remove port if present
            if ':' in full_host:
                full_host = full_host.split(':')[0]
            
            # Check if IP address
            is_ip = self._is_ip_address(full_host)
            
            if is_ip:
                domain = full_host
                subdomain = ''
                tld = ''
            else:
                # Split host into parts
                subdomain, domain, tld = self._split_host(full_host)
            
            return {
                'protocol': protocol,
                'full_host': full_host,
                'subdomain': subdomain,
                'domain': domain,
                'tld': tld,
                'path': path,
                'query': query,
                'fragment': fragment,
                'is_ip': is_ip,
                'original_url': url
            }
            
        except Exception as e:
            logger.warning(f"Error parsing URL '{url}': {e}")
            return {
                'protocol': 'http',
                'full_host': url,
                'subdomain': '',
                'domain': url,
                'tld': '',
                'path': '/',
                'query': '',
                'fragment': '',
                'is_ip': False,
                'original_url': url
            }
    
    def _is_ip_address(self, host: str) -> bool:
        """Check if host is an IP address."""
        # IPv4 pattern
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        # IPv6 pattern (simplified)
        ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$'
        
        return bool(re.match(ipv4_pattern, host) or re.match(ipv6_pattern, host))
    
    def _split_host(self, host: str) -> Tuple[str, str, str]:
        """
        Split hostname into subdomain, domain, and TLD.
        Handles complex TLDs like .co.uk, .ac.bd, .edu.au
        
        Args:
            host: Full hostname (e.g., 'student.cse.du.ac.bd')
            
        Returns:
            Tuple of (subdomain, domain, tld)
        """
        parts = host.split('.')
        
        if len(parts) < 2:
            return '', host, ''
        
        # Handle multi-part TLDs (e.g., .co.uk, .ac.bd, .com.au)
        # Check last two parts first
        if len(parts) >= 2:
            potential_tld = '.'.join(parts[-2:])
            # Common multi-part TLDs
            multi_tlds = {
                'co.uk', 'co.jp', 'co.in', 'co.nz', 'co.za',
                'com.au', 'com.br', 'com.cn', 'com.mx', 'com.sg',
                'ac.bd', 'ac.uk', 'ac.jp', 'ac.in', 'ac.nz',
                'edu.au', 'edu.cn', 'edu.sg',
                'gov.uk', 'gov.au', 'gov.in',
                'org.uk', 'org.au', 'net.au'
            }
            
            if potential_tld in multi_tlds:
                if len(parts) == 2:
                    return '', parts[0], potential_tld
                elif len(parts) == 3:
                    return '', parts[0], potential_tld
                else:
                    return '.'.join(parts[:-3]), parts[-3], potential_tld
        
        # Single TLD case
        tld = parts[-1]
        domain = parts[-2] if len(parts) >= 2 else ''
        subdomain = '.'.join(parts[:-2]) if len(parts) > 2 else ''
        
        return subdomain, domain, tld
    
    def extract_features(self, url: str) -> Dict:
        """
        Extract numerical and categorical features from URL.
        These are auxiliary features that can help the model.
        
        Args:
            url: Raw URL string
            
        Returns:
            Dictionary of extracted features
        """
        components = self.extract_url_components(url)
        
        # Length features
        url_length = len(url)
        host_length = len(components['full_host'])
        path_length = len(components['path'])
        
        # Count features
        num_dots = url.count('.')
        num_hyphens = url.count('-')
        num_underscores = url.count('_')
        num_slashes = url.count('/')
        num_digits = sum(c.isdigit() for c in url)
        num_params = components['query'].count('&') + (1 if components['query'] else 0)
        
        # Subdomain analysis
        subdomain = components['subdomain']
        num_subdomains = len(subdomain.split('.')) if subdomain else 0
        
        # Special character ratios
        special_chars = sum(not c.isalnum() and c not in './-_' for c in url)
        
        # Suspicious patterns
        has_ip = components['is_ip']
        has_at_symbol = '@' in url
        has_double_slash_redirect = '//' in url[8:]  # After protocol
        is_https = components['protocol'] == 'https'
        
        # Shortener check
        is_shortener = components['full_host'] in self.SHORTENERS
        
        # Suspicious keyword check
        url_lower = url.lower()
        suspicious_keyword_count = sum(
            1 for kw in self.SUSPICIOUS_KEYWORDS if kw in url_lower
        )
        
        # Brand in subdomain/path but not in domain (potential impersonation)
        brands = {'paypal', 'amazon', 'microsoft', 'apple', 'google', 
                  'facebook', 'netflix', 'instagram', 'twitter', 'linkedin'}
        brand_mismatch = any(
            brand in subdomain.lower() or brand in components['path'].lower()
            for brand in brands
            if brand not in components['domain'].lower()
        )
        
        return {
            'url_length': url_length,
            'host_length': host_length,
            'path_length': path_length,
            'num_dots': num_dots,
            'num_hyphens': num_hyphens,
            'num_underscores': num_underscores,
            'num_slashes': num_slashes,
            'num_digits': num_digits,
            'num_params': num_params,
            'num_subdomains': num_subdomains,
            'special_chars': special_chars,
            'has_ip': int(has_ip),
            'has_at_symbol': int(has_at_symbol),
            'has_double_slash': int(has_double_slash_redirect),
            'is_https': int(is_https),
            'is_shortener': int(is_shortener),
            'suspicious_keyword_count': suspicious_keyword_count,
            'brand_mismatch': int(brand_mismatch)
        }
    
    def create_model_input(self, url: str) -> str:
        """
        Create a structured text representation of the URL for DeBERTa.
        
        This method creates a human-readable representation that helps
        the transformer model understand URL structure and patterns.
        
        Args:
            url: Raw URL string
            
        Returns:
            Structured text representation for model input
        """
        components = self.extract_url_components(url)
        features = self.extract_features(url)
        
        # Build structured representation
        parts = []
        
        # Protocol
        parts.append(f"[PROTOCOL] {components['protocol']}")
        
        # Host structure
        if components['is_ip']:
            parts.append(f"[IP] {components['full_host']}")
        else:
            if components['subdomain']:
                parts.append(f"[SUBDOMAIN] {components['subdomain']}")
            parts.append(f"[DOMAIN] {components['domain']}")
            if components['tld']:
                parts.append(f"[TLD] {components['tld']}")
        
        # Path (truncate if too long)
        path = components['path']
        if len(path) > 100:
            path = path[:100] + '...'
        if path and path != '/':
            parts.append(f"[PATH] {path}")
        
        # Query parameters (just indicate presence and count)
        if components['query']:
            parts.append(f"[PARAMS] {features['num_params']} parameters")
        
        # Security indicators
        if features['is_https']:
            parts.append("[SECURE]")
        else:
            parts.append("[INSECURE]")
        
        # Warning flags
        warnings = []
        if features['has_ip']:
            warnings.append("IP-based")
        if features['has_at_symbol']:
            warnings.append("contains-@")
        if features['is_shortener']:
            warnings.append("shortener")
        if features['brand_mismatch']:
            warnings.append("brand-mismatch")
        if features['suspicious_keyword_count'] > 0:
            warnings.append(f"suspicious-keywords:{features['suspicious_keyword_count']}")
        
        if warnings:
            parts.append(f"[FLAGS] {', '.join(warnings)}")
        
        # Original URL (for reference patterns)
        parts.append(f"[URL] {url}")
        
        return ' '.join(parts)
    
    def preprocess_dataset(
        self,
        input_path: str,
        output_path: str,
        url_column: str = 'url',
        label_column: str = 'label'
    ) -> pd.DataFrame:
        """
        Preprocess entire dataset.
        
        Args:
            input_path: Path to input CSV
            output_path: Path to save processed CSV
            url_column: Name of URL column
            label_column: Name of label column
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        
        logger.info(f"Processing {len(df)} URLs...")
        
        # Create model inputs
        processed_texts = []
        all_features = []
        
        for url in tqdm(df[url_column], desc="Processing URLs"):
            # Create structured text for model
            model_input = self.create_model_input(url)
            processed_texts.append(model_input)
            
            # Extract features
            features = self.extract_features(url)
            all_features.append(features)
        
        # Add to dataframe
        df['processed_text'] = processed_texts
        
        # Add feature columns
        features_df = pd.DataFrame(all_features)
        df = pd.concat([df, features_df], axis=1)
        
        # Save processed dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved processed dataset to {output_path}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df


def main():
    """Main function to preprocess the URL dataset."""
    
    # Paths
    base_dir = r"C:\PhishLens"
    input_path = os.path.join(base_dir, "data", "raw", "url", "phishlens_url_dataset.csv")
    output_path = os.path.join(base_dir, "data", "processed", "url", "processed_urls.csv")
    config_path = os.path.join(base_dir, "backend", "inference", "config.json")
    
    # Initialize preprocessor
    preprocessor = URLPreprocessor(config_path)
    
    # Process dataset
    df = preprocessor.preprocess_dataset(
        input_path=input_path,
        output_path=output_path,
        url_column='url',
        label_column='label'
    )
    
    # Print sample
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nTotal URLs processed: {len(df)}")
    print(f"\nSample processed text:")
    print("-" * 60)
    print(df['processed_text'].iloc[0])
    print("-" * 60)
    print(df['processed_text'].iloc[1])
    print("-" * 60)
    
    # Show label distribution
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Show feature statistics
    print(f"\nFeature statistics:")
    feature_cols = ['url_length', 'num_subdomains', 'suspicious_keyword_count', 'is_https']
    print(df[feature_cols].describe())


if __name__ == "__main__":
    main()