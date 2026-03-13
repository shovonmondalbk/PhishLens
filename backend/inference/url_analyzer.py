"""
PhishLens - URL Analyzer Inference Module
Loads trained DeBERTa model and performs phishing URL detection.

Usage:
    from backend.inference.url_analyzer import URLAnalyzer
    
    analyzer = URLAnalyzer()
    result = analyzer.analyze("https://paypa1-secure.com/login")
    print(result)
"""

import os
import sys
import json
import torch
import requests
import urllib3
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote
import re
import logging

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Disable SSL warnings for redirect following
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class URLPreprocessor:
    """
    Preprocesses URLs for the DeBERTa model.
    Must match the preprocessing used during training.
    """
    
    COMMON_TLDS = {
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
        'co', 'io', 'ai', 'app', 'dev', 'me', 'info', 'biz',
        'uk', 'us', 'ca', 'au', 'de', 'fr', 'jp', 'cn', 'in', 'br',
        'ac', 'bd'
    }
    
    SHORTENERS = {
        'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly',
        'is.gd', 'buff.ly', 'short.link', 'tiny.cc', 'bc.vc',
        'j.mp', 'rb.gy', 'cutt.ly', 's.id', 'shorturl.at',
        'shorter.me', 'surl.li', 'rebrand.ly', 'bl.ink', 'short.io',
        'yourls.org', 'v.gd', 'clck.ru', 'qr.ae', 'trib.al',
        'ht.ly', 'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr'
    }
    
    SUSPICIOUS_KEYWORDS = {
        'login', 'signin', 'sign-in', 'verify', 'verification',
        'secure', 'security', 'account', 'update', 'confirm',
        'password', 'credential', 'authenticate', 'banking',
        'paypal', 'amazon', 'microsoft', 'apple', 'google',
        'facebook', 'instagram', 'netflix', 'walmart', 'ebay'
    }
    
    def __init__(self):
        pass
    
    def _is_ip_address(self, host: str) -> bool:
        """Check if host is an IP address."""
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$'
        return bool(re.match(ipv4_pattern, host) or re.match(ipv6_pattern, host))
    
    def _split_host(self, host: str) -> Tuple[str, str, str]:
        """Split hostname into subdomain, domain, and TLD."""
        parts = host.split('.')
        
        if len(parts) < 2:
            return '', host, ''
        
        # Handle multi-part TLDs
        if len(parts) >= 2:
            potential_tld = '.'.join(parts[-2:])
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
        
        tld = parts[-1]
        domain = parts[-2] if len(parts) >= 2 else ''
        subdomain = '.'.join(parts[:-2]) if len(parts) > 2 else ''
        
        return subdomain, domain, tld
    
    def extract_components(self, url: str) -> Dict[str, str]:
        """Extract all components from a URL."""
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        try:
            url = unquote(url)
            parsed = urlparse(url)
            
            protocol = parsed.scheme or 'http'
            full_host = parsed.netloc.lower()
            path = parsed.path or '/'
            query = parsed.query or ''
            fragment = parsed.fragment or ''
            
            if ':' in full_host:
                full_host = full_host.split(':')[0]
            
            is_ip = self._is_ip_address(full_host)
            
            if is_ip:
                domain = full_host
                subdomain = ''
                tld = ''
            else:
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
    
    def extract_features(self, url: str) -> Dict:
        """Extract numerical features from URL."""
        components = self.extract_components(url)
        
        url_length = len(url)
        host_length = len(components['full_host'])
        path_length = len(components['path'])
        
        num_dots = url.count('.')
        num_hyphens = url.count('-')
        num_underscores = url.count('_')
        num_slashes = url.count('/')
        num_digits = sum(c.isdigit() for c in url)
        num_params = components['query'].count('&') + (1 if components['query'] else 0)
        
        subdomain = components['subdomain']
        num_subdomains = len(subdomain.split('.')) if subdomain else 0
        
        special_chars = sum(not c.isalnum() and c not in './-_' for c in url)
        
        has_ip = components['is_ip']
        has_at_symbol = '@' in url
        has_double_slash_redirect = '//' in url[8:]
        is_https = components['protocol'] == 'https'
        is_shortener = components['full_host'] in self.SHORTENERS
        
        url_lower = url.lower()
        suspicious_keyword_count = sum(
            1 for kw in self.SUSPICIOUS_KEYWORDS if kw in url_lower
        )
        
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
        """Create structured text representation for the model."""
        components = self.extract_components(url)
        features = self.extract_features(url)
        
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
        
        # Path
        path = components['path']
        if len(path) > 100:
            path = path[:100] + '...'
        if path and path != '/':
            parts.append(f"[PATH] {path}")
        
        # Query parameters
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
        
        # Original URL
        parts.append(f"[URL] {url}")
        
        return ' '.join(parts)


class URLAnalyzer:
    """
    URL Analyzer for phishing detection using trained DeBERTa model.
    
    Features:
    - Load trained model
    - Preprocess URLs
    - Predict phishing probability
    - Handle URL redirects (for ALL URLs, not just shorteners)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the URL Analyzer.
        
        Args:
            model_path: Path to the trained model directory
            config_path: Path to config.json
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        # Default paths
        base_dir = r"C:\PhishLens"
        self.model_path = model_path or os.path.join(base_dir, "backend", "models", "url", "final_model")
        self.config_path = config_path or os.path.join(base_dir, "backend", "inference", "config.json")
        
        # Load config
        self.config = self._load_config()
        
        # Set device (force CPU for now due to RTX 5070 compatibility)
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize preprocessor
        self.preprocessor = URLPreprocessor()
        
        # Load model
        self._load_model()
        
        # Redirect settings
        self.follow_redirects = self.config.get("redirect", {}).get("follow_redirects", True)
        self.max_redirects = self.config.get("redirect", {}).get("max_redirects", 5)
        self.ignore_www_redirect = self.config.get("redirect", {}).get("ignore_www_redirect", True)
        self.shorteners = set(self.config.get("redirect", {}).get("known_shorteners", []))
        # Add shorteners from preprocessor
        self.shorteners.update(self.preprocessor.SHORTENERS)
        
    def _load_config(self) -> dict:
        """Load configuration from config.json"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        logger.info("Model loaded successfully!")
    
    def resolve_redirects(self, url: str) -> Tuple[str, List[str]]:
        """
        Follow URL redirects and return final URL.
        Works with ALL URLs, not just known shorteners.
        
        Args:
            url: Original URL
            
        Returns:
            Tuple of (final_url, redirect_chain)
        """
        if not self.follow_redirects:
            return url, []
        
        redirect_chain = []
        current_url = url
        
        # Ensure URL has scheme
        if not current_url.startswith(('http://', 'https://')):
            current_url = 'http://' + current_url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        try:
            for i in range(self.max_redirects):
                try:
                    # Try HEAD request first (faster)
                    response = requests.head(
                        current_url,
                        allow_redirects=False,
                        timeout=10,
                        headers=headers,
                        verify=True
                    )
                except requests.exceptions.SSLError:
                    # Try without SSL verification for problematic sites
                    response = requests.head(
                        current_url,
                        allow_redirects=False,
                        timeout=10,
                        headers=headers,
                        verify=False
                    )
                except requests.exceptions.RequestException:
                    # If HEAD fails, try GET request
                    try:
                        response = requests.get(
                            current_url,
                            allow_redirects=False,
                            timeout=10,
                            headers=headers,
                            verify=False,
                            stream=True  # Don't download body
                        )
                        response.close()
                    except:
                        break
                
                if response.status_code in (301, 302, 303, 307, 308):
                    next_url = response.headers.get('Location', '')
                    
                    if not next_url:
                        break
                    
                    # Handle relative URLs
                    if next_url.startswith('/'):
                        parsed = urlparse(current_url)
                        next_url = f"{parsed.scheme}://{parsed.netloc}{next_url}"
                    elif not next_url.startswith(('http://', 'https://')):
                        # Handle protocol-relative URLs
                        if next_url.startswith('//'):
                            parsed = urlparse(current_url)
                            next_url = f"{parsed.scheme}:{next_url}"
                        else:
                            parsed = urlparse(current_url)
                            next_url = f"{parsed.scheme}://{parsed.netloc}/{next_url}"
                    
                    # Check if it's just a www redirect (ignore these)
                    if self.ignore_www_redirect:
                        current_parsed = urlparse(current_url)
                        next_parsed = urlparse(next_url)
                        
                        current_host = current_parsed.netloc.lower().replace('www.', '')
                        next_host = next_parsed.netloc.lower().replace('www.', '')
                        
                        if current_host == next_host and current_parsed.path == next_parsed.path:
                            current_url = next_url
                            continue
                    
                    redirect_chain.append(next_url)
                    current_url = next_url
                else:
                    # No more redirects
                    break
                    
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout following redirects for {url}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error following redirects for {url}")
        except requests.RequestException as e:
            logger.warning(f"Error following redirects for {url}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error following redirects for {url}: {e}")
        
        return current_url, redirect_chain
    
    def is_shortener(self, url: str) -> bool:
        """Check if URL is from a known URL shortener."""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            parsed = urlparse(url)
            host = parsed.netloc.lower().replace('www.', '')
            return host in self.shorteners
        except:
            return False
    
    def analyze(
        self,
        url: str,
        follow_redirects: bool = True,
        return_details: bool = True
    ) -> Dict:
        """
        Analyze a URL for phishing.
        
        Args:
            url: URL to analyze
            follow_redirects: Whether to follow redirects
            return_details: Whether to return detailed analysis
            
        Returns:
            Dictionary with analysis results
        """
        original_url = url
        
        result = {
            "url": original_url,
            "final_url": url,
            "is_phishing": False,
            "confidence": 0.0,
            "label": "legitimate",
            "risk_score": 0.0,
            "redirect_chain": [],
            "is_shortener": False,
            "details": {}
        }
        
        # Check if it's a shortener
        result["is_shortener"] = self.is_shortener(url)
        
        # Check for redirects - Always try to resolve if enabled
        final_url = url
        redirect_chain = []
        
        if follow_redirects and self.follow_redirects:
            logger.info(f"Resolving redirects for: {url}")
            final_url, redirect_chain = self.resolve_redirects(url)
            result["redirect_chain"] = redirect_chain
            
            if redirect_chain:
                logger.info(f"Resolved: {url} -> {final_url}")
                logger.info(f"Redirect chain: {redirect_chain}")
        
        result["final_url"] = final_url
        
        # Preprocess the FINAL URL (after redirects)
        model_input = self.preprocessor.create_model_input(final_url)
        
        # Tokenize
        encoding = self.tokenizer(
            model_input,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Get prediction
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            phishing_prob = probabilities[0][1].item()
        
        # Determine result
        is_phishing = predicted_class == 1
        
        result["is_phishing"] = is_phishing
        result["confidence"] = round(confidence * 100, 2)
        result["label"] = "phishing" if is_phishing else "legitimate"
        result["risk_score"] = round(phishing_prob * 100, 2)
        
        # Add details if requested
        if return_details:
            components = self.preprocessor.extract_components(final_url)
            features = self.preprocessor.extract_features(final_url)
            
            result["details"] = {
                "analyzed_url": final_url,
                "original_url": original_url,
                "components": components,
                "features": features,
                "model_input": model_input,
                "probabilities": {
                    "legitimate": round(probabilities[0][0].item() * 100, 2),
                    "phishing": round(probabilities[0][1].item() * 100, 2)
                }
            }
        
        return result
    
    def analyze_batch(self, urls: List[str], follow_redirects: bool = True) -> List[Dict]:
        """
        Analyze multiple URLs.
        
        Args:
            urls: List of URLs to analyze
            follow_redirects: Whether to follow redirects
            
        Returns:
            List of analysis results
        """
        results = []
        for url in urls:
            result = self.analyze(url, follow_redirects=follow_redirects, return_details=False)
            results.append(result)
        return results


def main():
    """Test the URL analyzer."""
    print("=" * 60)
    print("PhishLens URL Analyzer - Test")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = URLAnalyzer()
    
    # Test URLs
    test_urls = [
        # Legitimate URLs
        "https://www.google.com",
        "https://www.paypal.com/signin",
        "https://student.cse.du.ac.bd",
        "https://github.com/microsoft/vscode",
        
        # Phishing URLs (examples)
        "http://paypa1-secure.com/login",
        "http://amaz0n-verify.com/account",
        "http://192.168.1.1/paypal/login.php",
        "http://faceb00k-login.xyz/auth",
    ]
    
    print("\nAnalyzing URLs...\n")
    print("-" * 60)
    
    for url in test_urls:
        result = analyzer.analyze(url, follow_redirects=True)
        
        status = "🔴 PHISHING" if result["is_phishing"] else "🟢 SAFE"
        
        print(f"URL: {url}")
        if result["final_url"] != url:
            print(f"Final URL: {result['final_url']}")
        print(f"Status: {status}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Risk Score: {result['risk_score']}%")
        print("-" * 60)
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()
