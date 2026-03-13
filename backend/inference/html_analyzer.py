"""
PhishLens - HTML Analyzer Inference Module
Loads trained ELECTRA model and performs phishing HTML detection.

Usage:
    from backend.inference.html_analyzer import HTMLAnalyzer
    
    analyzer = HTMLAnalyzer()
    
    # Analyze HTML content directly
    result = analyzer.analyze_html(html_content)
    
    # Analyze by fetching URL
    result = analyzer.analyze_url("https://example.com")
    
    print(result)
"""

import os
import sys
import re
import json
import torch
import requests
import urllib3
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup, Comment
import logging

from transformers import ElectraForSequenceClassification, ElectraTokenizer

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root
sys.path.insert(0, r"C:\PhishLens")


class HTMLPreprocessor:
    """
    Advanced HTML Preprocessor for ELECTRA model inference.
    Extracts comprehensive features from HTML content.
    """
    
    # ==================== KEYWORD DICTIONARIES ====================
    
    URGENCY_WORDS = {
        'urgent', 'immediately', 'instant', 'now', 'today', 'quick',
        'hurry', 'fast', 'limited time', 'expire', 'expiring', 'expires',
        'deadline', 'within 24 hours', 'within 48 hours', 'act now',
        'action required', 'immediate action', 'time sensitive',
        'don\'t delay', 'last chance', 'final notice', 'final warning',
        'respond immediately', 'asap', 'right away', 'promptly'
    }
    
    THREAT_WORDS = {
        'suspend', 'suspended', 'terminate', 'terminated', 'locked',
        'lock', 'disabled', 'disable', 'unauthorized', 'unusual activity',
        'suspicious activity', 'suspicious login', 'breach', 'breached',
        'compromised', 'hacked', 'illegal', 'violation', 'restricted',
        'restriction', 'blocked', 'permanently', 'deleted', 'removal',
        'fraud', 'fraudulent', 'security alert', 'security warning',
        'account at risk', 'identity theft', 'stolen', 'lost access',
        'will be closed', 'will be suspended', 'will be terminated'
    }
    
    CREDENTIAL_WORDS = {
        'password', 'passwd', 'pass', 'login', 'log in', 'log-in',
        'signin', 'sign in', 'sign-in', 'username', 'user name',
        'user id', 'userid', 'email', 'e-mail', 'email address',
        'ssn', 'social security', 'social security number',
        'credit card', 'creditcard', 'card number', 'card details',
        'cvv', 'cvc', 'ccv', 'security code', 'expiration date',
        'expiry date', 'pin', 'pin number', 'atm pin', 'debit card',
        'bank account', 'account number', 'routing number', 'iban',
        'swift', 'bic', 'sort code', 'mother\'s maiden name',
        'date of birth', 'dob', 'passport', 'driver license',
        'driving license', 'id number', 'tax id', 'ein'
    }
    
    ACTION_WORDS = {
        'verify', 'confirm', 'validate', 'update', 'upgrade',
        'restore', 'recover', 'reactivate', 'unlock', 'secure',
        'protect', 'claim', 'redeem', 'activate', 'enable',
        'continue', 'proceed', 'submit', 'enter', 'provide',
        'click here', 'click below', 'click the link', 'click button',
        'tap here', 'press here', 'follow the link', 'open attachment'
    }
    
    REWARD_WORDS = {
        'win', 'winner', 'won', 'prize', 'reward', 'gift', 'free',
        'bonus', 'cashback', 'refund', 'compensation', 'inheritance',
        'lottery', 'jackpot', 'million', 'selected', 'chosen',
        'congratulations', 'congrats', 'lucky', 'exclusive offer',
        'special offer', 'limited offer', 'discount', 'savings'
    }
    
    BRAND_NAMES = {
        'paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook',
        'meta', 'instagram', 'whatsapp', 'netflix', 'spotify', 'twitter',
        'linkedin', 'dropbox', 'adobe', 'zoom', 'slack', 'github',
        'outlook', 'office', 'office365', 'onedrive', 'icloud', 'gmail',
        'yahoo', 'aol', 'hotmail', 'live.com', 'msn',
        'ebay', 'alibaba', 'aliexpress', 'wish', 'etsy', 'shopify',
        'walmart', 'target', 'bestbuy', 'costco', 'homedepot', 'lowes',
        'chase', 'wellsfargo', 'wells fargo', 'bankofamerica', 'bank of america',
        'citibank', 'citi', 'usbank', 'pnc', 'capitalone', 'capital one',
        'discover', 'americanexpress', 'american express', 'amex',
        'visa', 'mastercard', 'venmo', 'zelle', 'cashapp', 'cash app',
        'coinbase', 'binance', 'kraken', 'robinhood', 'fidelity',
        'usps', 'ups', 'fedex', 'dhl', 'royalmail', 'royal mail',
        'irs', 'hmrc', 'ssa', 'social security', 'medicare',
        'att', 'at&t', 'verizon', 'tmobile', 't-mobile', 'sprint',
        'hulu', 'disney', 'hbo', 'paramount', 'peacock', 'youtube',
        'airbnb', 'booking', 'expedia', 'tripadvisor'
    }
    
    def __init__(self, max_text_length: int = 15000):
        self.max_text_length = max_text_length
    
    def clean_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup
        except Exception as e:
            logger.warning(f"Error parsing HTML: {e}")
            return BeautifulSoup("", 'html.parser')
    
    def get_clean_soup(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Get soup with scripts/styles removed."""
        import copy
        clean_soup = copy.copy(soup)
        
        for element in clean_soup(['script', 'style', 'noscript', 'iframe',
                                   'svg', 'canvas', 'video', 'audio', 'source']):
            element.decompose()
        
        for comment in clean_soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        return clean_soup
    
    def extract_visible_text(self, soup: BeautifulSoup) -> str:
        """Extract visible text content."""
        try:
            clean_soup = self.get_clean_soup(soup)
            text = clean_soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            words = [w for w in text.split() if len(w) < 50]
            return ' '.join(words)[:self.max_text_length]
        except:
            return ""
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        try:
            title = soup.find('title')
            return title.get_text(strip=True)[:300] if title else ""
        except:
            return ""
    
    def extract_meta_info(self, soup: BeautifulSoup) -> Dict:
        """Extract meta information."""
        meta_info = {'description': '', 'keywords': ''}
        try:
            desc = soup.find('meta', attrs={'name': 'description'})
            if desc:
                meta_info['description'] = desc.get('content', '')[:500]
            
            keywords = soup.find('meta', attrs={'name': 'keywords'})
            if keywords:
                meta_info['keywords'] = keywords.get('content', '')[:500]
        except:
            pass
        return meta_info
    
    def analyze_forms(self, soup: BeautifulSoup) -> Dict:
        """Analyze forms for phishing indicators."""
        form_info = {
            'has_form': False,
            'has_password_field': False,
            'has_email_field': False,
            'has_credit_card_field': False,
            'has_ssn_field': False,
            'has_hidden_field': False,
            'num_input_fields': 0,
            'num_password_fields': 0,
            'external_form_action': False,
            'form_actions': [],
            'input_names': []
        }
        
        try:
            forms = soup.find_all('form')
            form_info['has_form'] = len(forms) > 0
            
            for form in forms:
                action = form.get('action', '')
                if action:
                    form_info['form_actions'].append(action[:200])
                    if action.startswith(('http://', 'https://')):
                        form_info['external_form_action'] = True
                
                inputs = form.find_all(['input', 'select', 'textarea'])
                form_info['num_input_fields'] += len(inputs)
                
                for inp in inputs:
                    inp_type = inp.get('type', 'text').lower()
                    inp_name = inp.get('name', '').lower()
                    inp_placeholder = inp.get('placeholder', '').lower()
                    
                    if inp_name:
                        form_info['input_names'].append(inp_name)
                    
                    if inp_type == 'hidden':
                        form_info['has_hidden_field'] = True
                    
                    if inp_type == 'password':
                        form_info['has_password_field'] = True
                        form_info['num_password_fields'] += 1
                    
                    if inp_type == 'email' or 'email' in inp_name or 'email' in inp_placeholder:
                        form_info['has_email_field'] = True
                    
                    cc_patterns = ['card', 'credit', 'cvv', 'cvc', 'ccv', 'expir', 'ccnum']
                    if any(p in inp_name or p in inp_placeholder for p in cc_patterns):
                        form_info['has_credit_card_field'] = True
                    
                    ssn_patterns = ['ssn', 'social', 'security']
                    if any(p in inp_name or p in inp_placeholder for p in ssn_patterns):
                        form_info['has_ssn_field'] = True
        except:
            pass
        
        return form_info
    
    def analyze_links(self, soup: BeautifulSoup, base_url: str = "") -> Dict:
        """Analyze links for suspicious patterns."""
        link_info = {
            'num_links': 0,
            'num_external_links': 0,
            'has_ip_based_link': False,
            'has_javascript_href': False,
            'href_text_mismatch': False,
            'suspicious_links': []
        }
        
        try:
            links = soup.find_all('a', href=True)
            link_info['num_links'] = len(links)
            
            base_domain = ""
            if base_url:
                try:
                    base_domain = urlparse(base_url).netloc.lower()
                except:
                    pass
            
            for link in links[:100]:
                href = link.get('href', '')
                text = link.get_text(strip=True)[:100]
                href_lower = href.lower()
                
                if href_lower.startswith('javascript:'):
                    link_info['has_javascript_href'] = True
                
                try:
                    parsed = urlparse(href)
                    domain = parsed.netloc.lower()
                    
                    if domain:
                        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
                            link_info['has_ip_based_link'] = True
                        
                        if base_domain and domain != base_domain:
                            link_info['num_external_links'] += 1
                        
                        # Check for brand-href mismatch
                        if text:
                            text_lower = text.lower()
                            for brand in self.BRAND_NAMES:
                                if brand in text_lower and brand not in domain:
                                    link_info['href_text_mismatch'] = True
                                    link_info['suspicious_links'].append({
                                        'text': text[:50],
                                        'href': href[:100],
                                        'brand': brand
                                    })
                except:
                    pass
        except:
            pass
        
        return link_info
    
    def analyze_scripts(self, soup: BeautifulSoup) -> Dict:
        """Analyze JavaScript for suspicious patterns."""
        script_info = {
            'num_scripts': 0,
            'has_eval': False,
            'has_obfuscation': False,
            'has_redirect_script': False,
            'has_keylogger_patterns': False
        }
        
        try:
            scripts = soup.find_all('script')
            script_info['num_scripts'] = len(scripts)
            
            for script in scripts:
                content = script.string or ''
                content_lower = content.lower()
                
                if 'eval(' in content_lower:
                    script_info['has_eval'] = True
                
                if re.search(r'\\x[0-9a-fA-F]{2}', content) or re.search(r'\\u[0-9a-fA-F]{4}', content):
                    script_info['has_obfuscation'] = True
                
                redirect_patterns = ['window.location', 'document.location', 'location.href']
                if any(p in content_lower for p in redirect_patterns):
                    script_info['has_redirect_script'] = True
                
                keylogger_patterns = ['onkeypress', 'onkeydown', 'onkeyup', 'keycode']
                if any(p in content_lower for p in keylogger_patterns):
                    script_info['has_keylogger_patterns'] = True
        except:
            pass
        
        return script_info
    
    def analyze_phishing_indicators(self, text: str, soup: BeautifulSoup, url: str = "") -> Dict:
        """Analyze text for phishing patterns."""
        text_lower = text.lower()
        
        indicators = {
            'urgency_count': 0,
            'threat_count': 0,
            'credential_count': 0,
            'action_count': 0,
            'reward_count': 0,
            'brand_mentions': [],
            'brand_url_mismatch': False,
            'suspicious_phrases': [],
            'phishing_score': 0
        }
        
        try:
            # Count keywords
            for word in self.URGENCY_WORDS:
                indicators['urgency_count'] += text_lower.count(word)
            
            for word in self.THREAT_WORDS:
                indicators['threat_count'] += text_lower.count(word)
            
            for word in self.CREDENTIAL_WORDS:
                indicators['credential_count'] += text_lower.count(word)
            
            for word in self.ACTION_WORDS:
                indicators['action_count'] += text_lower.count(word)
            
            for word in self.REWARD_WORDS:
                indicators['reward_count'] += text_lower.count(word)
            
            # Brand detection
            for brand in self.BRAND_NAMES:
                if brand in text_lower:
                    indicators['brand_mentions'].append(brand)
            
            # Brand-URL mismatch
            if url and indicators['brand_mentions']:
                url_lower = url.lower()
                for brand in indicators['brand_mentions']:
                    if brand not in url_lower:
                        indicators['brand_url_mismatch'] = True
                        break
            
            # Suspicious patterns
            suspicious_patterns = [
                (r'verify\s+your\s+(account|identity)', 'verify_account'),
                (r'confirm\s+your\s+(account|identity|information)', 'confirm_account'),
                (r'update\s+your\s+(account|payment|billing)', 'update_account'),
                (r'(account|access)\s+(suspended|locked|limited)', 'account_suspended'),
                (r'unusual\s+(activity|sign.?in|login)', 'unusual_activity'),
                (r'click\s+(here|below|the\s+link)', 'click_to_action'),
                (r'within\s+\d+\s+(hours?|days?)', 'time_pressure'),
            ]
            
            for pattern, name in suspicious_patterns:
                if re.search(pattern, text_lower):
                    indicators['suspicious_phrases'].append(name)
            
            # Calculate phishing score
            form_info = self.analyze_forms(soup)
            indicators['phishing_score'] = min(100, (
                indicators['urgency_count'] * 10 +
                indicators['threat_count'] * 15 +
                indicators['credential_count'] * 12 +
                len(indicators['suspicious_phrases']) * 20 +
                (30 if indicators['brand_url_mismatch'] else 0) +
                (20 if form_info['has_password_field'] else 0) +
                (25 if form_info['has_credit_card_field'] else 0)
            ))
        except:
            pass
        
        return indicators
    
    def process_html(self, html_content: str, url: str = "") -> Dict:
        """Process HTML and extract all features."""
        soup = self.clean_html(html_content)
        
        visible_text = self.extract_visible_text(soup)
        title = self.extract_title(soup)
        meta_info = self.extract_meta_info(soup)
        form_info = self.analyze_forms(soup)
        link_info = self.analyze_links(soup, url)
        script_info = self.analyze_scripts(soup)
        phishing_indicators = self.analyze_phishing_indicators(visible_text, soup, url)
        
        return {
            'visible_text': visible_text,
            'title': title,
            'meta_description': meta_info.get('description', ''),
            'form_info': form_info,
            'link_info': link_info,
            'script_info': script_info,
            'phishing_indicators': phishing_indicators,
            'text_length': len(visible_text)
        }
    
    def create_model_input(self, features: Dict, url: str = "") -> str:
        """Create structured text input for ELECTRA model."""
        parts = []
        
        # URL
        if url:
            parts.append(f"[URL] {url[:300]}")
        
        # Title
        if features.get('title'):
            parts.append(f"[TITLE] {features['title'][:300]}")
        
        # Meta
        if features.get('meta_description'):
            parts.append(f"[META] {features['meta_description'][:300]}")
        
        # Form indicators
        form_info = features.get('form_info', {})
        form_flags = []
        if form_info.get('has_form'):
            form_flags.append("has-form")
        if form_info.get('has_password_field'):
            form_flags.append("PASSWORD-FIELD")
        if form_info.get('has_email_field'):
            form_flags.append("email-field")
        if form_info.get('has_credit_card_field'):
            form_flags.append("CREDIT-CARD-FIELD")
        if form_info.get('has_ssn_field'):
            form_flags.append("SSN-FIELD")
        if form_info.get('external_form_action'):
            form_flags.append("EXTERNAL-ACTION")
        
        if form_flags:
            parts.append(f"[FORM] {', '.join(form_flags)}")
        
        # Security indicators
        phishing = features.get('phishing_indicators', {})
        security_flags = []
        
        if phishing.get('urgency_count', 0) > 0:
            security_flags.append(f"urgency:{phishing['urgency_count']}")
        if phishing.get('threat_count', 0) > 0:
            security_flags.append(f"THREAT:{phishing['threat_count']}")
        if phishing.get('credential_count', 0) > 0:
            security_flags.append(f"credential-request:{phishing['credential_count']}")
        if phishing.get('brand_url_mismatch'):
            security_flags.append("BRAND-URL-MISMATCH")
        if len(phishing.get('suspicious_phrases', [])) > 0:
            security_flags.append(f"suspicious-patterns:{len(phishing['suspicious_phrases'])}")
        
        # Script indicators
        script_info = features.get('script_info', {})
        if script_info.get('has_eval'):
            security_flags.append("eval-detected")
        if script_info.get('has_obfuscation'):
            security_flags.append("OBFUSCATION")
        if script_info.get('has_keylogger_patterns'):
            security_flags.append("KEYLOGGER-PATTERN")
        
        # Link indicators
        link_info = features.get('link_info', {})
        if link_info.get('has_ip_based_link'):
            security_flags.append("IP-BASED-LINK")
        if link_info.get('href_text_mismatch'):
            security_flags.append("LINK-TEXT-MISMATCH")
        
        if security_flags:
            parts.append(f"[SECURITY] {', '.join(security_flags)}")
        
        # Brands
        brands = phishing.get('brand_mentions', [])
        if brands:
            parts.append(f"[BRANDS] {', '.join(brands[:10])}")
        
        # Patterns
        patterns = phishing.get('suspicious_phrases', [])
        if patterns:
            parts.append(f"[PATTERNS] {', '.join(patterns)}")
        
        # Content
        if features.get('visible_text'):
            parts.append(f"[CONTENT] {features['visible_text'][:5000]}")
        
        return ' '.join(parts)


class HTMLAnalyzer:
    """
    HTML Analyzer for phishing detection using trained ELECTRA model.
    
    Features:
    - Load trained ELECTRA model
    - Preprocess HTML content
    - Predict phishing probability
    - Fetch and analyze URLs
    - Detailed analysis results
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the HTML Analyzer.
        
        Args:
            model_path: Path to trained model directory
            config_path: Path to config.json
        """
        base_dir = r"C:\PhishLens"
        self.model_path = model_path or os.path.join(base_dir, "backend", "models", "html", "final_model")
        self.config_path = config_path or os.path.join(base_dir, "backend", "inference", "config.json")
        
        # Load config
        self.config = self._load_config()
        
        # Set device (CPU for RTX 5070 compatibility)
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize preprocessor
        self.preprocessor = HTMLPreprocessor()
        
        # Load model
        self._load_model()
        
        # HTTP session for fetching URLs
        self.session = self._create_session()
    
    def _load_config(self) -> dict:
        """Load configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_model(self):
        """Load trained ELECTRA model."""
        logger.info(f"Loading model from: {self.model_path}")
        
        self.tokenizer = ElectraTokenizer.from_pretrained(self.model_path)
        self.model = ElectraForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session for fetching URLs."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        return session
    
    def fetch_html(self, url: str, timeout: int = 15) -> Tuple[Optional[str], str]:
        """
        Fetch HTML content from URL, following redirects.
        
        Args:
            url: URL to fetch
            timeout: Request timeout
            
        Returns:
            Tuple of (HTML content or None, final URL after redirects)
        """
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        try:
            response = self.session.get(
                url,
                timeout=timeout,
                verify=False,
                allow_redirects=True
            )
            
            # Capture final URL after redirects
            final_url = response.url
            if final_url != url:
                logger.info(f"HTML redirect resolved: {url} → {final_url}")
            
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type or 'text/plain' in content_type:
                return response.text, final_url
            
            return None, final_url
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching URL {url}: {e}")
            return None, url
    
    def analyze_html(
        self,
        html_content: str,
        url: str = "",
        return_details: bool = True
    ) -> Dict:
        """
        Analyze HTML content for phishing.
        
        Args:
            html_content: Raw HTML string
            url: Original URL (for context)
            return_details: Whether to return detailed analysis
            
        Returns:
            Analysis result dictionary
        """
        result = {
            "url": url,
            "is_phishing": False,
            "confidence": 0.0,
            "label": "legitimate",
            "risk_score": 0.0,
            "details": {}
        }
        
        if not html_content or len(html_content) < 50:
            result["error"] = "HTML content too short or empty"
            return result
        
        try:
            # Extract features
            features = self.preprocessor.process_html(html_content, url)
            
            # Create model input
            model_input = self.preprocessor.create_model_input(features, url)
            
            # Tokenize
            encoding = self.tokenizer(
                model_input,
                add_special_tokens=True,
                max_length=512,
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
                
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                phishing_prob = probabilities[0][1].item()
            
            # Set results
            is_phishing = predicted_class == 1
            
            result["is_phishing"] = is_phishing
            result["confidence"] = round(confidence * 100, 2)
            result["label"] = "phishing" if is_phishing else "legitimate"
            result["risk_score"] = round(phishing_prob * 100, 2)
            
            # Add details
            if return_details:
                result["details"] = {
                    "title": features.get('title', ''),
                    "text_length": features.get('text_length', 0),
                    "form_analysis": features.get('form_info', {}),
                    "link_analysis": features.get('link_info', {}),
                    "script_analysis": features.get('script_info', {}),
                    "phishing_indicators": features.get('phishing_indicators', {}),
                    "probabilities": {
                        "legitimate": round(probabilities[0][0].item() * 100, 2),
                        "phishing": round(probabilities[0][1].item() * 100, 2)
                    }
                }
        
        except Exception as e:
            logger.error(f"Error analyzing HTML: {e}")
            result["error"] = str(e)
        
        return result
    
    def analyze_url(
        self,
        url: str,
        timeout: int = 15,
        return_details: bool = True
    ) -> Dict:
        """
        Fetch URL and analyze its HTML content.
        Follows redirects and uses the final URL for analysis.
        
        Args:
            url: URL to analyze
            timeout: Fetch timeout
            return_details: Whether to return detailed analysis
            
        Returns:
            Analysis result dictionary
        """
        result = {
            "url": url,
            "is_phishing": False,
            "confidence": 0.0,
            "label": "legitimate",
            "risk_score": 0.0,
            "html_fetched": False,
            "details": {}
        }
        
        # Fetch HTML (returns final URL after redirects)
        logger.info(f"Fetching HTML from: {url}")
        html_content, final_url = self.fetch_html(url, timeout)
        
        if not html_content:
            result["error"] = "Failed to fetch HTML content"
            result["html_fetched"] = False
            return result
        
        result["html_fetched"] = True
        result["html_length"] = len(html_content)
        
        # Use final URL (after redirects) for analysis context
        # This prevents short URL domains (bit.ly) from poisoning feature extraction
        analysis_url = final_url if final_url else url
        
        if final_url != url:
            result["redirect_detected"] = True
            result["final_url"] = final_url
            logger.info(f"HTML analyzing with resolved URL: {analysis_url}")
        
        # Analyze HTML with the resolved URL
        analysis = self.analyze_html(html_content, analysis_url, return_details)
        
        # Merge results
        result.update(analysis)
        result["html_fetched"] = True
        
        return result
    
    def get_risk_level(self, risk_score: float) -> str:
        """Get human-readable risk level."""
        if risk_score < 20:
            return "LOW"
        elif risk_score < 50:
            return "MEDIUM"
        elif risk_score < 80:
            return "HIGH"
        else:
            return "CRITICAL"


def main():
    """Test the HTML analyzer."""
    print("=" * 60)
    print("PhishLens HTML Analyzer - Test")
    print("=" * 60)
    
    # Initialize analyzer
    print("\nLoading model...")
    analyzer = HTMLAnalyzer()
    print("Model loaded!\n")
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "https://www.paypal.com",
        "https://www.microsoft.com",
    ]
    
    print("-" * 60)
    print("Testing URL Analysis:")
    print("-" * 60)
    
    for url in test_urls:
        print(f"\nAnalyzing: {url}")
        result = analyzer.analyze_url(url, return_details=False)
        
        status = "PHISHING" if result["is_phishing"] else "SAFE"
        print(f"  Status: {status}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Risk Score: {result['risk_score']}%")
        print(f"  Risk Level: {analyzer.get_risk_level(result['risk_score'])}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
