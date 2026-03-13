"""
PhishLens - HTML Preprocessor for ELECTRA Training
Extracts and cleans text content from HTML for model training.

Features extracted:
- Visible text content
- Form fields (login, password, etc.)
- Link text and URLs
- Meta information
- Script indicators
- Phishing language patterns
"""

import os
import re
import json
import pandas as pd
from bs4 import BeautifulSoup, Comment
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTMLPreprocessor:
    """
    Preprocesses HTML content for ELECTRA model training.
    
    Extracts meaningful text and structural features while
    removing noise (scripts, styles, etc.)
    """
    
    # Phishing indicator keywords
    URGENCY_WORDS = {
        'urgent', 'immediately', 'suspended', 'locked', 'limited',
        'verify', 'confirm', 'update', 'expire', 'hours', 'minutes',
        'action required', 'act now', 'deadline', 'important'
    }
    
    THREAT_WORDS = {
        'suspended', 'terminated', 'locked', 'disabled', 'unauthorized',
        'unusual activity', 'suspicious', 'breach', 'compromised',
        'illegal', 'violation', 'restricted'
    }
    
    CREDENTIAL_WORDS = {
        'password', 'login', 'signin', 'sign in', 'log in', 'username',
        'email', 'ssn', 'social security', 'credit card', 'cvv', 'pin',
        'account number', 'routing number', 'bank account'
    }
    
    BRAND_NAMES = {
        'paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook',
        'instagram', 'netflix', 'bank of america', 'wells fargo', 'chase',
        'citibank', 'usps', 'fedex', 'dhl', 'irs', 'dropbox', 'linkedin',
        'twitter', 'whatsapp', 'outlook', 'office365', 'adobe'
    }
    
    def __init__(self, max_text_length: int = 10000):
        """
        Initialize the preprocessor.
        
        Args:
            max_text_length: Maximum text length to keep
        """
        self.max_text_length = max_text_length
    
    def clean_html(self, html_content: str) -> BeautifulSoup:
        """Parse and clean HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'noscript', 'iframe', 
                                'svg', 'canvas', 'video', 'audio']):
                element.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            return soup
        except Exception as e:
            logger.warning(f"Error parsing HTML: {e}")
            return BeautifulSoup("", 'html.parser')
    
    def extract_visible_text(self, soup: BeautifulSoup) -> str:
        """Extract visible text from HTML."""
        try:
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove very long strings (likely encoded data)
            words = text.split()
            words = [w for w in words if len(w) < 50]
            text = ' '.join(words)
            
            return text[:self.max_text_length]
        except:
            return ""
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        try:
            title = soup.find('title')
            return title.get_text(strip=True) if title else ""
        except:
            return ""
    
    def extract_meta_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract meta tags information."""
        meta_info = {}
        try:
            # Description
            desc = soup.find('meta', attrs={'name': 'description'})
            if desc:
                meta_info['description'] = desc.get('content', '')[:500]
            
            # Keywords
            keywords = soup.find('meta', attrs={'name': 'keywords'})
            if keywords:
                meta_info['keywords'] = keywords.get('content', '')[:500]
            
            # OG title
            og_title = soup.find('meta', attrs={'property': 'og:title'})
            if og_title:
                meta_info['og_title'] = og_title.get('content', '')[:200]
        except:
            pass
        
        return meta_info
    
    def extract_forms(self, soup: BeautifulSoup) -> Dict:
        """Extract form information."""
        form_info = {
            'has_form': False,
            'has_password_field': False,
            'has_email_field': False,
            'has_credit_card_field': False,
            'num_input_fields': 0,
            'form_action': '',
            'input_types': []
        }
        
        try:
            forms = soup.find_all('form')
            if forms:
                form_info['has_form'] = True
                
                for form in forms:
                    # Get form action
                    action = form.get('action', '')
                    if action:
                        form_info['form_action'] = action[:200]
                    
                    # Find inputs
                    inputs = form.find_all('input')
                    form_info['num_input_fields'] += len(inputs)
                    
                    for inp in inputs:
                        inp_type = inp.get('type', 'text').lower()
                        inp_name = inp.get('name', '').lower()
                        inp_placeholder = inp.get('placeholder', '').lower()
                        
                        form_info['input_types'].append(inp_type)
                        
                        # Check for password
                        if inp_type == 'password' or 'password' in inp_name or 'password' in inp_placeholder:
                            form_info['has_password_field'] = True
                        
                        # Check for email
                        if inp_type == 'email' or 'email' in inp_name or 'email' in inp_placeholder:
                            form_info['has_email_field'] = True
                        
                        # Check for credit card
                        cc_indicators = ['card', 'credit', 'cvv', 'ccv', 'expir']
                        if any(ind in inp_name or ind in inp_placeholder for ind in cc_indicators):
                            form_info['has_credit_card_field'] = True
        except:
            pass
        
        return form_info
    
    def extract_links(self, soup: BeautifulSoup) -> Dict:
        """Extract link information."""
        link_info = {
            'num_links': 0,
            'num_external_links': 0,
            'has_suspicious_links': False,
            'link_texts': []
        }
        
        try:
            links = soup.find_all('a', href=True)
            link_info['num_links'] = len(links)
            
            for link in links[:50]:  # Limit to first 50 links
                href = link.get('href', '')
                text = link.get_text(strip=True)[:100]
                
                if text:
                    link_info['link_texts'].append(text)
                
                # Check for external links
                if href.startswith(('http://', 'https://')):
                    link_info['num_external_links'] += 1
                
                # Check for suspicious patterns
                if '@' in href or 'javascript:' in href.lower():
                    link_info['has_suspicious_links'] = True
        except:
            pass
        
        return link_info
    
    def analyze_phishing_indicators(self, text: str, soup: BeautifulSoup) -> Dict:
        """Analyze text for phishing indicators."""
        text_lower = text.lower()
        
        indicators = {
            'urgency_score': 0,
            'threat_score': 0,
            'credential_score': 0,
            'brand_mentions': [],
            'has_login_form': False,
            'suspicious_patterns': []
        }
        
        try:
            # Count urgency words
            for word in self.URGENCY_WORDS:
                if word in text_lower:
                    indicators['urgency_score'] += 1
            
            # Count threat words
            for word in self.THREAT_WORDS:
                if word in text_lower:
                    indicators['threat_score'] += 1
            
            # Count credential words
            for word in self.CREDENTIAL_WORDS:
                if word in text_lower:
                    indicators['credential_score'] += 1
            
            # Find brand mentions
            for brand in self.BRAND_NAMES:
                if brand in text_lower:
                    indicators['brand_mentions'].append(brand)
            
            # Check for login form
            form_info = self.extract_forms(soup)
            if form_info['has_password_field']:
                indicators['has_login_form'] = True
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'verify\s+your\s+(account|identity)',
                r'confirm\s+your\s+(account|identity|information)',
                r'update\s+your\s+(account|payment|information)',
                r'account\s+(suspended|locked|limited)',
                r'unusual\s+(activity|sign-?in)',
                r'click\s+here\s+to\s+(verify|confirm|update)',
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, text_lower):
                    indicators['suspicious_patterns'].append(pattern)
        except:
            pass
        
        return indicators
    
    def process_html(self, html_content: str) -> Dict:
        """
        Process HTML and extract all features.
        
        Args:
            html_content: Raw HTML string
            
        Returns:
            Dictionary with extracted features
        """
        # Clean HTML
        soup = self.clean_html(html_content)
        
        # Extract features
        visible_text = self.extract_visible_text(soup)
        title = self.extract_title(soup)
        meta_info = self.extract_meta_info(soup)
        form_info = self.extract_forms(soup)
        link_info = self.extract_links(soup)
        phishing_indicators = self.analyze_phishing_indicators(visible_text, soup)
        
        return {
            'visible_text': visible_text,
            'title': title,
            'meta_description': meta_info.get('description', ''),
            'has_form': form_info['has_form'],
            'has_password_field': form_info['has_password_field'],
            'has_email_field': form_info['has_email_field'],
            'has_credit_card_field': form_info['has_credit_card_field'],
            'num_input_fields': form_info['num_input_fields'],
            'num_links': link_info['num_links'],
            'num_external_links': link_info['num_external_links'],
            'has_suspicious_links': link_info['has_suspicious_links'],
            'urgency_score': phishing_indicators['urgency_score'],
            'threat_score': phishing_indicators['threat_score'],
            'credential_score': phishing_indicators['credential_score'],
            'brand_mentions': phishing_indicators['brand_mentions'],
            'has_login_form': phishing_indicators['has_login_form'],
            'num_suspicious_patterns': len(phishing_indicators['suspicious_patterns'])
        }
    
    def create_model_input(self, features: Dict, url: str = "") -> str:
        """
        Create structured text input for ELECTRA model.
        
        Args:
            features: Extracted features dictionary
            url: Original URL (optional)
            
        Returns:
            Structured text for model input
        """
        parts = []
        
        # Title
        if features.get('title'):
            parts.append(f"[TITLE] {features['title'][:200]}")
        
        # URL context (if provided)
        if url:
            parts.append(f"[URL] {url[:200]}")
        
        # Meta description
        if features.get('meta_description'):
            parts.append(f"[META] {features['meta_description'][:300]}")
        
        # Form indicators
        form_flags = []
        if features.get('has_form'):
            form_flags.append("form-present")
        if features.get('has_password_field'):
            form_flags.append("password-field")
        if features.get('has_email_field'):
            form_flags.append("email-field")
        if features.get('has_credit_card_field'):
            form_flags.append("credit-card-field")
        if features.get('has_login_form'):
            form_flags.append("login-form")
        
        if form_flags:
            parts.append(f"[FORM] {', '.join(form_flags)}")
        
        # Security indicators
        security_flags = []
        if features.get('urgency_score', 0) > 0:
            security_flags.append(f"urgency:{features['urgency_score']}")
        if features.get('threat_score', 0) > 0:
            security_flags.append(f"threat:{features['threat_score']}")
        if features.get('credential_score', 0) > 0:
            security_flags.append(f"credential-request:{features['credential_score']}")
        if features.get('has_suspicious_links'):
            security_flags.append("suspicious-links")
        if features.get('num_suspicious_patterns', 0) > 0:
            security_flags.append(f"suspicious-patterns:{features['num_suspicious_patterns']}")
        
        if security_flags:
            parts.append(f"[SECURITY] {', '.join(security_flags)}")
        
        # Brand mentions
        if features.get('brand_mentions'):
            parts.append(f"[BRANDS] {', '.join(features['brand_mentions'][:5])}")
        
        # Main content (truncated)
        if features.get('visible_text'):
            content = features['visible_text'][:3000]  # Limit content length
            parts.append(f"[CONTENT] {content}")
        
        return ' '.join(parts)


def process_single_html(args):
    """Process a single HTML file (for parallel processing)."""
    row, html_dir, preprocessor = args
    
    try:
        html_path = os.path.join(html_dir, row['html_file'])
        
        if not os.path.exists(html_path):
            return None
        
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        if not html_content or len(html_content) < 100:
            return None
        
        # Process HTML
        features = preprocessor.process_html(html_content)
        
        # Create model input
        model_input = preprocessor.create_model_input(features, row.get('url', ''))
        
        if not model_input or len(model_input) < 50:
            return None
        
        return {
            'id': row['id'],
            'url': row['url'],
            'label': row['label'],
            'processed_text': model_input,
            'title': features.get('title', ''),
            'has_form': int(features.get('has_form', False)),
            'has_password_field': int(features.get('has_password_field', False)),
            'has_login_form': int(features.get('has_login_form', False)),
            'urgency_score': features.get('urgency_score', 0),
            'threat_score': features.get('threat_score', 0),
            'credential_score': features.get('credential_score', 0),
            'num_suspicious_patterns': features.get('num_suspicious_patterns', 0),
            'text_length': len(features.get('visible_text', ''))
        }
    except Exception as e:
        return None


def main():
    """Main function to preprocess HTML dataset."""
    
    # Paths
    base_dir = r"C:\PhishLens"
    html_dataset_path = os.path.join(base_dir, "data", "raw", "html", "phishlens_html_dataset.csv")
    html_files_dir = os.path.join(base_dir, "data", "raw", "html", "html_files")
    output_path = os.path.join(base_dir, "data", "processed", "html", "processed_html.csv")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 60)
    print("PhishLens HTML Preprocessor")
    print("=" * 60)
    
    # Load HTML dataset
    print(f"\nLoading HTML dataset from: {html_dataset_path}")
    df = pd.read_csv(html_dataset_path)
    print(f"Total HTML samples: {len(df)}")
    print(f"Phishing: {len(df[df['label'] == 1])}")
    print(f"Legitimate: {len(df[df['label'] == 0])}")
    
    # Initialize preprocessor
    preprocessor = HTMLPreprocessor(max_text_length=10000)
    
    # Process HTML files
    print(f"\nProcessing HTML files...")
    print("This may take 10-30 minutes depending on your CPU.\n")
    
    processed_data = []
    errors = 0
    
    # Process with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing HTML"):
        result = process_single_html((row, html_files_dir, preprocessor))
        
        if result:
            processed_data.append(result)
        else:
            errors += 1
    
    # Create DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    # Save
    processed_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total processed: {len(processed_df)}")
    print(f"Errors/Skipped: {errors}")
    print(f"Phishing samples: {len(processed_df[processed_df['label'] == 1])}")
    print(f"Legitimate samples: {len(processed_df[processed_df['label'] == 0])}")
    print(f"\nSaved to: {output_path}")
    
    # Show sample
    print("\n" + "-" * 60)
    print("Sample processed text (first 500 chars):")
    print("-" * 60)
    print(processed_df['processed_text'].iloc[0][:500])
    print("-" * 60)
    
    # Show feature statistics
    print("\nFeature Statistics:")
    feature_cols = ['has_form', 'has_password_field', 'has_login_form', 
                    'urgency_score', 'threat_score', 'credential_score']
    print(processed_df[feature_cols].describe())


if __name__ == "__main__":
    main()