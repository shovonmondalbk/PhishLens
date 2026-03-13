"""
PhishLens - Visual Analyzer V5.0 (Known Brand URL Protection)

V5.0 CHANGES:
    - NEW: "Known Brand URL" check prevents CLIP false positives on legitimate brand sites
    - If URL belongs to Brand A (e.g., x.com → Twitter), CLIP detecting Brand B (e.g., Google)
      is treated as a CLIP false positive, NOT brand impersonation
    - Increased CLIP confidence thresholds to reduce false matches
    - Better handling of brand aliases (x.com = twitter.com)

PREVIOUS FIXES (still active):
    - V4.9: Redirect-aware, captures FINAL URL after all redirects
    - V4.8: Subprocess isolation for thread-safety

DETECTION LOGIC (V5.0):
    1. Capture screenshot → get FINAL URL after redirects
    2. Check if URL belongs to a known brand ("url_brand")
    3. OCR brand detection:
        - If OCR brand matches url_brand → LEGITIMATE (same brand)
        - If OCR brand != url_brand AND url_brand exists → LEGITIMATE (CLIP/OCR confusion between brands)
        - If OCR brand detected + domain mismatch + NO url_brand → PHISHING (high confidence)
    4. CLIP brand detection:
        - If url_brand exists → SKIP CLIP mismatch (known brand site won't impersonate another)
        - If CLIP brand + mismatch + login indicators + NO url_brand → PHISHING (medium confidence)
        - If CLIP brand + mismatch + NO login indicators → LEGITIMATE (false positive)
    5. No brand detected → LEGITIMATE
"""

import os
import sys
import io
import json
import logging
import subprocess
import tempfile
import uuid
import shutil
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from PIL import Image
import numpy as np

sys.path.insert(0, r"C:\PhishLens")

# ============================================================================
# PyTorch / RTX 5070 Compatibility
# ============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
torch.cuda.is_available = lambda: False

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Embedded Screenshot Worker Script (V4.9 - Returns Final URL)
# ============================================================================
SCREENSHOT_WORKER_CODE = '''
import sys
import os

def capture(url, output_path, width=1280, height=720, timeout=30000):
    """
    Capture screenshot and return final URL after redirects.
    Writes final URL to {output_path}.url file.
    """
    final_url = url  # Default to input URL
    
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-software-rasterizer',
                    '--disable-extensions'
                ]
            )
            
            page = browser.new_page(viewport={'width': width, 'height': height})
            
            try:
                page.goto(url, timeout=timeout, wait_until='domcontentloaded')
            except Exception:
                try:
                    page.goto(url, timeout=timeout, wait_until='commit')
                except Exception as e:
                    browser.close()
                    return False, url
            
            page.wait_for_timeout(2000)
            
            # CRITICAL: Get final URL after all redirects
            final_url = page.url
            
            page.screenshot(path=output_path, type='png', full_page=False)
            
            # Write final URL to sidecar file
            url_file = output_path + '.url'
            with open(url_file, 'w', encoding='utf-8') as f:
                f.write(final_url)
            
            page.close()
            browser.close()
            
            return os.path.exists(output_path), final_url
            
    except Exception as e:
        # Still try to write URL file even on error
        try:
            url_file = output_path + '.url'
            with open(url_file, 'w', encoding='utf-8') as f:
                f.write(final_url)
        except:
            pass
        return False, final_url

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    
    url = sys.argv[1]
    output_path = sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
    height = int(sys.argv[4]) if len(sys.argv) > 4 else 720
    timeout = int(sys.argv[5]) if len(sys.argv) > 5 else 30000
    
    success, final_url = capture(url, output_path, width, height, timeout)
    
    # Print final URL to stdout for parent process
    print(f"FINAL_URL:{final_url}")
    
    sys.exit(0 if success else 1)
'''


class VisualAnalyzer:
    """
    Visual Analyzer V5.0 - Known Brand URL Protection + Redirect-Aware
    
    Key improvements over V4.9:
    - Known Brand URL check: If URL belongs to Brand A, CLIP detecting Brand B
      is treated as a false positive (a real brand won't impersonate another brand)
    - Higher CLIP confidence threshold (40% instead of 30%) to reduce false matches
    - Wider confidence gap requirement (8% instead of 5%) for more decisive CLIP matches
    - Subprocess isolation for 100% thread-safety (from V4.8)
    - Returns final URL after redirects for accurate brand-domain comparison (from V4.9)
    """
    
    DEFAULT_CONFIG = r"C:\PhishLens\backend\inference\config.json"
    DEFAULT_BRAND_DOMAINS = r"C:\PhishLens\backend\inference\brand_domains.json"
    
    # Brands we check for
    KNOWN_BRANDS = sorted([
        "Amazon", "American Express", "Apple", "AT&T", "Bank of America",
        "Binance", "Capital One", "Cash App", "Chase", "Citibank", "Coinbase",
        "DHL", "Dropbox", "eBay", "Facebook", "FedEx", "GitHub", "Google",
        "HSBC", "iCloud", "Instagram", "LinkedIn", "Microsoft", "Netflix",
        "Outlook", "PayPal", "Slack", "Spotify", "Steam", "Stripe",
        "T-Mobile", "Twitter", "UPS", "USPS", "Venmo", "Verizon",
        "Visa", "Mastercard", "Walmart", "Wells Fargo", "WhatsApp",
        "Yahoo", "Zoom", "Adobe"
    ])
    
    # OCR keywords for brand detection
    BRAND_OCR_KEYWORDS = {
        "paypal": ["paypal"],
        "amazon": ["amazon"],
        "google": ["google", "gmail"],
        "facebook": ["facebook"],
        "instagram": ["instagram"],
        "microsoft": ["microsoft", "outlook", "office 365", "office365"],
        "apple": ["apple", "icloud"],
        "netflix": ["netflix"],
        "twitter": ["twitter"],
        "linkedin": ["linkedin"],
        "dropbox": ["dropbox"],
        "github": ["github"],
        "yahoo": ["yahoo"],
        "ebay": ["ebay"],
        "chase": ["chase bank", "chase"],
        "wells fargo": ["wells fargo", "wellsfargo"],
        "bank of america": ["bank of america", "bankofamerica"],
        "citibank": ["citibank", "citi"],
        "capital one": ["capital one", "capitalone"],
        "usps": ["usps", "united states postal"],
        "fedex": ["fedex", "federal express"],
        "dhl": ["dhl"],
        "ups": ["ups"],
        "coinbase": ["coinbase"],
        "binance": ["binance"],
        "whatsapp": ["whatsapp"],
        "zoom": ["zoom"],
        "slack": ["slack"],
        "spotify": ["spotify"],
        "steam": ["steam"],
        "venmo": ["venmo"],
        "cash app": ["cash app", "cashapp"],
        "adobe": ["adobe"],
        "stripe": ["stripe"],
    }
    
    # LOGIN INDICATORS - if these appear, the page is likely a login/credential page
    LOGIN_INDICATORS = [
        # English
        "login", "log in", "log-in", "signin", "sign in", "sign-in",
        "password", "passwd", "enter password",
        "username", "user name", "email address", "phone number",
        "verify", "verification", "confirm", "authenticate",
        "account", "my account", "your account",
        "continue", "submit", "next",
        "forgot password", "reset password", "recover",
        "secure", "security", "protected",
        "credit card", "card number", "cvv", "expiry",
        "ssn", "social security",
        "billing", "payment",
        "update", "confirm your", "verify your",
        "suspended", "limited", "unusual activity",
        "unlock", "restore", "reactivate",
    ]
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        brand_domains_path: Optional[str] = None,
        screenshot_width: int = 1280,
        screenshot_height: int = 720
    ):
        self.config_path = config_path or self.DEFAULT_CONFIG
        self.brand_domains_path = brand_domains_path or self.DEFAULT_BRAND_DOMAINS
        self.screenshot_width = screenshot_width
        self.screenshot_height = screenshot_height
        
        self.config = self._load_json(self.config_path) or {}
        self.brand_domains = self._load_brand_domains()
        self._domain_to_brand = self._build_domain_lookup()
        
        # Setup subprocess infrastructure
        self._setup_subprocess_worker()
        
        # CLIP and OCR are thread-safe, initialize once
        self._init_clip()
        self._init_ocr()
        
        if self.clip_model:
            self._precompute_clip_embeddings()
        
        logger.info("=" * 50)
        logger.info("Visual Analyzer V5.0 initialized!")
        logger.info("  Screenshot: Subprocess isolation (100%% thread-safe)")
        logger.info("  Redirect: Final URL captured for brand comparison")
        logger.info("  NEW: Known Brand URL protection (prevents cross-brand false positives)")
        logger.info("  CLIP: %s", "Loaded" if self.clip_model else "NOT LOADED")
        logger.info("  OCR: %s", "Loaded" if self.ocr_reader else "NOT LOADED")
        logger.info("  Brands: %d", len(self.KNOWN_BRANDS))
        logger.info("=" * 50)
    
    def _setup_subprocess_worker(self):
        """Create temp directory and worker script for subprocess screenshot capture."""
        self.temp_dir = tempfile.mkdtemp(prefix="phishlens_")
        self.worker_script = os.path.join(self.temp_dir, "screenshot_worker.py")
        with open(self.worker_script, 'w', encoding='utf-8') as f:
            f.write(SCREENSHOT_WORKER_CODE)
        self.python_exe = sys.executable
        logger.info("Subprocess worker setup: %s", self.worker_script)
    
    def _load_json(self, path: str) -> Optional[dict]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def _load_brand_domains(self) -> dict:
        data = self._load_json(self.brand_domains_path) or {}
        
        defaults = {
            "PayPal": ["paypal.com", "paypal.me"],
            "Google": ["google.com", "gmail.com", "youtube.com", "accounts.google.com"],
            "Amazon": ["amazon.com", "amazon.co.uk", "amazon.de", "amazon.in", "amazon.ca"],
            "Facebook": ["facebook.com", "fb.com", "messenger.com", "meta.com"],
            "Microsoft": ["microsoft.com", "live.com", "outlook.com", "office.com", "office365.com", "microsoftonline.com"],
            "Apple": ["apple.com", "icloud.com", "appleid.apple.com"],
            "Netflix": ["netflix.com"],
            "Instagram": ["instagram.com"],
            "Twitter": ["twitter.com", "x.com"],
            "LinkedIn": ["linkedin.com"],
            "Yahoo": ["yahoo.com", "mail.yahoo.com"],
            "GitHub": ["github.com"],
            "Dropbox": ["dropbox.com"],
            "eBay": ["ebay.com", "ebay.co.uk"],
            "Chase": ["chase.com"],
            "Wells Fargo": ["wellsfargo.com"],
            "Bank of America": ["bankofamerica.com", "bofa.com"],
            "Citibank": ["citi.com", "citibank.com"],
            "Capital One": ["capitalone.com"],
            "USPS": ["usps.com"],
            "FedEx": ["fedex.com"],
            "DHL": ["dhl.com"],
            "UPS": ["ups.com"],
            "Coinbase": ["coinbase.com"],
            "Binance": ["binance.com", "binance.us"],
            "WhatsApp": ["whatsapp.com", "web.whatsapp.com"],
            "Zoom": ["zoom.us", "zoom.com"],
            "Slack": ["slack.com"],
            "Spotify": ["spotify.com"],
            "Steam": ["steampowered.com", "store.steampowered.com"],
            "Venmo": ["venmo.com"],
            "Cash App": ["cash.app"],
            "Adobe": ["adobe.com"],
            "Stripe": ["stripe.com"],
            "American Express": ["americanexpress.com", "amex.com"],
            "Visa": ["visa.com"],
            "Mastercard": ["mastercard.com"],
        }
        
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
        
        return {k: v for k, v in data.items() if not k.startswith('_')}
    
    def _build_domain_lookup(self) -> Dict[str, str]:
        lookup = {}
        for brand, domains in self.brand_domains.items():
            for domain in domains:
                lookup[domain.lower().replace('www.', '')] = brand
        return lookup
    
    def _get_brand_from_url(self, url: str) -> Optional[str]:
        """
        Check if URL belongs to a known brand.
        Returns brand name if URL is a known brand's domain, None otherwise.
        
        Examples:
            x.com → "Twitter"
            instagram.com → "Instagram" 
            google.com → "Google"
            evil-site.com → None
        """
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
            
            if domain in self._domain_to_brand:
                return self._domain_to_brand[domain]
            
            parts = domain.split('.')
            for i in range(len(parts)):
                parent = '.'.join(parts[i:])
                if parent in self._domain_to_brand:
                    return self._domain_to_brand[parent]
            
            return None
        except:
            return None
    
    def _init_clip(self):
        self.clip_model = None
        self.clip_processor = None
        self.device = "cpu"
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device).eval()
            logger.info("✓ CLIP initialized")
        except Exception as e:
            logger.warning("✗ CLIP failed: %s", e)
    
    def _init_ocr(self):
        self.ocr_reader = None
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("✓ EasyOCR initialized")
        except Exception as e:
            logger.warning("✗ OCR failed: %s", e)
    
    def _precompute_clip_embeddings(self):
        if not self.clip_model:
            self.text_embeddings = None
            self.clip_brands = []
            return
        
        try:
            self.clip_brands = self.KNOWN_BRANDS.copy()
            prompts = [f"{brand} logo" for brand in self.clip_brands]
            
            inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                self.text_embeddings = self.clip_model.get_text_features(**inputs)
                self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)
            
            logger.info("Pre-computed CLIP embeddings for %d brands", len(self.clip_brands))
        except Exception as e:
            logger.warning("CLIP embeddings failed: %s", e)
            self.text_embeddings = None
            self.clip_brands = []
    
    def capture_screenshot(self, url: str, timeout: int = 30) -> Tuple[Optional[Image.Image], str]:
        """
        Capture screenshot using subprocess isolation.
        Returns (screenshot_image, final_url_after_redirects).
        """
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        original_url = url
        final_url = url
        
        output_path = os.path.join(self.temp_dir, f"screenshot_{uuid.uuid4().hex}.png")
        url_file = output_path + '.url'
        
        logger.info("Capturing screenshot via subprocess...")
        logger.info("  Original URL: %s", url)
        
        try:
            result = subprocess.run(
                [
                    self.python_exe,
                    self.worker_script,
                    url,
                    output_path,
                    str(self.screenshot_width),
                    str(self.screenshot_height),
                    str(timeout * 1000)
                ],
                capture_output=True,
                timeout=timeout + 15,
                text=True
            )
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('FINAL_URL:'):
                        final_url = line.replace('FINAL_URL:', '').strip()
                        break
            
            if os.path.exists(url_file):
                try:
                    with open(url_file, 'r', encoding='utf-8') as f:
                        file_url = f.read().strip()
                        if file_url:
                            final_url = file_url
                except:
                    pass
                finally:
                    try:
                        os.remove(url_file)
                    except:
                        pass
            
            if final_url != original_url:
                logger.info("  REDIRECT DETECTED!")
                logger.info("  Final URL: %s", final_url)
            
            if result.returncode == 0 and os.path.exists(output_path):
                image = Image.open(output_path).copy()
                try:
                    os.remove(output_path)
                except:
                    pass
                logger.info("Screenshot captured successfully via subprocess")
                return image, final_url
            else:
                if result.stderr:
                    logger.warning("Screenshot worker stderr: %s", result.stderr[:200])
                logger.warning("Screenshot subprocess failed (exit code: %d)", result.returncode)
                return None, final_url
                
        except subprocess.TimeoutExpired:
            logger.warning("Screenshot subprocess timed out")
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                if os.path.exists(url_file):
                    os.remove(url_file)
            except:
                pass
            return None, final_url
        except Exception as e:
            logger.warning("Screenshot subprocess error: %s", e)
            return None, final_url
    
    def extract_text_ocr(self, image: Image.Image) -> str:
        """Extract ALL text from image using OCR."""
        if not self.ocr_reader:
            return ""
        try:
            results = self.ocr_reader.readtext(np.array(image))
            text = ' '.join([t for _, t, c in results if c > 0.3])
            return text
        except:
            return ""
    
    def detect_login_indicators(self, text: str) -> List[str]:
        """Find login-related keywords in text."""
        if not text:
            return []
        
        found = []
        text_lower = text.lower()
        
        for indicator in self.LOGIN_INDICATORS:
            if indicator in text_lower:
                found.append(indicator)
        
        return found
    
    def detect_brand_from_ocr(self, text: str) -> Optional[Dict]:
        """Detect brand from OCR text."""
        if not text:
            return None
        
        text_lower = text.lower()
        
        for brand, keywords in self.BRAND_OCR_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    count = text_lower.count(keyword)
                    confidence = min(95, 50 + count * 15)
                    
                    brand_proper = brand.title()
                    for known in self.KNOWN_BRANDS:
                        if known.lower() == brand.lower():
                            brand_proper = known
                            break
                    
                    return {
                        'brand': brand_proper,
                        'keyword': keyword,
                        'occurrences': count,
                        'confidence': confidence,
                        'source': 'ocr'
                    }
        
        return None
    
    def detect_brand_from_clip(self, image: Image.Image) -> Optional[Dict]:
        """
        Detect brand logo using CLIP.
        
        V5.0: Increased thresholds to reduce false positives:
            - Min confidence: 40% (was 30%)
            - Min confidence gap: 8% (was 5%)
        """
        if not self.clip_model or self.text_embeddings is None:
            return None
        
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarities = (image_features @ self.text_embeddings.T).squeeze(0)
                probs = torch.softmax(similarities * 100, dim=-1)
                
                top_idx = probs.argmax().item()
                top_conf = probs[top_idx].item() * 100
                
                sorted_indices = probs.argsort(descending=True)
                second_idx = sorted_indices[1].item()
                second_conf = probs[second_idx].item() * 100
                
                confidence_gap = top_conf - second_conf
                
                # V5.0: Stricter thresholds (was 30% conf, 5% gap)
                if top_conf >= 40 and confidence_gap >= 8:
                    return {
                        'brand': self.clip_brands[top_idx],
                        'confidence': round(top_conf, 2),
                        'second_brand': self.clip_brands[second_idx],
                        'second_confidence': round(second_conf, 2),
                        'confidence_gap': round(confidence_gap, 4),
                        'source': 'clip'
                    }
            
            return None
            
        except:
            return None
    
    def check_brand_url_mismatch(self, brand: str, url: str) -> Dict:
        """Check if detected brand matches URL domain."""
        result = {
            'brand': brand,
            'url': url,
            'is_mismatch': False,
            'actual_domain': '',
            'expected_domains': []
        }
        
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            parsed = urlparse(url)
            actual = parsed.netloc.lower().replace('www.', '')
            result['actual_domain'] = actual
            
            expected = []
            for k, v in self.brand_domains.items():
                if k.lower() == brand.lower():
                    expected = v
                    break
            
            if not expected:
                for k, v in self.brand_domains.items():
                    if brand.lower() in k.lower() or k.lower() in brand.lower():
                        expected = v
                        break
            
            result['expected_domains'] = expected
            
            if not expected:
                return result
            
            for exp in expected:
                exp_clean = exp.lower().replace('www.', '')
                
                if actual == exp_clean:
                    return result
                if actual.endswith('.' + exp_clean):
                    return result
                if exp_clean.endswith('.' + actual):
                    return result
            
            result['is_mismatch'] = True
            
        except Exception as e:
            logger.warning("Mismatch check error: %s", e)
        
        return result
    
    def _is_www_redirect_only(self, original_url: str, final_url: str) -> bool:
        """Check if redirect is just a www addition/removal."""
        try:
            orig_parsed = urlparse(original_url if original_url.startswith('http') else 'https://' + original_url)
            final_parsed = urlparse(final_url if final_url.startswith('http') else 'https://' + final_url)
            
            orig_domain = orig_parsed.netloc.lower().replace('www.', '')
            final_domain = final_parsed.netloc.lower().replace('www.', '')
            
            return orig_domain == final_domain
        except:
            return False
    
    def analyze_url(self, url: str, return_details: bool = True) -> Dict:
        """
        Analyze URL for phishing.
        
        V5.0 Logic (Known Brand URL Protection + Redirect-Aware):
        
        1. Capture screenshot and get FINAL URL after all redirects
        2. Check if URL belongs to a known brand (url_brand)
        3. If url_brand exists:
            - OCR detecting same brand → LEGITIMATE (expected)
            - OCR detecting different brand → LEGITIMATE (OCR confusion, not impersonation)
            - CLIP detecting different brand → LEGITIMATE (CLIP false positive on real brand site)
        4. If url_brand is None (unknown domain):
            - OCR brand + mismatch → PHISHING (high confidence)
            - CLIP brand + mismatch + login indicators → PHISHING (medium confidence)
            - CLIP brand + mismatch + NO login → LEGITIMATE (probable false positive)
        5. No brand detected → LEGITIMATE
        """
        result = {
            'url': url,
            'final_url': url,
            'redirect_detected': False,
            'is_phishing': False,
            'confidence': 0.0,
            'risk_score': 0.0,
            'label': 'legitimate',
            'screenshot_captured': False,
            'detected_brand': None,
            'brand_url_mismatch': False,
            'detection_method': None,
            'details': {}
        }
        
        logger.info("=" * 60)
        logger.info("Analyzing: %s", url)
        
        # Capture screenshot via subprocess (returns final URL after redirects)
        screenshot, final_url = self.capture_screenshot(url)
        
        # Update result with redirect info
        result['final_url'] = final_url
        if final_url != url and not self._is_www_redirect_only(url, final_url):
            result['redirect_detected'] = True
            logger.info("REDIRECT: %s → %s", url, final_url)
        
        # Use FINAL URL for all brand-domain comparisons
        analysis_url = final_url
        
        # =====================================================================
        # V5.0 KEY CHECK: Does the URL itself belong to a known brand?
        # If yes, this is a KNOWN BRAND'S WEBSITE - it won't impersonate another brand.
        # Any CLIP/OCR detection of a DIFFERENT brand is a false positive.
        # =====================================================================
        url_brand = self._get_brand_from_url(analysis_url)
        if url_brand:
            logger.info("✓ URL belongs to known brand: %s", url_brand)
            logger.info("  Known brand site → Cross-brand CLIP/OCR false positives will be suppressed")
        
        if not screenshot:
            result['error'] = "Screenshot failed"
            logger.warning("Screenshot capture failed")
            return result
        
        result['screenshot_captured'] = True
        result['screenshot_size'] = screenshot.size
        
        # Step 1: OCR text extraction
        logger.info("Running OCR...")
        ocr_text = self.extract_text_ocr(screenshot)
        logger.info("  OCR extracted %d characters", len(ocr_text))
        
        # Step 2: Detect login indicators
        login_indicators = self.detect_login_indicators(ocr_text)
        has_login_indicators = len(login_indicators) >= 2
        
        if login_indicators:
            logger.info("  Login indicators found: %s", login_indicators[:5])
        
        # Step 3: Detect brand from OCR
        ocr_brand = self.detect_brand_from_ocr(ocr_text)
        if ocr_brand:
            logger.info("  OCR brand detected: %s (%.0f%%)", ocr_brand['brand'], ocr_brand['confidence'])
        
        # Step 4: CLIP detection
        logger.info("Running CLIP...")
        clip_brand = self.detect_brand_from_clip(screenshot)
        if clip_brand:
            logger.info("  CLIP brand detected: %s (%.0f%%)", clip_brand['brand'], clip_brand['confidence'])
        
        # Step 5: Determine verdict
        risk_score = 0
        indicators = []
        detected_brand = None
        method = None
        
        # =====================================================================
        # V5.0: KNOWN BRAND URL PROTECTION
        # If the URL belongs to a known brand, we trust it and suppress
        # cross-brand false positives from CLIP/OCR.
        # =====================================================================
        
        if url_brand:
            # URL belongs to a known brand - this is a real brand's website
            
            if ocr_brand:
                detected_brand = ocr_brand['brand']
                
                if ocr_brand['brand'].lower() == url_brand.lower():
                    # OCR found same brand as URL → Perfect match, definitely legitimate
                    method = 'ocr_matches_url_brand'
                    logger.info("🟢 LEGITIMATE: OCR brand '%s' matches URL brand '%s'", 
                              ocr_brand['brand'], url_brand)
                else:
                    # OCR found DIFFERENT brand than URL brand
                    # This happens when: x.com shows "Google" sign-in button, etc.
                    # A known brand site showing another brand's name is NOT impersonation
                    method = 'ocr_cross_brand_on_known_site'
                    logger.info("🟢 LEGITIMATE: OCR found '%s' on %s's site (%s) - cross-brand, not impersonation",
                              ocr_brand['brand'], url_brand, analysis_url)
            
            elif clip_brand:
                detected_brand = clip_brand['brand']
                
                if clip_brand['brand'].lower() == url_brand.lower():
                    # CLIP found same brand → Correct detection
                    method = 'clip_matches_url_brand'
                    logger.info("🟢 LEGITIMATE: CLIP brand '%s' matches URL brand '%s'",
                              clip_brand['brand'], url_brand)
                else:
                    # CLIP found DIFFERENT brand on a KNOWN brand's site
                    # This is a CLIP false positive (e.g., x.com flagged as "Google")
                    method = 'clip_false_positive_on_known_site'
                    logger.info("🟢 LEGITIMATE: CLIP detected '%s' on %s's site - CLIP false positive (suppressed)",
                              clip_brand['brand'], url_brand)
            else:
                # No brand detected on known brand site → Fine
                logger.info("🟢 LEGITIMATE: Known brand site (%s), no conflicting brand detected", url_brand)
            
            # IMPORTANT: risk_score stays 0 for known brand URLs
            # The URL IS the brand's legitimate domain
        
        else:
            # URL does NOT belong to any known brand → Apply standard detection logic
            
            # Case 1: OCR found brand text on unknown domain
            if ocr_brand:
                detected_brand = ocr_brand['brand']
                method = 'ocr'
                
                mismatch = self.check_brand_url_mismatch(detected_brand, analysis_url)
                result['brand_url_mismatch'] = mismatch['is_mismatch']
                
                if mismatch['is_mismatch']:
                    indicators.append(f"OCR: '{detected_brand}' text on {mismatch['actual_domain']}")
                    
                    base_risk = 70  # High confidence - text doesn't lie
                    ocr_bonus = min(20, ocr_brand['confidence'] * 0.2)
                    
                    if clip_brand and clip_brand['brand'].lower() == detected_brand.lower():
                        method = 'ocr+clip'
                        ocr_bonus += 5
                    
                    risk_score = base_risk + ocr_bonus
                    
                    logger.warning("🔴 PHISHING (OCR): '%s' branding on %s", detected_brand, mismatch['actual_domain'])
                else:
                    logger.info("🟢 LEGITIMATE: %s on correct domain", detected_brand)
            
            # Case 2: Only CLIP found brand on unknown domain
            elif clip_brand:
                detected_brand = clip_brand['brand']
                
                mismatch = self.check_brand_url_mismatch(detected_brand, analysis_url)
                result['brand_url_mismatch'] = mismatch['is_mismatch']
                
                if mismatch['is_mismatch']:
                    if has_login_indicators:
                        method = 'clip+login'
                        indicators.append(f"CLIP: '{detected_brand}' logo on {mismatch['actual_domain']}")
                        indicators.append(f"Login indicators: {login_indicators[:3]}")
                        
                        base_risk = 55
                        clip_bonus = min(20, clip_brand['confidence'] * 0.3)
                        login_bonus = min(15, len(login_indicators) * 3)
                        
                        risk_score = base_risk + clip_bonus + login_bonus
                        
                        logger.warning("🔴 PHISHING (CLIP+Login): '%s' logo + login page on %s", 
                                     detected_brand, mismatch['actual_domain'])
                    else:
                        method = 'clip_only_no_login'
                        logger.info("⚪ CLIP detected '%s' but NO login indicators - assuming LEGITIMATE", detected_brand)
                else:
                    method = 'clip'
                    logger.info("🟢 LEGITIMATE: %s on correct domain", detected_brand)
            
            # Case 3: No brand detected on unknown domain
            else:
                if has_login_indicators:
                    logger.info("⚪ Login page detected but no brand - assuming LEGITIMATE")
                else:
                    logger.info("🟢 No brand detected - LEGITIMATE")
        
        # Set final result
        result['detected_brand'] = detected_brand
        result['detection_method'] = method
        result['risk_score'] = round(min(100, risk_score), 2)
        result['confidence'] = round(max(result['risk_score'], 100 - result['risk_score']), 2)
        
        if result['risk_score'] >= 50:
            result['is_phishing'] = True
            result['label'] = 'phishing'
        
        if return_details:
            result['details'] = {
                'original_url': url,
                'final_url': final_url,
                'redirect_detected': result['redirect_detected'],
                'url_brand': url_brand,  # V5.0: Include the URL's own brand
                'ocr_brand': ocr_brand,
                'clip_brand': clip_brand,
                'login_indicators': login_indicators,
                'has_login_indicators': has_login_indicators,
                'ocr_text_sample': ocr_text[:300] if ocr_text else "",
                'phishing_indicators': indicators,
                'known_brand_url_protection': url_brand is not None,  # V5.0: Flag
                'version': 'v5.0-known-brand-protection'
            }
        
        logger.info("Result: %s (risk: %.0f%%)", result['label'].upper(), result['risk_score'])
        if url_brand:
            logger.info("  [Known Brand URL: %s → auto-legitimate]", url_brand)
        logger.info("=" * 60)
        
        return result
    
    def get_risk_level(self, score: float) -> str:
        if score < 20: return "LOW"
        elif score < 50: return "MEDIUM"
        elif score < 80: return "HIGH"
        return "CRITICAL"
    
    def get_supported_brands(self) -> List[str]:
        """Return list of supported brands."""
        return self.KNOWN_BRANDS.copy()
    
    def close(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Visual Analyzer V5.0 closed")
        except:
            pass
    
    def __del__(self):
        self.close()


def main():
    print("=" * 70)
    print("PhishLens Visual Analyzer V5.0 - Known Brand URL Protection")
    print("=" * 70)
    print()
    print("NEW IN V5.0:")
    print("  - Known Brand URL check: x.com/instagram.com/etc. auto-legitimate")
    print("  - CLIP detecting 'Google' on x.com → suppressed as false positive")
    print("  - Stricter CLIP thresholds (40% conf, 8% gap)")
    print("  - Only unknown domains can trigger brand impersonation")
    print()
    print("PREVIOUS FIXES (still active):")
    print("  - V4.9: Redirect-aware, captures FINAL URL after redirects")
    print("  - V4.8: Subprocess isolation for thread-safety")
    print()
    
    analyzer = VisualAnalyzer()
    
    test_urls = [
        ("https://www.google.com", "Google - should be SAFE"),
        ("https://www.paypal.com", "PayPal - should be SAFE"),
        ("https://x.com", "X/Twitter - should be SAFE (V5.0 fix)"),
        ("https://instagram.com", "Instagram - should be SAFE (V5.0 fix)"),
    ]
    
    print("-" * 70)
    
    for url, desc in test_urls:
        print(f"\nTest: {desc}")
        result = analyzer.analyze_url(url)
        
        status = "🔴 PHISHING" if result['is_phishing'] else "🟢 SAFE"
        print(f"  Result: {status}")
        print(f"  Screenshot: {'✓' if result.get('screenshot_captured') else '✗'}")
        print(f"  Redirect: {'Yes → ' + result.get('final_url', url) if result.get('redirect_detected') else 'No'}")
        print(f"  URL Brand: {result.get('details', {}).get('url_brand', 'None')}")
        print(f"  Detected Brand: {result.get('detected_brand', 'None')}")
        print(f"  Method: {result.get('detection_method', 'None')}")
        print(f"  Risk: {result.get('risk_score')}%")
        print(f"  Known Brand Protection: {result.get('details', {}).get('known_brand_url_protection', False)}")
    
    analyzer.close()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
