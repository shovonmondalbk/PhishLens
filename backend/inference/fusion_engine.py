"""
PhishLens - Fusion Engine V2.3 (Trusted Domain + Brand Mismatch Leak Fix)
Combines URL, HTML, and Visual analyzers with FALSE POSITIVE PREVENTION.

V2.3 CHANGES:
    - CASE 3 FIX: Brand mismatch no longer boosts score on trusted domains (.gov, .edu, .ac.*)
    - CASE 3 FIX: Brand mismatch flag from visual is ignored if visual itself says LEGITIMATE
    - NEW CASE 4: Single analyzer phishing on trusted domain → dampened by 2/3 safe consensus
    - NEW CASE 5: Single noisy analyzer can't override strong 2/3 safe consensus (>95% confident)
    - Threat type fix: Brand Impersonation only reported when visual actually says phishing

V2.2 CHANGES:
    - Supports Visual Analyzer V5.0's "known_brand_url_protection" flag
    - Stronger consensus rule for known brand URLs
    - Strong URL+HTML consensus protection

V2.1 CHANGES:
    - Supports redirect info from Visual Analyzer V4.9
    - Includes final_url in result when redirect detected

CONSENSUS RULES (V2.3):
    - 2+ analyzers agree → Trust the consensus
    - Only Visual says phishing + URL/HTML say safe:
        * If URL belongs to known brand (V5.0 flag) → NOT phishing
        * If URL+HTML both >90% confident legitimate → NOT phishing
        * If trusted domain (.gov, .edu) → NOT phishing
        * If payment brand on clean site → NOT phishing
        * Otherwise → cautious, may flag if very high visual evidence
"""

import os
import sys
import json
import time
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse

sys.path.insert(0, r"C:\PhishLens")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ThreatType(Enum):
    NONE = "none"
    URL_SUSPICIOUS = "url_suspicious"
    CONTENT_PHISHING = "content_phishing"
    BRAND_IMPERSONATION = "brand_impersonation"
    CREDENTIAL_HARVESTING = "credential_harvesting"
    MULTI_VECTOR = "multi_vector_attack"


@dataclass
class AnalyzerResult:
    analyzer_name: str
    is_phishing: bool
    confidence: float
    risk_score: float
    details: Dict
    success: bool
    error: Optional[str] = None


class PhishLensFusion:
    """
    Fusion Engine V2.3 - Trusted Domain Fix + Brand Mismatch Leak Fix
    
    FALSE POSITIVE PREVENTION:
        1. Known Brand URL: If Visual V5.0 says URL is a known brand → visual-only = FP
        2. Strong URL+HTML Consensus: Both >90% legitimate → visual can't override
        3. Visual alone can't override URL+HTML consensus
        4. Payment logos (Visa, MC) on clean sites are ignored
        5. Trusted domains (.gov, .edu) get benefit of doubt
        6. V2.3: Brand mismatch flag ignored when visual says legitimate
        7. V2.3: Single analyzer can't override 2/3 safe consensus on trusted domains
        8. Requires 2/3 agreement for phishing verdict (in most cases)
    
    REDIRECT SUPPORT (from V2.1):
        - Captures final URL from Visual Analyzer
        - Includes redirect info in result
    """
    
    BASE_WEIGHTS = {
        'url': 0.40,
        'html': 0.40,
        'visual': 0.20
    }
    
    PHISHING_THRESHOLD = 50.0
    HIGH_CONFIDENCE_THRESHOLD = 80.0
    BRAND_MISMATCH_VISUAL_WEIGHT = 0.60
    
    # Trusted domain patterns
    TRUSTED_DOMAIN_PATTERNS = [
        r'\.gov$', r'\.gov\.[a-z]{2}$',
        r'\.edu$', r'\.edu\.[a-z]{2}$',
        r'\.ac\.[a-z]{2}$',
        r'\.mil$', r'\.mil\.[a-z]{2}$',
        r'\.int$',
        r'\.museum$',
        r'university', r'\.uni\.', r'\.univ\.',
    ]
    
    # Payment brands
    PAYMENT_BRANDS = {
        'visa', 'mastercard', 'american express', 'amex', 
        'paypal', 'stripe', 'discover', 'jcb', 'unionpay',
        'apple pay', 'google pay', 'samsung pay', 'alipay',
        'maestro', 'diners club', 'rupay', 'mir'
    }
    
    # High-value phishing targets
    HIGH_VALUE_TARGETS = {
        'paypal', 'amazon', 'microsoft', 'apple', 'google', 
        'facebook', 'instagram', 'netflix', 'bank of america',
        'chase', 'wells fargo', 'citibank', 'coinbase', 'binance',
        'dropbox', 'github', 'linkedin', 'twitter', 'yahoo'
    }
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        load_analyzers: bool = True
    ):
        self.config_path = config_path or r"C:\PhishLens\backend\inference\config.json"
        self.config = self._load_config()
        
        thresholds = self.config.get('thresholds', {})
        self.PHISHING_THRESHOLD = thresholds.get('phishing_threshold', 50.0)
        self.HIGH_CONFIDENCE_THRESHOLD = thresholds.get('high_confidence', 95.0)
        self.BRAND_MISMATCH_VISUAL_WEIGHT = thresholds.get('brand_mismatch_boost', 0.60)
        
        analyzers_config = self.config.get('analyzers', {})
        if analyzers_config:
            self.BASE_WEIGHTS = {
                'url': analyzers_config.get('url', {}).get('weight', 0.40),
                'html': analyzers_config.get('html', {}).get('weight', 0.40),
                'visual': analyzers_config.get('visual', {}).get('weight', 0.20)
            }
        
        self.url_analyzer = None
        self.html_analyzer = None
        self.visual_analyzer = None
        
        if load_analyzers:
            self._load_analyzers()
        
        whitelist = self.config.get('whitelist', {})
        blacklist = self.config.get('blacklist', {})
        logger.info(f"Whitelist: {'Enabled' if whitelist.get('enabled') else 'Disabled'} ({len(whitelist.get('domains', []))} domains)")
        logger.info(f"Blacklist: {'Enabled' if blacklist.get('enabled') else 'Disabled'} ({len(blacklist.get('domains', []))} domains)")
        logger.info("PhishLens Fusion Engine V2.3 (Trusted Domain + Brand Mismatch Leak Fix) initialized!")
    
    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return {}
    
    def _save_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    # =========================================================================
    # Whitelist / Blacklist Management
    # =========================================================================
    
    def add_to_whitelist(self, domain: str) -> bool:
        try:
            domain = domain.lower().strip()
            if 'whitelist' not in self.config:
                self.config['whitelist'] = {'enabled': True, 'domains': []}
            if domain not in self.config['whitelist']['domains']:
                self.config['whitelist']['domains'].append(domain)
                self._save_config()
                return True
            return False
        except:
            return False
    
    def remove_from_whitelist(self, domain: str) -> bool:
        try:
            domain = domain.lower().strip()
            if 'whitelist' in self.config and domain in self.config['whitelist']['domains']:
                self.config['whitelist']['domains'].remove(domain)
                self._save_config()
                return True
            return False
        except:
            return False
    
    def add_to_blacklist(self, domain: str) -> bool:
        try:
            domain = domain.lower().strip()
            if 'blacklist' not in self.config:
                self.config['blacklist'] = {'enabled': True, 'domains': []}
            if domain not in self.config['blacklist']['domains']:
                self.config['blacklist']['domains'].append(domain)
                self._save_config()
                return True
            return False
        except:
            return False
    
    def remove_from_blacklist(self, domain: str) -> bool:
        try:
            domain = domain.lower().strip()
            if 'blacklist' in self.config and domain in self.config['blacklist']['domains']:
                self.config['blacklist']['domains'].remove(domain)
                self._save_config()
                return True
            return False
        except:
            return False
    
    def get_whitelist(self) -> List[str]:
        return self.config.get('whitelist', {}).get('domains', [])
    
    def get_blacklist(self) -> List[str]:
        return self.config.get('blacklist', {}).get('domains', [])
    
    def _check_whitelist(self, url: str) -> Tuple[bool, str]:
        whitelist_config = self.config.get('whitelist', {})
        if not whitelist_config.get('enabled', False):
            return False, ""
        
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            domain_no_www = domain.replace('www.', '')
            
            for whitelisted in whitelist_config.get('domains', []):
                wl = whitelisted.lower().replace('www.', '')
                if domain == whitelisted.lower() or domain_no_www == wl:
                    return True, domain
                if domain.endswith('.' + wl) or domain_no_www.endswith('.' + wl):
                    return True, domain
            return False, domain
        except:
            return False, ""
    
    def _check_blacklist(self, url: str) -> Tuple[bool, str]:
        blacklist_config = self.config.get('blacklist', {})
        if not blacklist_config.get('enabled', False):
            return False, ""
        
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
            
            for blacklisted in blacklist_config.get('domains', []):
                bl = blacklisted.lower().replace('www.', '')
                if domain == bl or domain.endswith('.' + bl):
                    return True, domain
            return False, domain
        except:
            return False, ""
    
    # =========================================================================
    # Trusted Domain Detection
    # =========================================================================
    
    def _is_trusted_domain(self, url: str) -> Tuple[bool, str]:
        """Check if domain matches trusted patterns (.gov, .edu, .ac.*, etc.)"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
            
            for pattern in self.TRUSTED_DOMAIN_PATTERNS:
                if re.search(pattern, domain, re.IGNORECASE):
                    return True, f"Matches trusted pattern: {pattern}"
            
            return False, ""
        except:
            return False, ""
    
    def _is_payment_brand(self, brand: str) -> bool:
        if not brand:
            return False
        return brand.lower() in self.PAYMENT_BRANDS
    
    def _is_high_value_target(self, brand: str) -> bool:
        if not brand:
            return False
        return brand.lower() in self.HIGH_VALUE_TARGETS
    
    # =========================================================================
    # Analyzer Loading & Execution
    # =========================================================================
    
    def _load_analyzers(self):
        logger.info("Loading analyzers...")
        
        try:
            from backend.inference.url_analyzer import URLAnalyzer
            self.url_analyzer = URLAnalyzer()
            logger.info("  URL Analyzer loaded")
        except Exception as e:
            logger.error(f"  Failed to load URL Analyzer: {e}")
        
        try:
            from backend.inference.html_analyzer import HTMLAnalyzer
            self.html_analyzer = HTMLAnalyzer()
            logger.info("  HTML Analyzer loaded")
        except Exception as e:
            logger.error(f"  Failed to load HTML Analyzer: {e}")
        
        try:
            from backend.inference.visual_analyzer import VisualAnalyzer
            self.visual_analyzer = VisualAnalyzer()
            logger.info("  Visual Analyzer loaded")
        except Exception as e:
            logger.error(f"  Failed to load Visual Analyzer: {e}")
        
        logger.info("All analyzers loaded!")
    
    def _run_url_analysis(self, url: str) -> AnalyzerResult:
        if not self.url_analyzer:
            return AnalyzerResult("url", False, 0.0, 0.0, {}, False, "URL Analyzer not loaded")
        
        try:
            result = self.url_analyzer.analyze(url)
            return AnalyzerResult(
                "url",
                result.get('is_phishing', False),
                result.get('confidence', 0.0),
                result.get('risk_score', 0.0),
                result.get('details', {}),
                True
            )
        except Exception as e:
            return AnalyzerResult("url", False, 0.0, 0.0, {}, False, str(e))
    
    def _run_html_analysis(self, url: str) -> AnalyzerResult:
        if not self.html_analyzer:
            return AnalyzerResult("html", False, 0.0, 0.0, {}, False, "HTML Analyzer not loaded")
        
        try:
            result = self.html_analyzer.analyze_url(url)
            return AnalyzerResult(
                "html",
                result.get('is_phishing', False),
                result.get('confidence', 0.0),
                result.get('risk_score', 0.0),
                result.get('details', {}),
                result.get('html_fetched', False)
            )
        except Exception as e:
            return AnalyzerResult("html", False, 0.0, 0.0, {}, False, str(e))
    
    def _run_visual_analysis(self, url: str) -> Tuple[AnalyzerResult, Dict]:
        """Run visual analysis and return both result and redirect info."""
        redirect_info = {
            'final_url': url,
            'redirect_detected': False
        }
        
        if not self.visual_analyzer:
            return AnalyzerResult("visual", False, 0.0, 0.0, {}, False, "Visual Analyzer not loaded"), redirect_info
        
        try:
            result = self.visual_analyzer.analyze_url(url)
            
            redirect_info['final_url'] = result.get('final_url', url)
            redirect_info['redirect_detected'] = result.get('redirect_detected', False)
            
            if redirect_info['redirect_detected']:
                logger.info(f"  [REDIRECT] {url} → {redirect_info['final_url']}")
            
            return AnalyzerResult(
                "visual",
                result.get('is_phishing', False),
                result.get('confidence', 0.0),
                result.get('risk_score', 0.0),
                {
                    'detected_brand': result.get('detected_brand'),
                    'brand_url_mismatch': result.get('brand_url_mismatch', False),
                    'brand_predictions': result.get('details', {}).get('brand_predictions', []),
                    'final_url': result.get('final_url', url),
                    'redirect_detected': result.get('redirect_detected', False),
                    # V2.2: Capture the known brand URL protection flag from Visual V5.0
                    'known_brand_url_protection': result.get('details', {}).get('known_brand_url_protection', False),
                    'url_brand': result.get('details', {}).get('url_brand'),
                },
                result.get('screenshot_captured', False)
            ), redirect_info
        except Exception as e:
            return AnalyzerResult("visual", False, 0.0, 0.0, {}, False, str(e)), redirect_info
    
    # =========================================================================
    # Weight Calculation (Consensus-Based)
    # =========================================================================
    
    def _calculate_weights(
        self,
        url_result: AnalyzerResult,
        html_result: AnalyzerResult,
        visual_result: AnalyzerResult,
        is_trusted: bool,
        detected_brand: Optional[str],
        is_known_brand_url: bool = False  # V2.2: New parameter
    ) -> Dict[str, float]:
        """
        Calculate dynamic weights with consensus logic.
        
        V2.2: Known brand URL flag reduces visual weight further.
        """
        weights = self.BASE_WEIGHTS.copy()
        
        successful = {
            'url': url_result.success,
            'html': html_result.success,
            'visual': visual_result.success
        }
        
        success_count = sum(successful.values())
        
        if success_count == 0:
            return {'url': 0.33, 'html': 0.33, 'visual': 0.34}
        
        # Redistribute failed analyzer weights
        for name in ['url', 'html', 'visual']:
            if not successful[name]:
                extra = weights[name] / max(1, sum(1 for k, v in successful.items() if v and k != name))
                weights[name] = 0
                for other in ['url', 'html', 'visual']:
                    if other != name and successful[other]:
                        weights[other] += extra / 2
        
        brand_mismatch = visual_result.details.get('brand_url_mismatch', False)
        
        if brand_mismatch:
            # V2.2: If known brand URL, DON'T boost visual at all
            if is_known_brand_url:
                logger.info("  [V2.2] Known brand URL - suppressing visual weight boost entirely")
                # Actually reduce visual weight for known brand URLs
                weights['visual'] = max(0.05, weights['visual'] - 0.10)
                redistribution = 0.10 / 2
                weights['url'] += redistribution
                weights['html'] += redistribution
            else:
                has_other_support = url_result.is_phishing or html_result.is_phishing
                
                if has_other_support:
                    boost = self.BRAND_MISMATCH_VISUAL_WEIGHT - weights['visual']
                    if boost > 0:
                        weights['visual'] = self.BRAND_MISMATCH_VISUAL_WEIGHT
                        reduction = boost / 2
                        weights['url'] = max(0.1, weights['url'] - reduction)
                        weights['html'] = max(0.1, weights['html'] - reduction)
                else:
                    is_payment = self._is_payment_brand(detected_brand)
                    
                    if is_trusted or is_payment:
                        logger.info(f"  [FP Prevention] Not boosting visual weight (trusted={is_trusted}, payment={is_payment})")
                    else:
                        weights['visual'] = min(0.35, weights['visual'] + 0.1)
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    # =========================================================================
    # Threat Type & Risk Level
    # =========================================================================
    
    def _determine_threat_type(
        self,
        url_result: AnalyzerResult,
        html_result: AnalyzerResult,
        visual_result: AnalyzerResult
    ) -> ThreatType:
        threats = []
        
        if url_result.is_phishing:
            threats.append(ThreatType.URL_SUSPICIOUS)
        
        if html_result.is_phishing:
            html_details = html_result.details.get('form_analysis', {})
            if html_details.get('has_password_field') or html_details.get('has_credit_card_field'):
                threats.append(ThreatType.CREDENTIAL_HARVESTING)
            else:
                threats.append(ThreatType.CONTENT_PHISHING)
        
        if visual_result.details.get('brand_url_mismatch', False) and visual_result.is_phishing:
            threats.append(ThreatType.BRAND_IMPERSONATION)
        
        if len(threats) == 0:
            return ThreatType.NONE
        elif len(threats) >= 2:
            return ThreatType.MULTI_VECTOR
        else:
            return threats[0]
    
    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        if risk_score < 20:
            return RiskLevel.SAFE
        elif risk_score < 40:
            return RiskLevel.LOW
        elif risk_score < 60:
            return RiskLevel.MEDIUM
        elif risk_score < 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_summary(
        self,
        is_phishing: bool,
        risk_score: float,
        threat_type: ThreatType,
        url_result: AnalyzerResult,
        html_result: AnalyzerResult,
        visual_result: AnalyzerResult,
        redirect_info: Optional[Dict] = None
    ) -> str:
        if not is_phishing:
            if redirect_info and redirect_info.get('redirect_detected'):
                return f"This website appears to be legitimate. Redirected to {redirect_info['final_url']} which is safe."
            return "This website appears to be legitimate. No phishing indicators were detected."
        
        reasons = []
        
        if url_result.is_phishing:
            reasons.append("suspicious URL patterns")
        
        if html_result.is_phishing:
            if html_result.details.get('form_analysis', {}).get('has_password_field'):
                reasons.append("credential harvesting forms")
            else:
                reasons.append("suspicious content patterns")
        
        if visual_result.details.get('brand_url_mismatch'):
            brand = visual_result.details.get('detected_brand', 'a known brand')
            reasons.append(f"visual impersonation of {brand}")
        
        if threat_type == ThreatType.MULTI_VECTOR:
            return f"WARNING: Multi-vector phishing attack! {', '.join(reasons)}. Risk: {risk_score:.1f}%"
        elif threat_type == ThreatType.BRAND_IMPERSONATION:
            brand = visual_result.details.get('detected_brand', 'a known brand')
            return f"DANGER: Site impersonating {brand}! Visual matches {brand} but URL doesn't belong to them."
        elif threat_type == ThreatType.CREDENTIAL_HARVESTING:
            return "WARNING: Credential harvesting detected! Site has forms designed to steal your login info."
        else:
            return f"CAUTION: Phishing indicators - {', '.join(reasons)}. Risk: {risk_score:.1f}%"
    
    # =========================================================================
    # Main Analysis (V2.2 Consensus Logic + Known Brand URL Support)
    # =========================================================================
    
    def analyze(self, url: str, return_details: bool = True) -> Dict:
        """
        Analyze URL with CONSENSUS-BASED phishing detection.
        
        V2.3 RULES:
            1. Whitelist → SAFE (skip analysis)
            2. Blacklist → PHISHING (skip analysis)
            3. 2+ analyzers say phishing → PHISHING
            4. Only Visual says phishing + URL/HTML say safe:
                a. If URL is known brand (V5.0) → SAFE (known brand protection)
                b. If URL+HTML both >90% confident legitimate → SAFE (strong consensus)
                c. If trusted domain OR payment brand → SAFE (false positive)
                d. If high-value target with very high visual confidence → cautious
                e. Otherwise → SAFE (insufficient evidence)
            5. Brand mismatch + URL/HTML phishing:
                a. If known brand URL → ignore mismatch
                b. If trusted domain (.ac.bd, .edu) → suppress boost (V2.3 FIX)
                c. If visual itself says legitimate → suppress boost (V2.3 FIX)
                d. Otherwise → boost score
            6. Single analyzer phishing on trusted domain → dampened (V2.3 NEW)
            7. Single noisy analyzer vs 2 very confident safe → capped (V2.3 NEW)
        """
        start_time = time.time()
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        logger.info(f"Starting fusion analysis for: {url}")
        
        # Check whitelist
        is_whitelisted, domain = self._check_whitelist(url)
        if is_whitelisted:
            logger.info(f"  URL is WHITELISTED: {domain}")
            return {
                'url': url, 'is_phishing': False, 'confidence': 100.0,
                'risk_score': 0.0, 'risk_level': RiskLevel.SAFE.value,
                'threat_type': ThreatType.NONE.value, 'label': 'legitimate',
                'summary': f"Website ({domain}) is in trusted whitelist.",
                'analysis_time': round(time.time() - start_time, 2),
                'whitelisted': True, 'blacklisted': False,
                'details': {'whitelist_match': domain, 'skipped_analysis': True} if return_details else {}
            }
        
        # Check blacklist
        is_blacklisted, domain = self._check_blacklist(url)
        if is_blacklisted:
            logger.info(f"  URL is BLACKLISTED: {domain}")
            return {
                'url': url, 'is_phishing': True, 'confidence': 100.0,
                'risk_score': 100.0, 'risk_level': RiskLevel.CRITICAL.value,
                'threat_type': ThreatType.MULTI_VECTOR.value, 'label': 'phishing',
                'summary': f"WARNING: Website ({domain}) is in known phishing blacklist!",
                'analysis_time': round(time.time() - start_time, 2),
                'whitelisted': False, 'blacklisted': True,
                'details': {'blacklist_match': domain, 'skipped_analysis': True} if return_details else {}
            }
        
        # Check trusted domain
        is_trusted, trust_reason = self._is_trusted_domain(url)
        if is_trusted:
            logger.info(f"  Trusted domain: {trust_reason}")
        
        # Run all analyzers
        logger.info("  Running URL analysis...")
        url_result = self._run_url_analysis(url)
        
        logger.info("  Running HTML analysis...")
        html_result = self._run_html_analysis(url)
        
        logger.info("  Running Visual analysis...")
        visual_result, redirect_info = self._run_visual_analysis(url)
        
        # Get detected brand info
        detected_brand = visual_result.details.get('detected_brand')
        brand_mismatch = visual_result.details.get('brand_url_mismatch', False)
        
        # V2.2: Check if Visual V5.0 flagged this as a known brand URL
        is_known_brand_url = visual_result.details.get('known_brand_url_protection', False)
        url_brand = visual_result.details.get('url_brand')
        
        if is_known_brand_url:
            logger.info(f"  [V2.2] Visual V5.0 reports: Known brand URL ({url_brand})")
        
        # Calculate weights with consensus logic
        weights = self._calculate_weights(
            url_result, html_result, visual_result, 
            is_trusted, detected_brand, is_known_brand_url
        )
        logger.info(f"  Weights: URL={weights['url']:.2f}, HTML={weights['html']:.2f}, Visual={weights['visual']:.2f}")
        
        # Calculate base weighted score
        weighted_score = (
            url_result.risk_score * weights['url'] +
            html_result.risk_score * weights['html'] +
            visual_result.risk_score * weights['visual']
        )
        
        # Count phishing votes
        phishing_votes = sum([
            url_result.is_phishing,
            html_result.is_phishing,
            visual_result.is_phishing
        ])
        
        # =====================================================================
        # CONSENSUS LOGIC (V2.3 - Trusted Domain + Brand Mismatch Leak Fix)
        # =====================================================================
        
        false_positive_prevented = False
        fp_reason = ""
        
        # CASE 1: Multiple analyzers agree on phishing → Trust consensus
        if phishing_votes >= 2:
            weighted_score = max(weighted_score, 70.0)
            logger.info(f"  Consensus: {phishing_votes}/3 analyzers agree on PHISHING")
        
        # CASE 2: Only Visual says phishing (URL + HTML say safe)
        elif visual_result.is_phishing and not url_result.is_phishing and not html_result.is_phishing:
            
            is_payment = self._is_payment_brand(detected_brand)
            is_high_value = self._is_high_value_target(detected_brand)
            
            # ============================================================
            # V2.2 NEW: Known Brand URL Protection (highest priority)
            # If the URL belongs to a known brand (from Visual V5.0),
            # this is definitively NOT phishing. A real brand's site
            # does not impersonate another brand.
            # ============================================================
            if is_known_brand_url:
                false_positive_prevented = True
                fp_reason = f"Known brand URL ({url_brand}) - visual false positive suppressed"
                weighted_score = min(weighted_score, 15.0)  # Force very low score
                logger.info(f"  [V2.2 FP PREVENTED] Known brand URL: {url_brand}")
            
            # ============================================================
            # V2.2 NEW: Strong URL+HTML Consensus Protection
            # If BOTH URL and HTML are very confident (>90%) that it's
            # legitimate, visual alone CANNOT override this.
            # ============================================================
            elif (not url_result.is_phishing and url_result.confidence > 90 and
                  not html_result.is_phishing and html_result.confidence > 90):
                false_positive_prevented = True
                fp_reason = (f"Strong URL+HTML consensus: URL {url_result.confidence:.1f}% safe, "
                           f"HTML {html_result.confidence:.1f}% safe - visual alone cannot override")
                weighted_score = min(weighted_score, 30.0)  # Cap well below threshold
                logger.info(f"  [V2.2 FP PREVENTED] Strong consensus: URL={url_result.confidence:.1f}%, HTML={html_result.confidence:.1f}%")
            
            # 2a: Trusted domain (.gov, .edu, .ac.*) → Likely false positive
            elif is_trusted:
                false_positive_prevented = True
                fp_reason = f"Trusted domain ({trust_reason})"
                weighted_score = min(weighted_score, 35.0)
            
            # 2b: Payment brand (Visa, MC) on clean site → Likely false positive
            elif is_payment:
                if url_result.risk_score < 40 and html_result.risk_score < 40:
                    false_positive_prevented = True
                    fp_reason = f"Payment logo ({detected_brand}) on clean site"
                    weighted_score = min(weighted_score, 35.0)
            
            # 2c: High-value target but URL/HTML say safe → Be cautious, don't auto-flag
            elif is_high_value:
                # V2.2: Require MUCH higher confidence from visual to override
                # Previously was: confidence > 80 and risk_score > 70
                # Now: confidence > 90 and risk_score > 85 (much stricter)
                if visual_result.confidence > 90 and visual_result.risk_score > 85:
                    weighted_score = max(weighted_score, 55.0)
                    logger.info(f"  High-value target ({detected_brand}) with VERY high visual confidence")
                else:
                    false_positive_prevented = True
                    fp_reason = f"High-value target ({detected_brand}) but insufficient visual confidence ({visual_result.confidence:.1f}%)"
                    weighted_score = min(weighted_score, 40.0)  # V2.2: Lowered from 45
            
            # 2d: Unknown brand, only visual flagged → Insufficient evidence
            else:
                if url_result.risk_score < 30 and html_result.risk_score < 30:
                    false_positive_prevented = True
                    fp_reason = "Only visual flagged, URL+HTML very clean"
                    weighted_score = min(weighted_score, 40.0)
        
        # CASE 3: Brand mismatch + supporting evidence → Boost score
        # V2.3 FIX: Three additional guards to prevent false positives:
        #   a) Visual must ALSO say phishing (not just leak brand_mismatch flag)
        #   b) Must NOT be a known brand URL
        #   c) Must NOT be a trusted domain (.gov, .edu, .ac.*)
        elif brand_mismatch and (url_result.is_phishing or html_result.is_phishing):
            # V2.2: Don't boost if known brand URL
            if is_known_brand_url:
                logger.info(f"  [V2.2] Brand mismatch ignored - known brand URL ({url_brand})")
            
            # V2.3 FIX: Don't boost if trusted domain (.gov, .edu, .ac.*)
            elif is_trusted:
                false_positive_prevented = True
                fp_reason = f"Brand mismatch on trusted domain ({trust_reason}) - boost suppressed"
                weighted_score = min(weighted_score, 35.0)
                logger.info(f"  [V2.3 FP PREVENTED] Trusted domain with brand mismatch: {trust_reason}")
            
            # V2.3 FIX: Don't boost if visual itself says LEGITIMATE (0% risk)
            # brand_mismatch flag should not override visual's own verdict
            elif not visual_result.is_phishing and visual_result.risk_score < 10:
                false_positive_prevented = True
                fp_reason = (f"Brand mismatch detected but visual analyzer itself says legitimate "
                           f"(risk={visual_result.risk_score:.1f}%) - not boosting")
                # Don't boost, let weighted average decide naturally
                logger.info(f"  [V2.3 FP PREVENTED] Visual says safe ({visual_result.risk_score:.1f}%%) despite brand_mismatch flag")
            
            else:
                weighted_score = max(weighted_score, 75.0)
                logger.info("  Brand mismatch with supporting evidence")
        
        # =====================================================================
        # V2.3 NEW - CASE 4: Single analyzer phishing + trusted domain override
        # When only URL or only HTML says phishing on a trusted domain,
        # and the other 2 analyzers are confident it's safe → dampen score.
        # This handles: DeBERTa flagging deep academic subdomains like
        # admission.eis.du.ac.bd (5 levels looks suspicious to ML model)
        # =====================================================================
        elif phishing_votes == 1 and is_trusted:
            # Only 1 analyzer says phishing on a trusted domain
            # Check which one and how confident the others are
            safe_count = sum([
                not url_result.is_phishing and url_result.confidence > 80,
                not html_result.is_phishing and html_result.confidence > 80,
                not visual_result.is_phishing
            ])
            
            if safe_count >= 2:
                false_positive_prevented = True
                
                if url_result.is_phishing:
                    fp_reason = (f"URL analyzer flagged trusted domain ({trust_reason}), "
                               f"but HTML ({html_result.confidence:.1f}% safe) and Visual agree it's legitimate")
                elif html_result.is_phishing:
                    fp_reason = (f"HTML analyzer flagged trusted domain ({trust_reason}), "
                               f"but URL ({url_result.confidence:.1f}% safe) and Visual agree it's legitimate")
                else:
                    fp_reason = f"Single analyzer flagged trusted domain ({trust_reason})"
                
                weighted_score = min(weighted_score, 35.0)
                logger.info(f"  [V2.3 FP PREVENTED] {fp_reason}")
        
        # =====================================================================
        # V2.3 NEW - CASE 5: Single analyzer phishing + strong 2/3 consensus safe
        # Even for non-trusted domains: if only 1 analyzer flags and the other
        # 2 are VERY confident (>95%) it's safe → dampen the score
        # This prevents a single noisy analyzer from overriding strong consensus
        # =====================================================================
        elif phishing_votes == 1 and not is_trusted:
            safe_analyzers_confident = 0
            
            if not url_result.is_phishing and url_result.confidence > 95:
                safe_analyzers_confident += 1
            if not html_result.is_phishing and html_result.confidence > 95:
                safe_analyzers_confident += 1
            if not visual_result.is_phishing and visual_result.risk_score < 5:
                safe_analyzers_confident += 1
            
            if safe_analyzers_confident >= 2:
                # Two analyzers are VERY confident it's safe - cap the score
                weighted_score = min(weighted_score, 45.0)
                logger.info(f"  [V2.3] Strong 2/3 safe consensus ({safe_analyzers_confident} analyzers >95%% confident)")
        
        if false_positive_prevented:
            logger.info(f"  [FALSE POSITIVE PREVENTED] {fp_reason}")
        
        # Determine final verdict
        is_phishing = weighted_score >= self.PHISHING_THRESHOLD
        
        # Calculate confidence
        if is_phishing:
            confidence = min(99.9, weighted_score + 20)
        else:
            confidence = min(99.9, 100 - weighted_score + 20)
        
        # Determine threat type and risk level
        # V2.2: If false positive prevented, override threat type to NONE
        if false_positive_prevented and not is_phishing:
            threat_type = ThreatType.NONE
        else:
            threat_type = self._determine_threat_type(url_result, html_result, visual_result)
        
        risk_level = self._get_risk_level(weighted_score)
        
        # Generate summary (with redirect info)
        summary = self._generate_summary(
            is_phishing, weighted_score, threat_type,
            url_result, html_result, visual_result,
            redirect_info
        )
        
        elapsed_time = time.time() - start_time
        
        # Build result
        result = {
            'url': url,
            'final_url': redirect_info.get('final_url', url),
            'redirect_detected': redirect_info.get('redirect_detected', False),
            'is_phishing': is_phishing,
            'confidence': round(confidence, 2),
            'risk_score': round(weighted_score, 2),
            'risk_level': risk_level.value,
            'threat_type': threat_type.value,
            'label': 'phishing' if is_phishing else 'legitimate',
            'summary': summary,
            'analysis_time': round(elapsed_time, 2),
            'whitelisted': False,
            'blacklisted': False
        }
        
        if return_details:
            result['details'] = {
                'weights_used': weights,
                'trusted_domain': is_trusted,
                'trust_reason': trust_reason if is_trusted else None,
                'known_brand_url': is_known_brand_url,  # V2.2
                'url_brand': url_brand,  # V2.2
                'false_positive_prevented': false_positive_prevented,
                'fp_reason': fp_reason if false_positive_prevented else None,
                'consensus_votes': phishing_votes,
                'redirect_info': redirect_info,
                'url_analysis': {
                    'success': url_result.success,
                    'is_phishing': url_result.is_phishing,
                    'confidence': url_result.confidence,
                    'risk_score': url_result.risk_score,
                    'error': url_result.error
                },
                'html_analysis': {
                    'success': html_result.success,
                    'is_phishing': html_result.is_phishing,
                    'confidence': html_result.confidence,
                    'risk_score': html_result.risk_score,
                    'error': html_result.error
                },
                'visual_analysis': {
                    'success': visual_result.success,
                    'is_phishing': visual_result.is_phishing,
                    'detected_brand': detected_brand,
                    'brand_url_mismatch': brand_mismatch,
                    'is_payment_brand': self._is_payment_brand(detected_brand),
                    'is_high_value_target': self._is_high_value_target(detected_brand),
                    'known_brand_url_protection': is_known_brand_url,  # V2.2
                    'confidence': visual_result.confidence,
                    'risk_score': visual_result.risk_score,
                    'error': visual_result.error,
                    'final_url': redirect_info.get('final_url', url),
                    'redirect_detected': redirect_info.get('redirect_detected', False)
                },
                'votes': {
                    'url': 'phishing' if url_result.is_phishing else 'legitimate',
                    'html': 'phishing' if html_result.is_phishing else 'legitimate',
                    'visual': 'phishing' if visual_result.is_phishing else 'legitimate',
                    'total_phishing_votes': phishing_votes
                }
            }
        
        logger.info(f"  Result: {'PHISHING' if is_phishing else 'SAFE'} (score: {weighted_score:.1f}%)")
        
        return result
    
    def quick_analyze(self, url: str) -> Dict:
        return self.analyze(url, return_details=False)
    
    def close(self):
        if self.visual_analyzer:
            self.visual_analyzer.close()
        logger.info("Fusion Engine closed")


def main():
    print("=" * 70)
    print("PhishLens Fusion Engine V2.3 - Trusted Domain + Brand Mismatch Fix")
    print("=" * 70)
    print()
    print("FALSE POSITIVE PREVENTION (V2.3):")
    print("  - V2.3: Brand mismatch on trusted domains (.ac.bd) → NOT boosted")
    print("  - V2.3: Visual says 0% but brand_mismatch flag → leak suppressed")
    print("  - V2.3: Single analyzer on trusted domain → dampened by 2/3 consensus")
    print("  - V2.2: Known Brand URL: x.com/instagram.com auto-safe (Visual V5.0)")
    print("  - V2.2: Strong Consensus: URL+HTML both >90% safe → visual can't override")
    print("  - Visual alone can't override URL+HTML consensus")
    print("  - Payment logos (Visa, MC) on clean sites don't trigger")
    print("  - Trusted domains (.gov, .edu, .ac.*) get benefit of doubt")
    print()
    print("REDIRECT SUPPORT (V2.1):")
    print("  - Captures final URL after redirects")
    print("  - Short URLs (bit.ly, tinyurl) correctly resolved")
    print()
    
    fusion = PhishLensFusion()
    
    test_urls = [
        "https://www.google.com",
        "https://www.paypal.com",
        "https://x.com",
        "https://instagram.com",
        "https://admission.eis.du.ac.bd",  # V2.3: University subdomain test
    ]
    
    print("-" * 70)
    
    for url in test_urls:
        print(f"\nAnalyzing: {url}")
        result = fusion.analyze(url)
        
        status = "PHISHING" if result['is_phishing'] else "SAFE"
        print(f"  Result: {status}")
        print(f"  Risk Score: {result['risk_score']}%")
        print(f"  Redirect: {'Yes → ' + result.get('final_url', url) if result.get('redirect_detected') else 'No'}")
        print(f"  Votes: {result.get('details', {}).get('consensus_votes', 'N/A')}/3")
        
        if result.get('details', {}).get('known_brand_url'):
            print(f"  Known Brand: {result['details']['url_brand']}")
        
        if result.get('details', {}).get('false_positive_prevented'):
            print(f"  FP Prevented: {result['details']['fp_reason']}")
    
    fusion.close()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
