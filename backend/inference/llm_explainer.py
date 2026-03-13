"""
PhishLens - LLM Explainer Module
Generates human-readable explanations for phishing detection results.

Features:
- Llama 3 8B Instruct on CPU (for demo/testing)
- Template-based explanations (default, instant)
- Easy toggle via config.json
- Graceful fallback if LLM fails

Usage:
    from backend.inference.llm_explainer import LLMExplainer
    
    explainer = LLMExplainer()
    explanation = explainer.explain(analysis_result)
    print(explanation)

Config Options:
    "llm": {
        "enabled": false,      <- Set to true for LLM, false for templates
        "device": "cpu",       <- Use "cpu" for RTX 5070 compatibility
        "use_quantization": false  <- Must be false for CPU
    }
"""

import os
import sys
import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

# Add project root
sys.path.insert(0, r"C:\PhishLens")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Result from explanation generation."""
    explanation: str
    method: str  # 'llm' or 'template'
    success: bool
    generation_time: float = 0.0
    error: Optional[str] = None


class TemplateExplainer:
    """
    Template-based explanation generator.
    Used when LLM is disabled or as fallback.
    Fast and reliable.
    """
    
    def __init__(self):
        self.phishing_templates = {
            'url_suspicious': [
                "The URL contains suspicious patterns commonly found in phishing attempts.",
                "The web address has characteristics typical of malicious sites.",
                "URL analysis detected anomalies that suggest this may be a phishing site."
            ],
            'content_phishing': [
                "The page content contains elements commonly used in phishing attacks.",
                "Text patterns and page structure suggest this is a phishing attempt.",
                "The webpage includes suspicious content designed to deceive users."
            ],
            'brand_impersonation': [
                "This site appears to be impersonating {brand}. The visual elements match {brand} but the URL is not official.",
                "WARNING: Brand impersonation detected! The page displays {brand} branding but is hosted on an unofficial domain.",
                "This website is mimicking {brand}'s appearance. Do not enter any personal information."
            ],
            'credential_harvesting': [
                "This page contains forms designed to steal your login credentials.",
                "WARNING: Credential harvesting detected! The site has suspicious login forms.",
                "The page requests sensitive information (passwords, personal data) in a suspicious manner."
            ],
            'multi_vector_attack': [
                "Multiple phishing indicators detected! This site shows signs of URL manipulation, suspicious content, and potential brand impersonation.",
                "DANGER: This is a sophisticated phishing attack using multiple deception techniques.",
                "High-risk phishing site detected with multiple attack vectors."
            ]
        }
        
        self.safe_templates = [
            "This website appears to be legitimate. No phishing indicators were detected by our analysis.",
            "Our multi-modal analysis found no signs of phishing on this website.",
            "This site passed all security checks. URL, content, and visual analysis indicate it is safe."
        ]
        
        self.whitelisted_template = "This website ({domain}) is in our trusted whitelist and is considered safe."
        self.blacklisted_template = "WARNING: This website ({domain}) is in our known phishing blacklist. Do not proceed!"
    
    def generate(self, analysis_result: Dict) -> str:
        """Generate template-based explanation."""
        import random
        
        # Check whitelist/blacklist
        if analysis_result.get('whitelisted'):
            domain = analysis_result.get('details', {}).get('whitelist_match', 'this domain')
            return self.whitelisted_template.format(domain=domain)
        
        if analysis_result.get('blacklisted'):
            domain = analysis_result.get('details', {}).get('blacklist_match', 'this domain')
            return self.blacklisted_template.format(domain=domain)
        
        # Generate explanation based on verdict
        if not analysis_result.get('is_phishing', False):
            return random.choice(self.safe_templates)
        
        # Phishing detected - build detailed explanation
        threat_type = analysis_result.get('threat_type', 'content_phishing')
        risk_score = analysis_result.get('risk_score', 0)
        details = analysis_result.get('details', {})
        
        explanation_parts = []
        
        # Main threat explanation
        if threat_type in self.phishing_templates:
            template = random.choice(self.phishing_templates[threat_type])
            
            # Handle brand impersonation specially
            if threat_type == 'brand_impersonation':
                brand = details.get('visual_analysis', {}).get('detected_brand', 'a known brand')
                template = template.format(brand=brand)
            
            explanation_parts.append(template)
        else:
            explanation_parts.append(random.choice(self.phishing_templates['content_phishing']))
        
        # Add specific findings
        findings = []
        
        # URL findings
        url_analysis = details.get('url_analysis', {})
        if url_analysis.get('is_phishing'):
            findings.append("suspicious URL patterns")
        
        # HTML findings
        html_analysis = details.get('html_analysis', {})
        if html_analysis.get('is_phishing'):
            findings.append("malicious content elements")
        
        # Visual findings
        visual_analysis = details.get('visual_analysis', {})
        if visual_analysis.get('brand_url_mismatch'):
            brand = visual_analysis.get('detected_brand', 'a brand')
            findings.append(f"impersonation of {brand}")
        
        if findings:
            explanation_parts.append(f"Specific concerns: {', '.join(findings)}.")
        
        # Add risk level warning
        if risk_score >= 80:
            explanation_parts.append("This is a HIGH-RISK site. Do not enter any personal information!")
        elif risk_score >= 60:
            explanation_parts.append("Exercise extreme caution with this website.")
        
        return " ".join(explanation_parts)


class LLMExplainer:
    """
    LLM-based explanation generator using Llama 3 8B Instruct.
    Supports CPU mode for RTX 5070 compatibility.
    Falls back to template-based explanations if disabled or on error.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LLM Explainer.
        
        Args:
            config_path: Path to config.json
        """
        self.config_path = config_path or r"C:\PhishLens\backend\inference\config.json"
        self.config = self._load_config()
        
        # LLM settings
        self.llm_config = self.config.get('llm', {})
        self.enabled = self.llm_config.get('enabled', False)
        self.model_name = self.llm_config.get('model_name', 'unsloth/llama-3-8b-Instruct')
        self.device = self.llm_config.get('device', 'cpu')
        self.use_quantization = self.llm_config.get('use_quantization', False)
        self.max_new_tokens = self.llm_config.get('max_new_tokens', 200)
        self.temperature = self.llm_config.get('temperature', 0.7)
        self.fallback_to_template = self.llm_config.get('fallback_to_template', True)
        
        # Template explainer (always available)
        self.template_explainer = TemplateExplainer()
        
        # LLM components
        self.model = None
        self.tokenizer = None
        self.llm_loaded = False
        
        if self.enabled:
            self._load_llm()
        else:
            logger.info("LLM Explainer initialized (LLM disabled, using templates)")
    
    def _load_config(self) -> dict:
        """Load configuration."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return {}
    
    def _load_llm(self):
        """Load the Llama model on CPU (no quantization)."""
        try:
            logger.info(f"Loading LLM: {self.model_name}")
            logger.info(f"Device: {self.device} (CPU mode for RTX 5070 compatibility)")
            logger.info("This may take 1-2 minutes...")
            
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model on CPU (no quantization)
            logger.info("Loading model on CPU (this takes ~1-2 minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            self.llm_loaded = True
            
            # Get model size
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"LLM loaded successfully! Parameters: {param_count:,}")
            logger.info("NOTE: CPU inference is slow (~30-60 seconds per explanation)")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            self.llm_loaded = False
            if self.fallback_to_template:
                logger.info("Will use template-based explanations instead")
    
    def _build_prompt(self, analysis_result: Dict) -> str:
        """Build prompt for LLM."""
        url = analysis_result.get('url', 'Unknown URL')
        is_phishing = analysis_result.get('is_phishing', False)
        risk_score = analysis_result.get('risk_score', 0)
        threat_type = analysis_result.get('threat_type', 'none')
        details = analysis_result.get('details', {})
        
        # Extract key findings
        findings = []
        
        # URL analysis
        url_analysis = details.get('url_analysis', {})
        if url_analysis.get('is_phishing'):
            findings.append(f"- URL Analysis: SUSPICIOUS (risk: {url_analysis.get('risk_score', 0):.1f}%)")
        else:
            findings.append(f"- URL Analysis: SAFE (risk: {url_analysis.get('risk_score', 0):.1f}%)")
        
        # HTML analysis
        html_analysis = details.get('html_analysis', {})
        if html_analysis.get('is_phishing'):
            findings.append(f"- Content Analysis: SUSPICIOUS (risk: {html_analysis.get('risk_score', 0):.1f}%)")
        else:
            findings.append(f"- Content Analysis: SAFE (risk: {html_analysis.get('risk_score', 0):.1f}%)")
        
        # Visual analysis
        visual_analysis = details.get('visual_analysis', {})
        detected_brand = visual_analysis.get('detected_brand')
        brand_mismatch = visual_analysis.get('brand_url_mismatch', False)
        
        if detected_brand:
            findings.append(f"- Detected Brand: {detected_brand}")
        if brand_mismatch:
            findings.append("- Brand-URL Mismatch: YES (CRITICAL - Brand impersonation!)")
        
        # Build prompt
        verdict = "PHISHING" if is_phishing else "SAFE"
        findings_text = "\n".join(findings)
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a cybersecurity expert. Explain phishing detection results clearly and concisely.
Keep your response under 100 words. Do not use markdown.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Explain this phishing scan result:

URL: {url}
Verdict: {verdict}
Risk Score: {risk_score:.1f}%
Threat Type: {threat_type.replace('_', ' ').title()}

Findings:
{findings_text}

Give a brief, clear explanation for the user.<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
    
    def _generate_llm_explanation(self, analysis_result: Dict) -> str:
        """Generate explanation using LLM."""
        if not self.llm_loaded or not self.model or not self.tokenizer:
            raise RuntimeError("LLM not loaded")
        
        import torch
        
        # Build prompt
        prompt = self._build_prompt(analysis_result)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant" in generated_text.lower():
            parts = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")
            if len(parts) > 1:
                explanation = parts[-1].strip()
            else:
                # Try another split method
                explanation = generated_text.split("assistant")[-1].strip()
        else:
            explanation = generated_text[len(prompt):].strip()
        
        # Clean up
        explanation = explanation.replace("<|eot_id|>", "").strip()
        explanation = explanation.replace("<|end_of_text|>", "").strip()
        
        # Ensure we have something
        if not explanation or len(explanation) < 10:
            raise RuntimeError("LLM generated empty response")
        
        return explanation
    
    def explain(self, analysis_result: Dict) -> ExplanationResult:
        """
        Generate explanation for analysis result.
        
        Args:
            analysis_result: Result from fusion engine
            
        Returns:
            ExplanationResult with explanation text and method used
        """
        import time
        start_time = time.time()
        
        # If LLM is disabled, use template
        if not self.enabled:
            explanation = self.template_explainer.generate(analysis_result)
            return ExplanationResult(
                explanation=explanation,
                method='template',
                success=True,
                generation_time=time.time() - start_time
            )
        
        # Try LLM first
        if self.llm_loaded:
            try:
                logger.info("Generating LLM explanation (this may take 30-60 seconds)...")
                explanation = self._generate_llm_explanation(analysis_result)
                return ExplanationResult(
                    explanation=explanation,
                    method='llm',
                    success=True,
                    generation_time=time.time() - start_time
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                
                # Fallback to template if enabled
                if self.fallback_to_template:
                    explanation = self.template_explainer.generate(analysis_result)
                    return ExplanationResult(
                        explanation=explanation,
                        method='template',
                        success=True,
                        generation_time=time.time() - start_time,
                        error=f"LLM failed, used template: {str(e)}"
                    )
                else:
                    return ExplanationResult(
                        explanation="Unable to generate explanation.",
                        method='none',
                        success=False,
                        generation_time=time.time() - start_time,
                        error=str(e)
                    )
        
        # LLM not loaded, use template
        explanation = self.template_explainer.generate(analysis_result)
        return ExplanationResult(
            explanation=explanation,
            method='template',
            success=True,
            generation_time=time.time() - start_time,
            error="LLM not loaded, used template"
        )
    
    def explain_with_template(self, analysis_result: Dict) -> ExplanationResult:
        """Force template-based explanation (always fast)."""
        import time
        start_time = time.time()
        explanation = self.template_explainer.generate(analysis_result)
        return ExplanationResult(
            explanation=explanation,
            method='template',
            success=True,
            generation_time=time.time() - start_time
        )
    
    def explain_with_llm(self, analysis_result: Dict) -> ExplanationResult:
        """Force LLM-based explanation (requires LLM loaded)."""
        import time
        start_time = time.time()
        
        if not self.llm_loaded:
            return ExplanationResult(
                explanation="LLM not loaded.",
                method='none',
                success=False,
                generation_time=time.time() - start_time,
                error="LLM not loaded"
            )
        
        try:
            logger.info("Generating LLM explanation...")
            explanation = self._generate_llm_explanation(analysis_result)
            return ExplanationResult(
                explanation=explanation,
                method='llm',
                success=True,
                generation_time=time.time() - start_time
            )
        except Exception as e:
            return ExplanationResult(
                explanation="LLM generation failed.",
                method='none',
                success=False,
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    def is_llm_enabled(self) -> bool:
        """Check if LLM is enabled in config."""
        return self.enabled
    
    def is_llm_loaded(self) -> bool:
        """Check if LLM is loaded and ready."""
        return self.llm_loaded
    
    def get_status(self) -> Dict:
        """Get explainer status."""
        return {
            'enabled': self.enabled,
            'llm_loaded': self.llm_loaded,
            'model_name': self.model_name if self.enabled else None,
            'device': self.device if self.enabled else None,
            'fallback_to_template': self.fallback_to_template,
            'method_available': 'llm' if self.llm_loaded else 'template'
        }


def main():
    """Test the LLM Explainer."""
    print("=" * 65)
    print("PhishLens LLM Explainer - Test")
    print("=" * 65)
    
    # Initialize explainer
    explainer = LLMExplainer()
    
    # Show status
    status = explainer.get_status()
    print(f"\nExplainer Status:")
    print(f"  LLM Enabled:  {status['enabled']}")
    print(f"  LLM Loaded:   {status['llm_loaded']}")
    print(f"  Device:       {status['device']}")
    print(f"  Method:       {status['method_available']}")
    
    # Test with mock results
    print("\n" + "-" * 65)
    print("Testing Explanations:")
    print("-" * 65)
    
    # Test 1: Safe site
    safe_result = {
        'url': 'https://www.google.com',
        'is_phishing': False,
        'risk_score': 2.5,
        'threat_type': 'none',
        'whitelisted': False,
        'details': {
            'url_analysis': {'is_phishing': False, 'risk_score': 1.2},
            'html_analysis': {'is_phishing': False, 'risk_score': 3.1},
            'visual_analysis': {'detected_brand': 'Google', 'brand_url_mismatch': False}
        }
    }
    
    print("\n[Test 1] Safe website (google.com)")
    result = explainer.explain(safe_result)
    print(f"  Method: {result.method}")
    print(f"  Time: {result.generation_time:.2f}s")
    print(f"  Explanation: {result.explanation}")
    
    # Test 2: Phishing site
    phishing_result = {
        'url': 'https://paypa1-secure-login.com',
        'is_phishing': True,
        'risk_score': 85.5,
        'threat_type': 'brand_impersonation',
        'whitelisted': False,
        'details': {
            'url_analysis': {'is_phishing': True, 'risk_score': 78.5},
            'html_analysis': {'is_phishing': True, 'risk_score': 82.3},
            'visual_analysis': {
                'detected_brand': 'PayPal',
                'brand_url_mismatch': True
            }
        }
    }
    
    print("\n[Test 2] Phishing website (fake PayPal)")
    result = explainer.explain(phishing_result)
    print(f"  Method: {result.method}")
    print(f"  Time: {result.generation_time:.2f}s")
    print(f"  Explanation: {result.explanation}")
    
    # Test 3: Whitelisted
    whitelisted_result = {
        'url': 'https://www.microsoft.com',
        'is_phishing': False,
        'risk_score': 0,
        'whitelisted': True,
        'details': {
            'whitelist_match': 'microsoft.com'
        }
    }
    
    print("\n[Test 3] Whitelisted website (microsoft.com)")
    result = explainer.explain(whitelisted_result)
    print(f"  Method: {result.method}")
    print(f"  Time: {result.generation_time:.2f}s")
    print(f"  Explanation: {result.explanation}")
    
    print("\n" + "=" * 65)
    print("Test complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
