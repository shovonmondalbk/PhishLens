"""
PhishLens - FastAPI Backend API (Optimized for Extension)
REST API for phishing detection system with batch processing.

Run:
    cd C:\PhishLens
    venv\Scripts\activate
    uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import json
import logging
import asyncio
from typing import Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add project root
sys.path.insert(0, r"C:\PhishLens")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=10)

# ============================================================================
# Pydantic Models
# ============================================================================

class URLRequest(BaseModel):
    url: str

class BatchURLRequest(BaseModel):
    urls: List[str]

class WhitelistRequest(BaseModel):
    domain: str

class BlacklistRequest(BaseModel):
    domain: str

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="PhishLens API",
    description="Multi-Modal AI-Powered Phishing Detection System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State
# ============================================================================

fusion_engine = None
llm_explainer = None
start_time = time.time()

# In-memory cache
url_cache = {}
CACHE_TTL = 3600  # 1 hour

def get_fusion_engine():
    """Lazy load fusion engine."""
    global fusion_engine
    if fusion_engine is None:
        logger.info("Loading Fusion Engine...")
        from backend.inference.fusion_engine import PhishLensFusion
        fusion_engine = PhishLensFusion()
        logger.info("Fusion Engine loaded!")
    return fusion_engine

def get_llm_explainer():
    """Lazy load LLM explainer."""
    global llm_explainer
    if llm_explainer is None:
        logger.info("Loading LLM Explainer...")
        from backend.inference.llm_explainer import LLMExplainer
        llm_explainer = LLMExplainer()
        logger.info("LLM Explainer loaded!")
    return llm_explainer

def load_config():
    """Load config.json."""
    config_path = r"C:\PhishLens\backend\inference\config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def normalize_url(url: str) -> str:
    """Normalize URL for caching."""
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url.rstrip('/')

def get_cached(key: str):
    """Get cached result."""
    if key in url_cache:
        cached = url_cache[key]
        if time.time() - cached['ts'] < CACHE_TTL:
            return cached['data']
    return None

def set_cache(key: str, data: dict):
    """Cache result."""
    url_cache[key] = {'data': data, 'ts': time.time()}

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {"message": "PhishLens API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/status")
async def get_status():
    """System status."""
    config = load_config()
    return {
        "status": "online",
        "version": "1.0.0",
        "analyzers": {
            "url": config.get("analyzers", {}).get("url", {}).get("enabled", True),
            "html": config.get("analyzers", {}).get("html", {}).get("enabled", True),
            "visual": config.get("analyzers", {}).get("visual", {}).get("enabled", True)
        },
        "llm_enabled": config.get("llm", {}).get("enabled", False),
        "whitelist_count": len(config.get("whitelist", {}).get("domains", [])),
        "blacklist_count": len(config.get("blacklist", {}).get("domains", [])),
        "cache_size": len(url_cache),
        "uptime": round(time.time() - start_time, 2)
    }

# ============================================================================
# Quick Check (URL-only, fast)
# ============================================================================

@app.post("/api/quick-check")
async def quick_check(request: URLRequest):
    """Quick URL-only analysis."""
    url = normalize_url(request.url)
    if not url:
        raise HTTPException(400, "URL required")
    
    # Check cache
    cached = get_cached(f"q:{url}")
    if cached:
        return {**cached, "cached": True}
    
    try:
        engine = get_fusion_engine()
        
        # Whitelist check
        is_wl, _ = engine._check_whitelist(url)
        if is_wl:
            result = {"url": url, "is_phishing": False, "risk_score": 0, 
                     "label": "legitimate", "whitelisted": True, "blacklisted": False}
            set_cache(f"q:{url}", result)
            return result
        
        # Blacklist check
        is_bl, _ = engine._check_blacklist(url)
        if is_bl:
            result = {"url": url, "is_phishing": True, "risk_score": 100,
                     "label": "phishing", "whitelisted": False, "blacklisted": True}
            set_cache(f"q:{url}", result)
            return result
        
        # URL analysis only
        url_result = engine._run_url_analysis(url)
        result = {
            "url": url,
            "is_phishing": url_result.is_phishing,
            "risk_score": round(url_result.risk_score, 2),
            "confidence": round(url_result.confidence, 2),
            "label": "phishing" if url_result.is_phishing else "legitimate",
            "whitelisted": False,
            "blacklisted": False
        }
        set_cache(f"q:{url}", result)
        return result
        
    except Exception as e:
        logger.error(f"Quick check error: {e}")
        raise HTTPException(500, str(e))

# ============================================================================
# Quick Batch Check (Multiple URLs, fast)
# ============================================================================

@app.post("/api/quick-batch")
async def quick_batch(request: BatchURLRequest):
    """Batch quick URL check (up to 50 URLs)."""
    urls = [normalize_url(u) for u in request.urls if u.strip()][:50]
    if not urls:
        raise HTTPException(400, "URLs required")
    
    engine = get_fusion_engine()
    results = []
    
    for url in urls:
        try:
            # Check cache
            cached = get_cached(f"q:{url}")
            if cached:
                results.append({**cached, "cached": True})
                continue
            
            # Whitelist
            is_wl, _ = engine._check_whitelist(url)
            if is_wl:
                r = {"url": url, "is_phishing": False, "risk_score": 0,
                    "label": "legitimate", "whitelisted": True, "blacklisted": False}
                set_cache(f"q:{url}", r)
                results.append(r)
                continue
            
            # Blacklist
            is_bl, _ = engine._check_blacklist(url)
            if is_bl:
                r = {"url": url, "is_phishing": True, "risk_score": 100,
                    "label": "phishing", "whitelisted": False, "blacklisted": True}
                set_cache(f"q:{url}", r)
                results.append(r)
                continue
            
            # URL analysis
            url_result = engine._run_url_analysis(url)
            r = {
                "url": url,
                "is_phishing": url_result.is_phishing,
                "risk_score": round(url_result.risk_score, 2),
                "label": "phishing" if url_result.is_phishing else "legitimate",
                "whitelisted": False,
                "blacklisted": False
            }
            set_cache(f"q:{url}", r)
            results.append(r)
            
        except Exception as e:
            results.append({"url": url, "error": str(e), "is_phishing": False, "risk_score": 0})
    
    return {
        "results": results,
        "total": len(results),
        "phishing_count": sum(1 for r in results if r.get("is_phishing")),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Full Analysis (All analyzers + explanation)
# ============================================================================

def _run_full_analysis(url: str) -> dict:
    """
    Run full analysis in a separate thread.
    This is needed because Playwright's sync API conflicts with FastAPI's async loop.
    """
    engine = get_fusion_engine()
    result = engine.analyze(url, return_details=True)
    
    explainer = get_llm_explainer()
    explanation = explainer.explain(result)
    
    return {
        "url": result.get("url", url),
        "is_phishing": result.get("is_phishing", False),
        "confidence": result.get("confidence", 0),
        "risk_score": result.get("risk_score", 0),
        "risk_level": result.get("risk_level", "SAFE"),
        "threat_type": result.get("threat_type", "none"),
        "label": result.get("label", "legitimate"),
        "summary": result.get("summary", ""),
        "explanation": explanation.explanation,
        "explanation_method": explanation.method,
        "analysis_time": result.get("analysis_time", 0),
        "whitelisted": result.get("whitelisted", False),
        "blacklisted": result.get("blacklisted", False),
        "details": result.get("details", {}),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/analyze")
async def analyze(request: URLRequest):
    """Full phishing analysis with all analyzers + explanation."""
    url = normalize_url(request.url)
    if not url:
        raise HTTPException(400, "URL required")
    
    # Check cache
    cached = get_cached(f"f:{url}")
    if cached:
        return {**cached, "cached": True}
    
    try:
        # Run analysis in thread executor to avoid Playwright async conflicts
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, _run_full_analysis, url)
        
        set_cache(f"f:{url}", response)
        return response
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(500, str(e))

# ============================================================================
# Batch Full Analysis
# ============================================================================

def _run_batch_analysis(urls: list) -> list:
    """Run batch analysis in a separate thread."""
    engine = get_fusion_engine()
    explainer = get_llm_explainer()
    results = []
    
    for url in urls:
        try:
            cached = get_cached(f"f:{url}")
            if cached:
                results.append({**cached, "cached": True})
                continue
            
            result = engine.analyze(url, return_details=True)
            explanation = explainer.explain(result)
            
            r = {
                "url": result.get("url", url),
                "is_phishing": result.get("is_phishing", False),
                "confidence": result.get("confidence", 0),
                "risk_score": result.get("risk_score", 0),
                "risk_level": result.get("risk_level", "SAFE"),
                "label": result.get("label", "legitimate"),
                "explanation": explanation.explanation,
                "explanation_method": explanation.method,
                "whitelisted": result.get("whitelisted", False),
                "blacklisted": result.get("blacklisted", False),
                "details": result.get("details", {})
            }
            set_cache(f"f:{url}", r)
            results.append(r)
            
        except Exception as e:
            results.append({"url": url, "error": str(e)})
    
    return results


@app.post("/api/analyze-batch")
async def analyze_batch(request: BatchURLRequest):
    """Batch full analysis (up to 10 URLs)."""
    urls = [normalize_url(u) for u in request.urls if u.strip()][:10]
    if not urls:
        raise HTTPException(400, "URLs required")
    
    # Run in thread executor
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(executor, _run_batch_analysis, urls)
    
    return {
        "results": results,
        "total": len(results),
        "phishing_count": sum(1 for r in results if r.get("is_phishing")),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Whitelist/Blacklist Management
# ============================================================================

@app.get("/api/whitelist")
async def get_whitelist():
    config = load_config()
    domains = config.get("whitelist", {}).get("domains", [])
    return {"domains": domains, "count": len(domains)}

@app.post("/api/whitelist")
async def add_whitelist(request: WhitelistRequest):
    try:
        engine = get_fusion_engine()
        success = engine.add_to_whitelist(request.domain)
        # Clear cache for domain
        keys = [k for k in url_cache if request.domain.lower() in k.lower()]
        for k in keys:
            del url_cache[k]
        return {"success": success, "message": f"Added {request.domain}"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/whitelist")
async def remove_whitelist(request: WhitelistRequest):
    try:
        engine = get_fusion_engine()
        success = engine.remove_from_whitelist(request.domain)
        return {"success": success, "message": f"Removed {request.domain}"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/blacklist")
async def get_blacklist():
    config = load_config()
    domains = config.get("blacklist", {}).get("domains", [])
    return {"domains": domains, "count": len(domains)}

@app.post("/api/blacklist")
async def add_blacklist(request: BlacklistRequest):
    try:
        engine = get_fusion_engine()
        success = engine.add_to_blacklist(request.domain)
        return {"success": success, "message": f"Added {request.domain}"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/blacklist")
async def remove_blacklist(request: BlacklistRequest):
    try:
        engine = get_fusion_engine()
        success = engine.remove_from_blacklist(request.domain)
        return {"success": success, "message": f"Removed {request.domain}"}
    except Exception as e:
        raise HTTPException(500, str(e))

# ============================================================================
# Cache Management
# ============================================================================

@app.get("/api/cache/stats")
async def cache_stats():
    return {"cached_urls": len(url_cache), "cache_ttl": CACHE_TTL}

@app.delete("/api/cache/clear")
async def clear_cache():
    global url_cache
    count = len(url_cache)
    url_cache = {}
    return {"cleared": count}

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("PhishLens API Server")
    print("=" * 50)
    print("API: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
