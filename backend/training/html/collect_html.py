"""
PhishLens - HTML Dataset Collector (60K URLs)
Fetches HTML content from URLs in the dataset.

This script:
1. Loads the URL dataset
2. Samples 60K URLs (30K phishing + 30K legitimate)
3. Fetches HTML content from each URL
4. Saves HTML with labels for training

Expected Results:
- Attempts: 60,000 URLs
- Success Rate: ~50-60%
- Usable Samples: ~28-36K
- Time: 12-24 hours
"""

import os
import sys
import time
import random
import hashlib
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import json
import logging
from tqdm import tqdm
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
log_dir = r"C:\PhishLens\logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "html_collection.log"))
    ]
)
logger = logging.getLogger(__name__)


class HTMLCollector:
    """
    Collects HTML content from URLs for training the HTML analyzer.
    Optimized for large-scale collection (60K URLs).
    """
    
    def __init__(
        self,
        output_dir: str = r"C:\PhishLens\data\raw\html",
        max_html_size: int = 500 * 1024,  # 500KB max
        timeout: int = 12,
        delay_range: Tuple[float, float] = (0.3, 0.8),  # Faster for large collection
        max_workers: int = 5
    ):
        """
        Initialize the HTML collector.
        
        Args:
            output_dir: Directory to save collected HTML
            max_html_size: Maximum HTML size in bytes
            timeout: Request timeout in seconds
            delay_range: Random delay range between requests (min, max)
            max_workers: Number of concurrent workers
        """
        self.output_dir = output_dir
        self.max_html_size = max_html_size
        self.timeout = timeout
        self.delay_range = delay_range
        self.max_workers = max_workers
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "html_files"), exist_ok=True)
        
        # Setup session with retries
        self.session = self._create_session()
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        ]
        
        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'timeout': 0,
            'too_large': 0,
            'empty': 0,
            'not_html': 0,
            'phishing_success': 0,
            'legitimate_success': 0
        }
        
        # Resume support
        self.resume_file = os.path.join(output_dir, "collection_progress.json")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with random user agent."""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def _generate_file_id(self, url: str) -> str:
        """Generate unique file ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()[:16]
    
    def _save_resume_state(self, processed_indices: set, collected_data: list):
        """Save current progress for resume capability."""
        state = {
            'processed_indices': list(processed_indices),
            'stats': self.stats,
            'last_update': datetime.now().isoformat()
        }
        with open(self.resume_file, 'w') as f:
            json.dump(state, f)
    
    def _load_resume_state(self) -> Tuple[set, dict]:
        """Load previous progress if exists."""
        if os.path.exists(self.resume_file):
            try:
                with open(self.resume_file, 'r') as f:
                    state = json.load(f)
                return set(state.get('processed_indices', [])), state.get('stats', {})
            except:
                pass
        return set(), {}
    
    def fetch_html(self, url: str, url_id: int, label: int) -> Optional[Dict]:
        """
        Fetch HTML content from a URL.
        
        Args:
            url: URL to fetch
            url_id: ID of the URL in dataset
            label: Label (0=legitimate, 1=phishing)
            
        Returns:
            Dictionary with HTML content and metadata, or None if failed
        """
        # Ensure URL has scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        file_id = self._generate_file_id(url)
        
        try:
            # Add random delay
            time.sleep(random.uniform(*self.delay_range))
            
            # Fetch URL
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=self.timeout,
                verify=False,
                allow_redirects=True
            )
            
            # Check status code
            if response.status_code != 200:
                self.stats['failed'] += 1
                return None
            
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type and 'text/plain' not in content_type:
                self.stats['not_html'] += 1
                return None
            
            # Check size
            content_length = len(response.content)
            if content_length > self.max_html_size:
                self.stats['too_large'] += 1
                html_content = response.content[:self.max_html_size].decode('utf-8', errors='ignore')
            else:
                html_content = response.text
            
            # Check if empty or too small
            if not html_content or len(html_content.strip()) < 100:
                self.stats['empty'] += 1
                return None
            
            # Success!
            self.stats['success'] += 1
            if label == 1:
                self.stats['phishing_success'] += 1
            else:
                self.stats['legitimate_success'] += 1
            
            return {
                'url_id': url_id,
                'url': url,
                'final_url': response.url,
                'file_id': file_id,
                'html_length': len(html_content),
                'status_code': response.status_code,
                'content_type': content_type,
                'html_content': html_content,
                'label': label
            }
            
        except requests.exceptions.Timeout:
            self.stats['timeout'] += 1
            return None
        except requests.exceptions.ConnectionError:
            self.stats['failed'] += 1
            return None
        except requests.exceptions.RequestException:
            self.stats['failed'] += 1
            return None
        except Exception:
            self.stats['failed'] += 1
            return None
    
    def collect_from_dataset(
        self,
        dataset_path: str,
        output_csv: str,
        sample_size: int = 100000,
        balance: bool = True,
        resume: bool = True
    ):
        """
        Collect HTML from URLs in the dataset.
        
        Args:
            dataset_path: Path to URL dataset CSV
            output_csv: Path to save HTML dataset CSV
            sample_size: Number of URLs to sample (60K for best results)
            balance: Whether to balance phishing/legitimate samples
            resume: Whether to resume from previous progress
        """
        logger.info("=" * 60)
        logger.info("PhishLens HTML Dataset Collector (60K URLs)")
        logger.info("=" * 60)
        
        # Load URL dataset
        logger.info(f"Loading URL dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        logger.info(f"Total URLs in dataset: {len(df)}")
        
        # Sample URLs
        if balance:
            n_per_class = sample_size // 2
            
            phishing_df = df[df['label'] == 1]
            legitimate_df = df[df['label'] == 0]
            
            # Sample from each class
            phishing_sample = phishing_df.sample(
                n=min(n_per_class, len(phishing_df)),
                random_state=42
            )
            legitimate_sample = legitimate_df.sample(
                n=min(n_per_class, len(legitimate_df)),
                random_state=42
            )
            
            sampled_df = pd.concat([phishing_sample, legitimate_sample]).sample(frac=1, random_state=42)
            logger.info(f"Sampled {len(phishing_sample)} phishing + {len(legitimate_sample)} legitimate URLs")
        else:
            sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        logger.info(f"Total URLs to fetch: {len(sampled_df)}")
        self.stats['total'] = len(sampled_df)
        
        # Check for resume
        processed_indices = set()
        collected_data = []
        
        if resume:
            processed_indices, saved_stats = self._load_resume_state()
            if processed_indices:
                logger.info(f"Resuming from previous progress: {len(processed_indices)} URLs already processed")
                self.stats.update(saved_stats)
                
                # Load existing collected data
                if os.path.exists(output_csv):
                    existing_df = pd.read_csv(output_csv)
                    collected_data = existing_df.to_dict('records')
                    logger.info(f"Loaded {len(collected_data)} existing records")
        
        # Prepare data
        urls_to_fetch = []
        for idx, row in sampled_df.iterrows():
            if idx not in processed_indices:
                urls_to_fetch.append((idx, row['id'], row['url'], row['label']))
        
        logger.info(f"URLs remaining to fetch: {len(urls_to_fetch)}")
        
        if not urls_to_fetch:
            logger.info("All URLs already processed!")
            return collected_data
        
        # Estimate time
        estimated_hours = len(urls_to_fetch) * 0.7 / 3600
        logger.info(f"\nEstimated time remaining: {estimated_hours:.1f} hours")
        logger.info("Progress is saved every 500 URLs. You can stop and resume anytime.\n")
        
        # Collect HTML
        html_files_dir = os.path.join(self.output_dir, "html_files")
        start_time = datetime.now()
        
        # Process URLs with progress bar
        for i, (idx, url_id, url, label) in enumerate(tqdm(urls_to_fetch, desc="Collecting HTML")):
            result = self.fetch_html(url, url_id, label)
            
            if result:
                # Save HTML file
                html_filename = f"{result['file_id']}.html"
                html_path = os.path.join(html_files_dir, html_filename)
                
                try:
                    with open(html_path, 'w', encoding='utf-8', errors='ignore') as f:
                        f.write(result['html_content'])
                    
                    # Add to collected data
                    collected_data.append({
                        'id': url_id,
                        'url': url,
                        'final_url': result['final_url'],
                        'label': label,
                        'file_id': result['file_id'],
                        'html_file': html_filename,
                        'html_length': result['html_length'],
                        'status_code': result['status_code']
                    })
                except Exception as e:
                    logger.warning(f"Error saving HTML file: {e}")
            
            # Mark as processed
            processed_indices.add(idx)
            
            # Save progress every 500 URLs
            if (i + 1) % 500 == 0:
                self._save_progress(collected_data, output_csv)
                self._save_resume_state(processed_indices, collected_data)
                
                elapsed = datetime.now() - start_time
                rate = (i + 1) / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                remaining = (len(urls_to_fetch) - i - 1) / rate if rate > 0 else 0
                
                logger.info(
                    f"Progress: {i+1}/{len(urls_to_fetch)} | "
                    f"Success: {self.stats['success']} | "
                    f"Failed: {self.stats['failed']} | "
                    f"Timeout: {self.stats['timeout']} | "
                    f"ETA: {remaining/3600:.1f}h"
                )
        
        # Final save
        self._save_progress(collected_data, output_csv)
        self._save_resume_state(processed_indices, collected_data)
        
        # Calculate time
        elapsed_time = datetime.now() - start_time
        
        # Print summary
        self._print_summary(elapsed_time, output_csv, html_files_dir)
        
        return collected_data
    
    def _save_progress(self, collected_data: list, output_csv: str):
        """Save current progress to CSV."""
        if collected_data:
            df = pd.DataFrame(collected_data)
            df.to_csv(output_csv, index=False)
    
    def _print_summary(self, elapsed_time, output_csv: str, html_files_dir: str):
        """Print collection summary."""
        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total URLs attempted: {self.stats['total']}")
        logger.info(f"Successfully collected: {self.stats['success']}")
        logger.info(f"  - Phishing: {self.stats['phishing_success']}")
        logger.info(f"  - Legitimate: {self.stats['legitimate_success']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Timeout: {self.stats['timeout']}")
        logger.info(f"Not HTML: {self.stats['not_html']}")
        logger.info(f"Too large: {self.stats['too_large']}")
        logger.info(f"Empty pages: {self.stats['empty']}")
        logger.info(f"Success rate: {self.stats['success']/self.stats['total']*100:.1f}%")
        logger.info(f"Time elapsed: {elapsed_time}")
        logger.info(f"\nDataset saved to: {output_csv}")
        logger.info(f"HTML files saved to: {html_files_dir}")
        
        # Save final statistics
        stats_path = os.path.join(self.output_dir, "collection_stats.json")
        with open(stats_path, 'w') as f:
            json.dump({
                **self.stats,
                'elapsed_time': str(elapsed_time),
                'output_csv': output_csv,
                'success_rate': self.stats['success']/self.stats['total']*100
            }, f, indent=2)


def main():
    """Main function to collect HTML dataset."""
    
    # Paths
    base_dir = r"C:\PhishLens"
    url_dataset_path = os.path.join(base_dir, "data", "raw", "url", "phishlens_url_dataset.csv")
    output_dir = os.path.join(base_dir, "data", "raw", "html")
    output_csv = os.path.join(output_dir, "phishlens_html_dataset.csv")
    
    print("=" * 60)
    print("PhishLens HTML Dataset Collector (100K URLs - FULL DATASET)")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Use ALL 100,000 URLs (50K phishing + 50K legitimate)")
    print("  2. Fetch HTML content from each URL")
    print("  3. Save HTML files and create a dataset CSV")
    print("\n📊 Expected Results:")
    print("  - Attempts: 100,000 URLs")
    print("  - Success Rate: ~50-60%")
    print("  - Usable Samples: ~50-60K")
    print("  - Estimated Time: 20-40 hours")
    print("\n✅ Features:")
    print("  - Progress saved every 500 URLs")
    print("  - Resume capability (can stop and continue later)")
    print("  - Detailed logging")
    print("\n" + "=" * 60)
    
    # Ask for confirmation
    confirm = input("\nDo you want to start collection? (yes/no): ").strip().lower()
    
    if confirm not in ('yes', 'y'):
        print("Collection cancelled.")
        return
    
    # Check for resume
    resume_file = os.path.join(output_dir, "collection_progress.json")
    resume = False
    if os.path.exists(resume_file):
        resume_choice = input("Previous progress found. Resume? (yes/no): ").strip().lower()
        resume = resume_choice in ('yes', 'y')
    
    # Initialize collector
    collector = HTMLCollector(
        output_dir=output_dir,
        max_html_size=500 * 1024,  # 500KB
        timeout=12,
        delay_range=(0.3, 0.8),
        max_workers=5
    )
    
    # Collect HTML - FULL 100K DATASET
    collector.collect_from_dataset(
        dataset_path=url_dataset_path,
        output_csv=output_csv,
        sample_size=100000,  # FULL DATASET
        balance=True,
        resume=resume
    )
    
    print("\n" + "=" * 60)
    print("HTML collection complete!")
    print(f"Dataset saved to: {output_csv}")
    print("=" * 60)
    print("\nNext step: Run HTML preprocessing and train ELECTRA model")


if __name__ == "__main__":
    main()