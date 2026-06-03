import os
import sys
import logging
import shutil
from datetime import datetime
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        "historical_data/processed_events",
        "historical_data/processed_markets",
        "historical_data/similar_events",
    ]
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Checked directory: {directory}")

def run_subprocess(command, cwd=None):
    """Run a subprocess with controlled output."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        for line in result.stdout.split('\n'):
            if any(key in line.lower() for key in ['collected', 'processed', 'saved', 'similar']):
                logger.info(line.strip())
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(e.stderr)
        return False

def copy_processed_data(date_str):
    """Copy processed data from pipelines to main directory."""
    sources = {
        'kalshi': {
            'source': f'./kalshi_pipeline/historical_data/processed_markets/processed_markets_{date_str}.csv',
            'dest': f'./historical_data/processed_markets/kalshi_markets_{date_str}.csv'
        },
        'polymarket': {
            'source': f'./polymarket_pipeline/historical_data/open_markets/open_markets_{date_str}.csv',
            'dest': f'./historical_data/processed_markets/polymarket_markets_{date_str}.csv'
        }
    }
    
    for source_name, paths in sources.items():
        if os.path.exists(paths['source']):
            # Create the destination directory if it doesn't exist
            os.makedirs(os.path.dirname(paths['dest']), exist_ok=True)
            # Copy the file
            shutil.copy2(paths['source'], paths['dest'])
            logger.info(f"Copied {source_name} markets data to: {paths['dest']}")
        else:
            logger.warning(f"Source file not found: {paths['source']}")
            
def run_pipeline(matcher_script="find_similar_events.py"):
    """Run the complete market data pipeline.

    matcher_script selects the similarity step:
      - find_similar_events.py       (SBERT, needs PyTorch)
      - find_similar_events_lite.py  (TF-IDF, no PyTorch — host-friendly)
    """
    date_today = datetime.now().strftime('%Y%m%d')
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Run Kalshi Pipeline
    logger.info("Running Kalshi pipeline...")
    if not run_subprocess([sys.executable, "pipeline_all_kalshi.py"], cwd="./kalshi_pipeline"):
        return
    
    # Step 3: Run Polymarket Pipeline
    logger.info("Running Polymarket pipeline...")
    if not run_subprocess([sys.executable, "pipeline_all_poly.py"], cwd="./polymarket_pipeline"):
        return
    
    # Step 4: Copy processed data
    logger.info("Copying processed data...")
    copy_processed_data(date_today)
    
    # Step 5: Run similarity analysis
    logger.info(f"Running similarity analysis ({matcher_script})...")
    if not run_subprocess(
        [sys.executable, matcher_script],
        cwd="./similarity_analysis"
    ):
        return
    
    logger.info(f"Pipeline completed successfully - {date_today}")

if __name__ == "__main__":
    start_time = datetime.now()
    lite = "--lite" in sys.argv
    matcher = "find_similar_events_lite.py" if lite else "find_similar_events.py"
    logger.info(f"Starting complete market data pipeline ({'lite/TF-IDF' if lite else 'SBERT'})...")
    
    try:
        run_pipeline(matcher_script=matcher)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Pipeline execution completed in {duration:.2f} seconds")