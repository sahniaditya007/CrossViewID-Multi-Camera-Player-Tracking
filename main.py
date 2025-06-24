"""
main.py

Main entry point for CrossViewID Multi-Camera Player Tracking.

This script orchestrates the detection, tracking, and cross-view matching of players in synchronized sports videos (e.g., broadcast and tacticam). It loads the YOLO model, runs detection on both videos, tracks players in each view, matches players across views, and saves the results to a JSON file.

Functions:
    check_requirements: Ensure all required model and video files are present.
    setup_environment: Prepare output directory and device configuration.
    run_detection_pipeline: Run player detection on both videos.
    run_tracking_pipeline: Track players in each video.
    run_matching_pipeline: Match player tracks across camera views.
    display_results: Print summary of results to the console.
    save_results: Save results and statistics to a JSON file.
    main: Main workflow for the pipeline.
"""
import sys
import os
from pathlib import Path
import torch
import logging
from datetime import datetime

MODEL_DIR = Path('models')
DATA_DIR = Path('data')

src_path = Path(__file__).parent / 'utils'
sys.path.insert(0, str(src_path))

from detector import run_detection
from tracker import track_players
from matcher import match_players_across_views

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """
    Check for the presence of required model and video files.

    Returns:
        bool: True if all required files are present, False otherwise.
    """
    required_files = {
        str(MODEL_DIR / 'best.pt'): 'YOLO model file',
        str(DATA_DIR / 'broadcast.mp4'): 'Broadcast video file',
        str(DATA_DIR / 'tacticam.mp4'): 'Tacticam video file'
    }
    missing_files = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_path} ({description})")
    if missing_files:
        logger.error("Missing required files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        return False
    logger.info("All required files found")
    return True

def setup_environment():
    """
    Set up output directory and device configuration.

    Returns:
        dict: Configuration dictionary with paths and device info.
    """
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'model_path': MODEL_DIR / 'best.pt',
        'broadcast_video': DATA_DIR / 'broadcast.mp4',
        'tacticam_video': DATA_DIR / 'tacticam.mp4',
        'output_dir': output_dir,
        'device': device
    }
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    return config

def run_detection_pipeline(config):
    """
    Run player detection on both broadcast and tacticam videos.

    Args:
        config (dict): Configuration dictionary.
    Returns:
        Tuple[List[List[dict]], List[List[dict]]]: Detections for both videos.
    """
    logger.info("=== DETECTION PHASE ===")
    logger.info("Processing broadcast video...")
    broadcast_detections = run_detection(
        config['broadcast_video'], 
        config['model_path'], 
        device=config['device']
    )
    logger.info("Processing tacticam video...")
    tacticam_detections = run_detection(
        config['tacticam_video'], 
        config['model_path'], 
        device=config['device']
    )
    broadcast_total = sum(len(frame) for frame in broadcast_detections)
    tacticam_total = sum(len(frame) for frame in tacticam_detections)
    logger.info(f"Broadcast: {len(broadcast_detections)} frames, {broadcast_total} detections")
    logger.info(f"Tacticam: {len(tacticam_detections)} frames, {tacticam_total} detections")
    return broadcast_detections, tacticam_detections

def run_tracking_pipeline(broadcast_detections, tacticam_detections, config):
    """
    Track players in both broadcast and tacticam videos.

    Args:
        broadcast_detections (List[List[dict]]): Detections for broadcast video.
        tacticam_detections (List[List[dict]]): Detections for tacticam video.
        config (dict): Configuration dictionary.
    Returns:
        Tuple[dict, dict]: Valid tracks for both videos.
    """
    logger.info("=== TRACKING PHASE ===")
    logger.info("Tracking players in broadcast video...")
    broadcast_tracks = track_players(
        broadcast_detections, 
        device=config['device']
    )
    logger.info("Tracking players in tacticam video...")
    tacticam_tracks = track_players(
        tacticam_detections, 
        device=config['device']
    )
    logger.info(f"Broadcast: {len(broadcast_tracks)} valid tracks")
    logger.info(f"Tacticam: {len(tacticam_tracks)} valid tracks")
    return broadcast_tracks, tacticam_tracks

def run_matching_pipeline(broadcast_tracks, tacticam_tracks, config):
    """
    Match player tracks across broadcast and tacticam views.

    Args:
        broadcast_tracks (dict): Tracks from broadcast video.
        tacticam_tracks (dict): Tracks from tacticam video.
        config (dict): Configuration dictionary.
    Returns:
        dict: Mapping from tacticam track IDs to broadcast track IDs.
    """
    logger.info("=== MATCHING PHASE ===")
    logger.info("Matching players across camera views...")
    player_mapping = match_players_across_views(
        broadcast_tracks, 
        tacticam_tracks, 
        device=config['device']
    )
    logger.info(f"Successfully matched {len(player_mapping)} player pairs")
    return player_mapping

def display_results(broadcast_tracks, tacticam_tracks, player_mapping):
    """
    Print a summary of the results to the console.

    Args:
        broadcast_tracks (dict): Tracks from broadcast video.
        tacticam_tracks (dict): Tracks from tacticam video.
        player_mapping (dict): Mapping from tacticam to broadcast track IDs.
    """
    print("\n" + "=" * 60)
    print("CROSSVIEWID RESULTS")
    print("=" * 60)
    print(f"Broadcast video: {len(broadcast_tracks)} player tracks")
    print(f"Tacticam video: {len(tacticam_tracks)} player tracks")
    print(f"Cross-camera matches: {len(player_mapping)}")
    if player_mapping:
        print("\n Player ID Mapping:")
        for tacticam_id, broadcast_id in sorted(player_mapping.items()):
            print(f"   Tacticam #{tacticam_id} → Broadcast #{broadcast_id}")
        match_rate = len(player_mapping) / len(tacticam_tracks) * 100
        print(f"\n📊 Match rate: {match_rate:.1f}%")
    else:
        print("\n  No cross-camera matches found")
        print("   This could be due to:")
        print("   - Different time segments in videos")
        print("   - Large camera angle differences")
        print("   - Low detection quality")
    print("\n" + "=" * 60)

def save_results(broadcast_tracks, tacticam_tracks, player_mapping, config):
    """
    Save results and statistics to a JSON file in the output directory.

    Args:
        broadcast_tracks (dict): Tracks from broadcast video.
        tacticam_tracks (dict): Tracks from tacticam video.
        player_mapping (dict): Mapping from tacticam to broadcast track IDs.
        config (dict): Configuration dictionary.
    """
    import json
    output_dir = config['output_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'config': {
            'device': config['device'],
            'model_path': str(config['model_path']),
            'broadcast_video': str(config['broadcast_video']),
            'tacticam_video': str(config['tacticam_video'])
        },
        'statistics': {
            'broadcast_tracks': len(broadcast_tracks),
            'tacticam_tracks': len(tacticam_tracks),
            'matched_players': len(player_mapping),
            'match_rate': len(player_mapping) / len(tacticam_tracks) if tacticam_tracks else 0
        },
        'player_mapping': player_mapping
    }
    output_file = output_dir / f"crossviewid_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_file}")

def main():
    """
    Main workflow for CrossViewID pipeline: detection, tracking, matching, and saving results.

    Returns:
        int: Exit code (0 for success, 1 for error/interruption).
    """
    try:
        print("Starting CrossViewID Multi-Camera Player Tracking")
        print("=" * 60)
        if not check_requirements():
            logger.warning("Proceeding in dummy mode: required assets are missing. Generating empty results JSON.")
            config = setup_environment()
            broadcast_tracks = {}
            tacticam_tracks = {}
            player_mapping = {}
            save_results(broadcast_tracks, tacticam_tracks, player_mapping, config)
            logger.info("✅ Dummy run completed – empty results saved.")
            return 0
        config = setup_environment()
        broadcast_detections, tacticam_detections = run_detection_pipeline(config)
        broadcast_tracks, tacticam_tracks = run_tracking_pipeline(
            broadcast_detections, tacticam_detections, config
        )
        player_mapping = run_matching_pipeline(
            broadcast_tracks, tacticam_tracks, config
        )
        display_results(broadcast_tracks, tacticam_tracks, player_mapping)
        save_results(broadcast_tracks, tacticam_tracks, player_mapping, config)
        logger.info("CrossViewID completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.info("\n  Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f" Error during execution: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

