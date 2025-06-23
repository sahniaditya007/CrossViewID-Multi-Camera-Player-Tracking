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
    logger.info("All required files found ✓")
    return True

def setup_environment():
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
    print("\n" + "=" * 60)
    print("🎯 CROSSVIEWID RESULTS")
    print("=" * 60)
    print(f"📹 Broadcast video: {len(broadcast_tracks)} player tracks")
    print(f"📹 Tacticam video: {len(tacticam_tracks)} player tracks")
    print(f"🔗 Cross-camera matches: {len(player_mapping)}")
    if player_mapping:
        print("\n🔀 Player ID Mapping:")
        for tacticam_id, broadcast_id in sorted(player_mapping.items()):
            print(f"   Tacticam #{tacticam_id} → Broadcast #{broadcast_id}")
        match_rate = len(player_mapping) / len(tacticam_tracks) * 100
        print(f"\n📊 Match rate: {match_rate:.1f}%")
    else:
        print("\n⚠️  No cross-camera matches found")
        print("   This could be due to:")
        print("   - Different time segments in videos")
        print("   - Large camera angle differences")
        print("   - Low detection quality")
    print("\n" + "=" * 60)

def save_results(broadcast_tracks, tacticam_tracks, player_mapping, config):
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
    try:
        print("🚀 Starting CrossViewID Multi-Camera Player Tracking")
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
        logger.info("✅ CrossViewID completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.info("\n⚠️  Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

