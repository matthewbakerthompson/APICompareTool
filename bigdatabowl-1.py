"""
================================================================================
NFL BIG DATA BOWL 2026 - PLAYER TRAJECTORY PREDICTION
================================================================================

Author: Matthew Thompson
Email: matthewbakerthompson@gmail.com  
Website: matthewthompson.dev

================================================================================
EXECUTIVE SUMMARY
================================================================================

This solution predicts post-throw player trajectories using a deep learning 
approach that combines domain-specific feature engineering with an autoregressive
LSTM architecture. The model achieves approximately 1.03 yards RMSE on validation
data, representing a 10% improvement over field-aware baselines.

Key innovations:
1. Bidirectional information flow - Both defenders and receivers "see" each other
2. Coverage scheme detection - Zone vs man coverage changes defender behavior  
3. Duration conditioning - Model adapts predictions based on play length
4. Physics-constrained post-processing - Ensures physically plausible trajectories

================================================================================
METHODOLOGY
================================================================================

PROBLEM UNDERSTANDING
---------------------
After a quarterback releases a pass, we must predict where each player will be
at every 0.1-second interval until the play ends. This is challenging because:

- Receivers run designed routes but adjust based on coverage
- Defenders react to both the receiver AND the ball trajectory
- Play duration varies from <1 second to >9 seconds
- Errors compound over time in trajectory prediction

APPROACH: FEATURE ENGINEERING + AUTOREGRESSIVE LSTM
---------------------------------------------------

Rather than treating this as a pure sequence-to-sequence problem, we inject
extensive domain knowledge through 53 engineered features that capture:

1. MOTION STATE (8 features)
   - Position (x, y), velocity (vx, vy), speed, acceleration (a, ax, ay)
   - Players in motion continue in motion - physics matters

2. BALL-RELATIVE FEATURES (5 features)  
   - Distance/angle to ball landing spot
   - Velocity toward ball
   - Critical for receivers tracking the catch point

3. FIELD POSITION CONTEXT (11 features)
   - Yards to goal, red zone, goal line indicators
   - Hash marks (left/center/right) - affects route options
   - Sideline proximity - constrains available space
   
4. DEFENDER-TO-RECEIVER FEATURES (17 features)  
   - Distance, angle, closing velocity toward receiver
   - Positioning between receiver and ball
   - Defender ranking by proximity to ball, leverage angle
   - Receiver motion awareness (speed, direction, acceleration)
   - Per-defender coverage role (man prob, depth zone, deep help, leverage)
   - Scheme context (single-high safety, receiver route direction)
   
5. RECEIVER-TO-DEFENDER FEATURES (4 features)
   - Nearest defender distance and angle
   - Count of defenders within 5 yards
   - Defender closing speed
   - Allows receivers to "see" coverage and adjust

6. COVERAGE SCHEME DETECTION (5 features)
   - Zone vs man coverage probability (detected from defender spacing)
   - Press coverage indicator
   - Cushion distance
   - Defender spacing uniformity
   
7. ROUTE CLASSIFICATION (10 features)
   - GMM-based clustering of input trajectories
   - Soft probabilities for route type
   - Captures "slant", "out", "go" patterns from pre-throw movement

MODEL ARCHITECTURE
------------------

    Input Sequence (63 features x T frames)
                    |
                    v
    ┌─────────────────────────────────┐
    │  Bidirectional LSTM Encoder     │
    │  (2 layers, 128 hidden units)   │
    └─────────────────────────────────┘
                    |
                    v
    ┌─────────────────────────────────┐
    │  Attention Mechanism            │
    │  (learns which frames matter)   │
    └─────────────────────────────────┘
                    |
         ┌─────────┴─────────┐
         v                   v
    [Role Embed]    [Field/Coverage Context]    [Duration Context]
         |                   |                        |
         └─────────┬─────────┴────────────────────────┘
                   v
    ┌─────────────────────────────────┐
    │  Autoregressive LSTM Decoder    │
    │  - Predicts displacement Δ(x,y) │
    │  - Feeds prediction back        │
    │  - Teacher forcing during train │
    └─────────────────────────────────┘
                   |
                   v
    ┌─────────────────────────────────┐
    │  Physics Post-Processing        │
    │  - Max speed: 12 yd/s           │
    │  - Field boundary clipping      │
    │  - Role-based physics blending  │
    └─────────────────────────────────┘

TRAINING DETAILS
----------------
- Data: Weeks 1-16 for training, Weeks 17-18 for validation
- Optimizer: AdamW with weight decay 1e-5
- Learning rate: 0.001 with ReduceLROnPlateau
- Teacher forcing: Starts at 50%, decays to 0%
- Early stopping: Patience of 12 epochs
- Batch size: 64
- Gradient clipping: Max norm 1.0

RESULTS
-------
| Model Version              | Val RMSE | Notes                          |
|---------------------------|----------|--------------------------------|
| Baseline (physics only)    | ~1.79    | Simple ball interpolation      |
| Field-aware LSTM           | 1.147    | Added field context            |
| + Defender interactions    | 1.061    | Defenders track receivers      |
| + Coverage detection       | 1.048    | Zone vs man matters            |
| + Duration conditioning    | 1.031    | Best model                     |

By Role:
- Receiver RMSE: ~0.66 yards (most predictable - follows routes)
- Defender RMSE: ~1.15 yards (reactive, harder to predict)
- Passer RMSE: ~0.42 yards (mostly stationary post-throw)

LESSONS LEARNED
---------------
1. Frame-weighted loss (penalizing later frames more) HURT performance
   - Hypothesis: Over-focusing on hard cases degraded easy cases
   
2. Acceleration features (ax, ay) were computed but not used initially
   - Simple oversight that cost ~1% improvement
   
3. Receiver awareness of defenders is crucial
   - Adding receiver->defender features improved predictions
   
4. Physics constraints help but shouldn't override the model
   - Blend LSTM output with physics, don't hard-constrain

================================================================================
TECHNICAL DETAILS
================================================================================

Features (50 total):
- 8 motion (x, y, vx, vy, s, a, ax, ay)
- 5 ball-relative (dist, angle, velocity toward ball, landing position)
- 11 field position (yards to goal, red zone, hash marks, sidelines)
- 11 motion (position, velocity, acceleration, temporal position, orientation)
- 5 ball-relative (distance, angle, velocity toward ball)
- 11 field position (yards to goal, zones, hash marks, sidelines)
- 17 defender->receiver (dist/angle/velocity, positioning, motion awareness, coverage role, scheme)
- 4 receiver->defender (nearest defender dist/angle, count within 5yds, closing speed)
- 5 coverage scheme (zone/man prob, press, cushion, spacing)
- 10 route cluster probabilities
= 63 total input features

Per-Defender Coverage Features:
- is_likely_man_coverage: Is THIS defender in man coverage? (not just play-level)
- defender_depth_zone: What zone is this defender responsible for? (short/mid/deep)
- has_deep_help: Is there a safety behind this defender?
- inside_leverage: Is defender positioned inside or outside the receiver?

Scheme Context Features:
- single_high_safety: Cover 1/3 indicator (affects corner aggressiveness)
- receiver_route_vertical: Is receiver running vertical? (go/post/corner)
- receiver_route_lateral: Is receiver running lateral? (cross/out/in)

Submission Format:
- Uses Kaggle NFL Inference Server (real-time predictions)
- Model trains on first run, then serves predictions via inference_server.serve()

Version History:
    v1.0 - Initial combined model
    v2.0 - Coverage detection, physics post-processing
    v2.4 - Receiver-defender features, acceleration features
    v3.0 - Kaggle inference server format
    v3.1 - Defender motion awareness (receivers' speed/direction/accel visible to defenders)
    v3.2 - Temporal position (normalized_frame), orientation features (facing_ball, orientation_diff)
    v3.3 - Bug fix: receiver-defender features now use frame-specific positions (not just last frame)
         - Decoder initialized from encoder context (not zeros)
         - Improved physics blending weights (more LSTM trust for defenders)
    v3.4 - Per-defender coverage role features (man vs zone, depth zone, deep help, leverage)
    v3.5 - Scheme context features (single-high safety, receiver route direction)
         - NOW: 63 input features (was 56)
    v3.6 - Reduced inference logging (every 100 batches instead of every batch)
         - Added batch counter and completion message
    v3.7 - QUICK WIN 1: Relative/displacement targets (position relative to ball landing)
         - QUICK WIN 2: Displacement targets (predict change from start position)
         - QUICK WIN 3: Ensemble of 3 models with averaged predictions
         - Target normalization makes different plays comparable
"""

# Kaggle standard imports
import numpy as np
import pandas as pd
import os

# List input files (Kaggle standard)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')


def timestamp():
    """Return current timestamp for logging."""
    return datetime.now().strftime("%H:%M:%S")


# =============================================================================
# GPU CONFIGURATION
# =============================================================================

print(f"[{timestamp()}] " + "=" * 60)
print(f"[{timestamp()}] GPU CONFIGURATION")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"[{timestamp()}] CUDA available: YES")
    print(f"[{timestamp()}] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[{timestamp()}] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print(f"[{timestamp()}] CUDA available: NO - using CPU")

print(f"[{timestamp()}] " + "=" * 60)


# =============================================================================
# KAGGLE PATHS
# =============================================================================

DATA_DIR = Path('/kaggle/input/nfl-big-data-bowl-2026-prediction')
TRAIN_DIR = DATA_DIR / 'train'
OUTPUT_DIR = Path('/kaggle/working')

print(f"[{timestamp()}] Data directory: {DATA_DIR}")


# =============================================================================
# CONFIGURATION - Same as field_aware (proven to work)
# =============================================================================

CONFIG = {
    # Competition provides weeks 1-18 of 2023
    'train_weeks': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 16 weeks
    'val_weeks': [17, 18],  # 2 weeks for validation
    
    'n_route_clusters': 10,
    
    'hidden_dim': 128,  # Same as field_aware
    'num_layers': 2,
    'dropout': 0.3,
    'max_output_frames': 94,
    
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'num_epochs': 40,
    'patience': 12,
    
    # Teacher forcing settings (reverted to original - 70%->10% hurt performance)
    'teacher_forcing_ratio': 0.5,  # Original setting
    'teacher_forcing_min': 0.0,    # Decay to 0% (original behavior)
    
    # Frame-weighted loss settings (DISABLED - hurt performance in testing)
    'use_frame_weighted_loss': False,  # Was True, caused regression from 1.031 to 1.065
    'frame_weight_max': 2.0,  # Later frames weighted up to 2x (not used when disabled)
    
    # QUICK WIN 1: Relative targets (predict position relative to ball landing)
    # This normalizes the prediction space and should reduce variance
    'use_relative_targets': True,
    
    # QUICK WIN 2: Displacement targets (predict cumulative displacement from start)
    # Combined with relative targets: we predict (pos - start_pos) normalized by ball_landing
    'use_displacement_targets': True,
    
    # QUICK WIN 3: Ensemble (train multiple models and average predictions)
    'n_ensemble': 3,  # Number of models in ensemble
    
    'device': device
}

print(f"[{timestamp()}] Device: {CONFIG['device']}")
print(f"[{timestamp()}] Quick Wins Enabled:")
print(f"[{timestamp()}]   - Displacement targets: {CONFIG.get('use_displacement_targets', False)}")
print(f"[{timestamp()}]   - Relative normalization: {CONFIG.get('use_relative_targets', False)}")
print(f"[{timestamp()}]   - Ensemble size: {CONFIG.get('n_ensemble', 1)}")
DT = 0.1
FIELD_WIDTH = 53.3
FIELD_LENGTH = 100
MAX_SPEED = 12.0  # yards per second (elite NFL speed)
FIELD_X_MIN = 0.0
FIELD_X_MAX = 120.0


def apply_physics_constraints(trajectory, last_pos, last_vel, role):
    """
    Post-process predictions to ensure physical plausibility.
    - Clamp to field boundaries
    - Limit maximum speed between frames
    - Smooth unrealistic jumps
    """
    result = trajectory.copy()
    n_frames = len(result)
    
    prev_pos = last_pos.copy()
    
    for i in range(n_frames):
        # Current predicted position
        curr_pos = result[i]
        
        # Calculate implied velocity
        displacement = curr_pos - prev_pos
        distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
        speed = distance / DT
        
        # Clamp speed if too fast
        if speed > MAX_SPEED:
            scale = MAX_SPEED / speed
            displacement = displacement * scale
            curr_pos = prev_pos + displacement
        
        # Clamp to field boundaries
        curr_pos[0] = np.clip(curr_pos[0], FIELD_X_MIN, FIELD_X_MAX)
        curr_pos[1] = np.clip(curr_pos[1], 0, FIELD_WIDTH)
        
        result[i] = curr_pos
        prev_pos = curr_pos.copy()
    
    return result


def blend_with_physics(lstm_pred, physics_pred, role, n_frames):
    """
    Blend LSTM predictions with physics baseline based on role.
    - Receivers: Trust LSTM heavily (following designed routes)
    - Defenders: Trust LSTM more now that we have defender-specific features
    - Passers: Heavy physics (mostly stationary after throw)
    """
    if role == 'Targeted Receiver':
        # Trust LSTM for receivers - they run designed routes
        lstm_weight = 0.95
        decay_rate = 0.2  # Slower decay for receivers
        floor = 0.5
    elif role == 'Passer':
        # Passers mostly stay put after throw
        lstm_weight = 0.3
        decay_rate = 0.5  # Faster decay
        floor = 0.1
    else:  # Defender
        # Increased from 0.7 to 0.85 - we now have receiver motion awareness
        lstm_weight = 0.85
        decay_rate = 0.25
        floor = 0.4
    
    # Blend more toward physics for later frames (uncertainty grows)
    blended = np.zeros_like(lstm_pred)
    for i in range(n_frames):
        # Decay LSTM confidence over time
        frame_weight = lstm_weight * (1 - decay_rate * i / n_frames)
        frame_weight = max(floor, frame_weight)
        blended[i] = frame_weight * lstm_pred[i] + (1 - frame_weight) * physics_pred[i]
    
    return blended


def compute_physics_baseline(last_pos, last_vel, ball_land, role, n_frames):
    """Compute physics-based prediction as baseline."""
    trajectory = np.zeros((n_frames, 2))
    
    for i in range(n_frames):
        t = (i + 1) * DT
        
        if role == 'Targeted Receiver':
            # Smoothstep interpolation toward ball
            alpha = min(1.0, t / (n_frames * DT))
            alpha = 3 * alpha**2 - 2 * alpha**3
            trajectory[i, 0] = last_pos[0] + alpha * (ball_land[0] - last_pos[0])
            trajectory[i, 1] = last_pos[1] + alpha * (ball_land[1] - last_pos[1])
            
        elif role == 'Passer':
            # Minimal movement post-throw
            trajectory[i, 0] = last_pos[0] + last_vel[0] * t * 0.2
            trajectory[i, 1] = last_pos[1] + last_vel[1] * t * 0.2
            
        else:  # Defender
            # Blend momentum with pursuit
            physics_x = last_pos[0] + last_vel[0] * t
            physics_y = last_pos[1] + last_vel[1] * t
            
            pursuit_factor = min(0.5, t / 2)
            pursuit_x = last_pos[0] + pursuit_factor * (ball_land[0] - last_pos[0])
            pursuit_y = last_pos[1] + pursuit_factor * (ball_land[1] - last_pos[1])
            
            blend = min(0.4, t / 3)
            trajectory[i, 0] = (1 - blend) * physics_x + blend * pursuit_x
            trajectory[i, 1] = (1 - blend) * physics_y + blend * pursuit_y
    
    return trajectory


# =============================================================================
# DATA LOADING
# =============================================================================

class NFLDataLoader:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'train'
        
    def load_weeks(self, weeks: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f"[{timestamp()}] Loading weeks: {weeks}")
        all_input, all_output = [], []
        for week in weeks:
            input_file = self.train_dir / f'input_2023_w{week:02d}.csv'
            output_file = self.train_dir / f'output_2023_w{week:02d}.csv'
            all_input.append(pd.read_csv(input_file))
            all_output.append(pd.read_csv(output_file))
            print(f"[{timestamp()}]   Week {week}: {len(all_input[-1]):,} rows")
        return pd.concat(all_input, ignore_index=True), pd.concat(all_output, ignore_index=True)


# =============================================================================
# ROUTE STEM CLASSIFIER (from route_stem model)
# =============================================================================

class RouteStemClassifier:
    """Classifies routes based on INPUT trajectory (the route stem)."""
    
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.gmm = None
        self.scaler = StandardScaler()
    
    def extract_stem_features(self, trajectory: np.ndarray) -> np.ndarray:
        """Extract features from route stem."""
        if len(trajectory) < 3:
            return None
        
        start = trajectory[0]
        end = trajectory[-1]
        mid = trajectory[len(trajectory)//2]
        
        # Total displacement
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Path length vs direct distance
        total_dist = sum(np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + 
                                  (trajectory[i][1] - trajectory[i-1][1])**2)
                        for i in range(1, len(trajectory)))
        direct_dist = np.sqrt(dx**2 + dy**2)
        efficiency = direct_dist / (total_dist + 1e-6)
        
        # Direction
        angle = np.arctan2(dy, dx)
        
        # Midpoint deviation (indicates break)
        mid_dx = mid[0] - (start[0] + end[0])/2
        mid_dy = mid[1] - (start[1] + end[1])/2
        
        # Speed change (if velocities available)
        return np.array([dx, dy, total_dist, direct_dist, efficiency, 
                        angle, mid_dx, mid_dy])
    
    def fit(self, trajectories: List[np.ndarray]):
        """Fit GMM on route stems."""
        features = []
        for traj in trajectories:
            feat = self.extract_stem_features(traj)
            if feat is not None:
                features.append(feat)
        
        if len(features) < 100:
            print(f"[{timestamp()}]   Warning: Only {len(features)} valid stems for fitting")
            return self
        
        features = np.array(features)
        features_scaled = self.scaler.fit_transform(features)
        
        self.gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        self.gmm.fit(features_scaled)
        
        print(f"[{timestamp()}]   Fitted route classifier on {len(features)} stems")
        return self
    
    def predict_proba(self, trajectory: np.ndarray) -> np.ndarray:
        """Get route cluster probabilities."""
        if self.gmm is None:
            return np.ones(self.n_clusters) / self.n_clusters
        
        feat = self.extract_stem_features(trajectory)
        if feat is None:
            return np.ones(self.n_clusters) / self.n_clusters
        
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        return self.gmm.predict_proba(feat_scaled)[0]


# =============================================================================
# FEATURE ENGINEERING - Combined from all models
# =============================================================================

def add_base_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add motion, ball, and field features (from field_aware)."""
    if verbose:
        print(f"[{timestamp()}]   Adding base features...")
    start_time = time.time()
    df = df.copy()
    
    # Motion
    df['dir_rad'] = np.deg2rad(df['dir'])
    df['vx'] = df['s'] * np.cos(df['dir_rad'])
    df['vy'] = df['s'] * np.sin(df['dir_rad'])
    df['ax'] = df['a'] * np.cos(df['dir_rad'])
    df['ay'] = df['a'] * np.sin(df['dir_rad'])
    
    # Temporal position in play - CRITICAL for knowing "where are we in the play"
    # 0 = just after snap, 1 = end of play
    df['normalized_frame'] = df['frame_id'] / df['num_frames_output'].clip(lower=1)
    df['normalized_frame'] = df['normalized_frame'].clip(0, 1)
    
    # Orientation (which way player is facing - different from movement direction!)
    df['o_rad'] = np.deg2rad(df['o'])
    
    # Orientation vs movement direction - CRITICAL for backpedaling, looking for ball, etc.
    # sin captures signed difference, range -1 to 1
    df['orientation_diff'] = np.sin(df['o_rad'] - df['dir_rad'])
    
    # Ball-relative
    df['ball_dx'] = df['ball_land_x'] - df['x']
    df['ball_dy'] = df['ball_land_y'] - df['y']
    df['dist_to_ball'] = np.sqrt(df['ball_dx']**2 + df['ball_dy']**2)
    df['angle_to_ball'] = np.arctan2(df['ball_dy'], df['ball_dx'])
    df['v_toward_ball'] = (df['vx'] * df['ball_dx'] + df['vy'] * df['ball_dy']) / (df['dist_to_ball'] + 1e-6)
    
    # Is player facing toward the ball? (Critical for receivers looking back, defenders breaking on ball)
    # 1 = directly facing ball, -1 = facing away
    df['facing_ball'] = np.cos(df['o_rad'] - df['angle_to_ball'])
    
    # Field position
    df['yards_to_goal'] = df['absolute_yardline_number']
    df['yards_to_goal_norm'] = df['yards_to_goal'] / 100.0
    df['red_zone'] = (df['yards_to_goal'] <= 20).astype(float)
    df['goal_line'] = (df['yards_to_goal'] <= 5).astype(float)
    df['own_territory_deep'] = (df['yards_to_goal'] >= 80).astype(float)
    df['midfield'] = ((df['yards_to_goal'] >= 40) & (df['yards_to_goal'] <= 60)).astype(float)
    
    # Hash marks
    df['hash_left'] = (df['y'] < 22).astype(float)
    df['hash_right'] = (df['y'] > 31.3).astype(float)
    df['hash_center'] = ((df['y'] >= 22) & (df['y'] <= 31.3)).astype(float)
    
    # Sidelines
    df['near_left_sideline'] = (df['y'] < 8).astype(float)
    df['near_right_sideline'] = (df['y'] > 45.3).astype(float)
    df['sideline_constrained'] = (df['near_left_sideline'] + df['near_right_sideline'])
    df['ball_near_sideline'] = ((df['ball_land_y'] < 10) | (df['ball_land_y'] > 43.3)).astype(float)
    
    # Space to sidelines
    df['space_to_nearest_sideline'] = np.minimum(df['y'], FIELD_WIDTH - df['y'])
    df['space_to_nearest_sideline_norm'] = df['space_to_nearest_sideline'] / (FIELD_WIDTH / 2)
    
    # Ball landing normalized
    df['ball_land_x_norm'] = df['ball_land_x'] / 120.0
    df['ball_land_y_norm'] = df['ball_land_y'] / FIELD_WIDTH
    
    if verbose:
        print(f"[{timestamp()}]   Base features done in {time.time()-start_time:.1f}s")
    return df


def add_coverage_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add coverage scheme detection features.
    Zone vs Man coverage fundamentally changes defender behavior.
    """
    if verbose:
        print(f"[{timestamp()}]   Detecting coverage schemes...")
    start_time = time.time()
    
    # Initialize columns
    df['coverage_zone_prob'] = 0.5
    df['coverage_man_prob'] = 0.5
    df['is_pressed_coverage'] = 0.0
    df['cushion_yards'] = 0.5
    df['defender_spacing_score'] = 0.5
    
    play_count = 0
    total_plays = df.groupby(['game_id', 'play_id']).ngroups
    
    for (game_id, play_id), play_df in df.groupby(['game_id', 'play_id']):
        play_count += 1
        if verbose and play_count % 2000 == 0:
            print(f"[{timestamp()}]     Coverage detection: {play_count}/{total_plays}")
        
        # Get last frame (at ball release)
        last_frame = play_df['frame_id'].max()
        snap_df = play_df[play_df['frame_id'] == last_frame]
        
        # Get defenders and receiver
        defenders = snap_df[snap_df['player_role'] == 'Defensive Coverage']
        receiver = snap_df[snap_df['player_role'] == 'Targeted Receiver']
        
        if len(defenders) == 0 or len(receiver) == 0:
            continue
        
        rec_x, rec_y = receiver.iloc[0]['x'], receiver.iloc[0]['y']
        
        # 1. Defender spacing analysis (zone = evenly spaced)
        if len(defenders) >= 2:
            def_positions = defenders[['x', 'y']].values
            pairwise_dists = []
            for i in range(len(def_positions)):
                for j in range(i+1, len(def_positions)):
                    dist = np.sqrt((def_positions[i,0] - def_positions[j,0])**2 + 
                                   (def_positions[i,1] - def_positions[j,1])**2)
                    pairwise_dists.append(dist)
            
            if pairwise_dists:
                avg_spacing = np.mean(pairwise_dists)
                spacing_std = np.std(pairwise_dists)
                spacing_uniformity = 1 - (spacing_std / (avg_spacing + 1e-6))
                spacing_uniformity = np.clip(spacing_uniformity, 0, 1)
            else:
                spacing_uniformity = 0.5
        else:
            spacing_uniformity = 0.5
        
        # 2. Defender depth spread (zone = multiple levels)
        def_depths = defenders['x'].values
        depth_spread = np.std(def_depths) if len(def_depths) > 1 else 0
        depth_score = min(1.0, depth_spread / 10)
        
        # 3. Press coverage detection
        def_dists_to_rec = np.sqrt((defenders['x'] - rec_x)**2 + 
                                    (defenders['y'] - rec_y)**2)
        min_dist = def_dists_to_rec.min()
        is_pressed = 1.0 if min_dist < 5 else 0.0
        cushion = min(min_dist, 15) / 15  # Normalize to 0-1
        
        # 4. Calculate coverage probabilities
        zone_score = 0.4 * spacing_uniformity + 0.4 * depth_score + 0.2 * (1 - is_pressed)
        man_score = 1 - zone_score
        
        # Update all rows for this play
        mask = (df['game_id'] == game_id) & (df['play_id'] == play_id)
        df.loc[mask, 'coverage_zone_prob'] = zone_score
        df.loc[mask, 'coverage_man_prob'] = man_score
        df.loc[mask, 'is_pressed_coverage'] = is_pressed
        df.loc[mask, 'cushion_yards'] = cushion
        df.loc[mask, 'defender_spacing_score'] = spacing_uniformity
    
    if verbose:
        print(f"[{timestamp()}]   Coverage detection done in {time.time()-start_time:.1f}s")
    return df


def add_defender_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add defender interaction features for BOTH defenders AND receivers."""
    if verbose:
        print(f"[{timestamp()}]   Computing defender interaction features...")
    start_time = time.time()
    
    # Initialize columns for DEFENDERS (defender -> receiver)
    df['dist_to_receiver'] = 0.0
    df['angle_to_receiver'] = 0.0
    df['v_toward_receiver'] = 0.0
    df['between_receiver_ball'] = 0.0
    df['is_closest_to_ball'] = 0.0
    df['defender_rank'] = 0.0
    df['leverage'] = 0.0
    
    # NEW: Receiver motion awareness for defenders (so defenders can react)
    df['receiver_speed'] = 0.0           # How fast is the receiver moving?
    df['receiver_direction_rel'] = 0.0   # Receiver direction relative to defender (-1=away, 1=toward)
    df['receiver_accel'] = 0.0           # Is receiver accelerating/decelerating?
    
    # NEW: Per-defender coverage role features (not just play-level!)
    df['is_likely_man_coverage'] = 0.0   # Is THIS defender likely in man coverage?
    df['defender_depth_zone'] = 0.5      # 0=short, 0.5=mid, 1=deep (defender's zone responsibility)
    df['has_deep_help'] = 0.0            # Is there a safety behind this defender?
    df['inside_leverage'] = 0.0          # Is defender inside or outside the receiver? (-1=outside, 1=inside)
    
    # NEW: Safety shell and route context for defenders
    df['single_high_safety'] = 0.0       # Is there only one deep safety? (Cover 1/3 indicator)
    df['receiver_route_vertical'] = 0.0  # Is receiver running a vertical route? (go/post/corner)
    df['receiver_route_lateral'] = 0.0   # Is receiver running a lateral route? (cross/out/in)
    
    # Initialize columns for RECEIVERS (receiver -> nearest defender)
    df['nearest_defender_dist'] = 10.0  # Default: 10 yards (open)
    df['nearest_defender_angle'] = 0.0
    df['defenders_within_5yds'] = 0.0
    df['defender_closing_speed'] = 0.0  # Relative velocity of nearest defender
    
    # Group by play and compute
    play_count = 0
    total_plays = df.groupby(['game_id', 'play_id']).ngroups
    
    for (game_id, play_id), play_df in df.groupby(['game_id', 'play_id']):
        play_count += 1
        if play_count % 500 == 0:
            elapsed = time.time() - start_time
            rate = play_count / elapsed
            remaining = (total_plays - play_count) / rate / 60
            if verbose:
                print(f"[{timestamp()}]     Processing play {play_count}/{total_plays} ({rate:.1f}/sec, ~{remaining:.0f}min left)")
        
        # Get last frame for this play
        last_frame_id = play_df['frame_id'].max()
        last_frame = play_df[play_df['frame_id'] == last_frame_id]
        
        # Find receiver
        receiver = last_frame[last_frame['player_role'] == 'Targeted Receiver']
        if len(receiver) == 0:
            continue
        
        rec_x, rec_y = receiver.iloc[0]['x'], receiver.iloc[0]['y']
        rec_vx = receiver.iloc[0].get('vx', 0)
        rec_vy = receiver.iloc[0].get('vy', 0)
        rec_speed = np.sqrt(rec_vx**2 + rec_vy**2)
        rec_accel = receiver.iloc[0].get('a', 0)
        ball_x, ball_y = receiver.iloc[0]['ball_land_x'], receiver.iloc[0]['ball_land_y']
        
        # Find defenders
        defenders = last_frame[last_frame['player_role'] == 'Defensive Coverage']
        if len(defenders) == 0:
            continue
        
        # === NEW: Safety shell detection (single-high vs two-high) ===
        # Deep defenders = those 15+ yards beyond ball landing
        deep_defenders = defenders[defenders['x'] > ball_x + 10]
        
        # Check if single high: one deep safety vs two
        # Single high = Cover 1, Cover 3 (more aggressive corners)
        # Two high = Cover 2, Cover 4 (safer, more zone)
        single_high = 1.0 if len(deep_defenders) == 1 else 0.0
        
        # === NEW: Receiver route direction classification ===
        # Look at receiver trajectory to determine route type
        rec_frames = play_df[(play_df['nfl_id'] == receiver.iloc[0]['nfl_id'])]
        if len(rec_frames) > 1:
            first_frame = rec_frames[rec_frames['frame_id'] == rec_frames['frame_id'].min()].iloc[0]
            
            # Route displacement
            route_dx = rec_x - first_frame['x']  # Vertical (downfield)
            route_dy = rec_y - first_frame['y']  # Lateral (sideline to sideline)
            
            # Classify route direction
            # Vertical: significant downfield movement (>10 yards)
            is_vertical = 1.0 if route_dx > 10 else 0.0
            
            # Lateral: significant sideline movement (>5 yards) with less vertical
            is_lateral = 1.0 if abs(route_dy) > 5 and route_dx < 15 else 0.0
        else:
            is_vertical = 0.0
            is_lateral = 0.0
        
        # Compute defender distances to ball for ranking
        def_dists_to_ball = np.sqrt((defenders['x'] - ball_x)**2 + (defenders['y'] - ball_y)**2)
        defender_ranks = def_dists_to_ball.rank()
        
        # Update features for all frames of each defender
        for idx, def_row in defenders.iterrows():
            nfl_id = def_row['nfl_id']
            mask = (df['game_id'] == game_id) & (df['play_id'] == play_id) & (df['nfl_id'] == nfl_id)
            
            for frame_idx in df[mask].index:
                def_x, def_y = df.loc[frame_idx, 'x'], df.loc[frame_idx, 'y']
                def_vx, def_vy = df.loc[frame_idx, 'vx'], df.loc[frame_idx, 'vy']
                
                # Distance to receiver
                dx, dy = rec_x - def_x, rec_y - def_y
                dist = np.sqrt(dx**2 + dy**2)
                df.loc[frame_idx, 'dist_to_receiver'] = dist / 20.0  # Normalize
                
                # Angle to receiver
                df.loc[frame_idx, 'angle_to_receiver'] = np.arctan2(dy, dx)
                
                # Velocity toward receiver
                if dist > 0:
                    df.loc[frame_idx, 'v_toward_receiver'] = (def_vx * dx + def_vy * dy) / dist
                
                # Between receiver and ball?
                rec_to_ball = np.array([ball_x - rec_x, ball_y - rec_y])
                rec_to_def = np.array([def_x - rec_x, def_y - rec_y])
                
                len_sq = np.dot(rec_to_ball, rec_to_ball)
                if len_sq > 0:
                    t = max(0, min(1, np.dot(rec_to_def, rec_to_ball) / len_sq))
                    proj = np.array([rec_x, rec_y]) + t * rec_to_ball
                    dist_to_line = np.sqrt((def_x - proj[0])**2 + (def_y - proj[1])**2)
                    df.loc[frame_idx, 'between_receiver_ball'] = max(0, 1 - dist_to_line / 10)
                
                # Defender rank (1 = closest to ball)
                df.loc[frame_idx, 'defender_rank'] = defender_ranks[idx] / len(defenders)
                
                # Is closest to ball?
                if defender_ranks[idx] == 1:
                    df.loc[frame_idx, 'is_closest_to_ball'] = 1.0
                
                # Leverage (angle advantage)
                def_to_ball = np.array([ball_x - def_x, ball_y - def_y])
                rec_to_ball_vec = np.array([ball_x - rec_x, ball_y - rec_y])
                def_to_ball_norm = def_to_ball / (np.linalg.norm(def_to_ball) + 1e-6)
                rec_to_ball_norm = rec_to_ball_vec / (np.linalg.norm(rec_to_ball_vec) + 1e-6)
                leverage = np.dot(def_to_ball_norm, rec_to_ball_norm)
                df.loc[frame_idx, 'leverage'] = leverage
                
                # NEW: Receiver motion awareness for defenders
                # Receiver speed (normalized 0-1, max ~10 yds/sec)
                df.loc[frame_idx, 'receiver_speed'] = min(rec_speed / 10.0, 1.0)
                
                # Receiver direction relative to defender (-1=away, 1=toward)
                # Project receiver velocity onto defender-to-receiver direction
                if dist > 0:
                    # Negative because we want "receiver moving toward defender" to be positive
                    rec_dir_rel = -(rec_vx * dx + rec_vy * dy) / (dist * (rec_speed + 0.1))
                    df.loc[frame_idx, 'receiver_direction_rel'] = np.clip(rec_dir_rel, -1, 1)
                
                # Receiver acceleration (normalized -1 to 1)
                df.loc[frame_idx, 'receiver_accel'] = np.clip(rec_accel / 5.0, -1, 1)
                
                # === NEW: Per-defender coverage role features ===
                
                # 1. Is this defender likely in man coverage?
                # Heuristic: Close to receiver AND moving toward them = likely man
                man_dist_score = max(0, 1 - dist / 15)  # Closer = higher score
                if dist > 0:
                    v_toward = (def_vx * dx + def_vy * dy) / dist
                    man_pursuit_score = max(0, v_toward / 5)  # Moving toward = higher
                else:
                    man_pursuit_score = 0
                is_man = 0.6 * man_dist_score + 0.4 * man_pursuit_score
                df.loc[frame_idx, 'is_likely_man_coverage'] = np.clip(is_man, 0, 1)
                
                # 2. Defender depth zone (relative to LOS/ball position)
                # Use x-position relative to ball landing spot to estimate zone depth
                depth_from_ball = ball_x - def_x  # Positive = defender behind ball
                if depth_from_ball < 7:
                    depth_zone = 0.0  # Short zone (flat, curl)
                elif depth_from_ball < 15:
                    depth_zone = 0.5  # Intermediate zone (hook, seam)
                else:
                    depth_zone = 1.0  # Deep zone (deep third, safety)
                df.loc[frame_idx, 'defender_depth_zone'] = depth_zone
                
                # 3. Has deep help? Check if there's a safety behind this defender
                has_help = 0.0
                for _, other_def in defenders.iterrows():
                    if other_def['nfl_id'] == nfl_id:
                        continue
                    other_x = other_def['x']
                    # If another defender is deeper (larger x toward endzone)
                    if other_x > def_x + 10:  # At least 10 yards behind
                        # And reasonably close horizontally (within 15 yards)
                        if abs(other_def['y'] - def_y) < 15:
                            has_help = 1.0
                            break
                df.loc[frame_idx, 'has_deep_help'] = has_help
                
                # 4. Inside leverage: Is defender positioned inside or outside the receiver?
                # Inside = between receiver and center of field (y=26.65)
                field_center_y = FIELD_WIDTH / 2
                rec_side = 1 if rec_y > field_center_y else -1  # 1=right side, -1=left side
                def_relative_to_rec = def_y - rec_y
                # If defender is between receiver and center = inside leverage
                inside_leverage = -rec_side * np.sign(def_relative_to_rec) * min(1, abs(def_relative_to_rec) / 5)
                df.loc[frame_idx, 'inside_leverage'] = np.clip(inside_leverage, -1, 1)
                
                # 5. Safety shell indicator (same for all defenders on play)
                df.loc[frame_idx, 'single_high_safety'] = single_high
                
                # 6. Receiver route direction (what type of route is defender reacting to?)
                df.loc[frame_idx, 'receiver_route_vertical'] = is_vertical
                df.loc[frame_idx, 'receiver_route_lateral'] = is_lateral
        
        # === RECEIVER-RELATIVE FEATURES ===
        # Compute how the receiver sees the defenders at EACH FRAME (not just last frame)
        receiver_nfl_id = receiver.iloc[0]['nfl_id']
        defender_nfl_ids = defenders['nfl_id'].unique()
        rec_mask = (df['game_id'] == game_id) & (df['play_id'] == play_id) & (df['nfl_id'] == receiver_nfl_id)
        
        for frame_idx in df[rec_mask].index:
            rec_frame_id = df.loc[frame_idx, 'frame_id']
            rec_x_frame = df.loc[frame_idx, 'x']
            rec_y_frame = df.loc[frame_idx, 'y']
            rec_vx = df.loc[frame_idx, 'vx']
            rec_vy = df.loc[frame_idx, 'vy']
            
            # Get defender positions at THIS SPECIFIC FRAME (not last frame!)
            min_dist = 99.0
            nearest_angle = 0.0
            closing_speed = 0.0
            count_within_5 = 0
            
            for def_nfl_id in defender_nfl_ids:
                # Get this defender's position at the same frame
                def_at_frame = play_df[(play_df['nfl_id'] == def_nfl_id) & 
                                       (play_df['frame_id'] == rec_frame_id)]
                if len(def_at_frame) == 0:
                    continue
                    
                def_row = def_at_frame.iloc[0]
                def_x, def_y = def_row['x'], def_row['y']
                def_vx = def_row.get('vx', 0) if 'vx' in def_row else 0
                def_vy = def_row.get('vy', 0) if 'vy' in def_row else 0
                
                dx = def_x - rec_x_frame
                dy = def_y - rec_y_frame
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < 5:
                    count_within_5 += 1
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_angle = np.arctan2(dy, dx)
                    # Closing speed: relative velocity toward receiver
                    if dist > 0:
                        closing_speed = ((def_vx - rec_vx) * (-dx) + (def_vy - rec_vy) * (-dy)) / dist
            
            df.loc[frame_idx, 'nearest_defender_dist'] = min(min_dist, 20) / 20  # Normalize to 0-1
            df.loc[frame_idx, 'nearest_defender_angle'] = nearest_angle
            df.loc[frame_idx, 'defenders_within_5yds'] = min(count_within_5, 4) / 4  # Normalize
            df.loc[frame_idx, 'defender_closing_speed'] = np.clip(closing_speed, -10, 10) / 10  # Normalize
    
    if verbose:
        print(f"[{timestamp()}]   Defender features done in {time.time()-start_time:.1f}s")
    return df


# =============================================================================
# DATASET
# =============================================================================

class CombinedDataset(Dataset):
    """Dataset with ALL features: field + defender + coverage + route."""
    
    # Motion features (including acceleration, temporal position, and orientation)
    MOTION_FEATURES = ['x', 'y', 'vx', 'vy', 's', 'a', 'ax', 'ay', 
                       'normalized_frame', 'orientation_diff', 'facing_ball']
    
    # Ball-relative features
    BALL_FEATURES = ['dist_to_ball', 'angle_to_ball', 'v_toward_ball', 
                     'ball_land_x_norm', 'ball_land_y_norm']
    
    # Field position features
    FIELD_FEATURES = ['yards_to_goal_norm', 'red_zone', 'goal_line', 
                      'own_territory_deep', 'midfield',
                      'hash_left', 'hash_right', 'hash_center',
                      'sideline_constrained', 'ball_near_sideline',
                      'space_to_nearest_sideline_norm']
    
    # Defender interaction features (for defenders)
    # Defender features (defender -> receiver, including receiver motion and coverage role)
    DEFENDER_FEATURES = ['dist_to_receiver', 'angle_to_receiver', 
                         'v_toward_receiver', 'between_receiver_ball',
                         'is_closest_to_ball', 'defender_rank', 'leverage',
                         'receiver_speed', 'receiver_direction_rel', 'receiver_accel',
                         'is_likely_man_coverage', 'defender_depth_zone', 
                         'has_deep_help', 'inside_leverage',
                         'single_high_safety', 'receiver_route_vertical', 'receiver_route_lateral']
    
    # Receiver-relative defender features (for receivers - how they see defenders)
    RECEIVER_DEFENDER_FEATURES = ['nearest_defender_dist', 'nearest_defender_angle',
                                   'defenders_within_5yds', 'defender_closing_speed']
    
    # Coverage scheme features (NEW)
    COVERAGE_FEATURES = ['coverage_zone_prob', 'coverage_man_prob',
                         'is_pressed_coverage', 'cushion_yards', 
                         'defender_spacing_score']
    
    BASE_FEATURES = MOTION_FEATURES + BALL_FEATURES + FIELD_FEATURES + DEFENDER_FEATURES + RECEIVER_DEFENDER_FEATURES + COVERAGE_FEATURES
    
    def __init__(self, df_input: pd.DataFrame, df_output: pd.DataFrame,
                 route_classifier: RouteStemClassifier = None, fit_routes: bool = False):
        self.n_route_clusters = CONFIG['n_route_clusters']
        self.route_classifier = route_classifier
        
        # Add base features
        print(f"[{timestamp()}] Creating dataset...")
        df_input = add_base_features(df_input)
        
        # Add coverage scheme features (zone vs man detection)
        df_input = add_coverage_features(df_input)
        
        # Add defender interaction features
        df_input = add_defender_features(df_input)
        
        # Fit or use route classifier
        if fit_routes:
            print(f"[{timestamp()}]   Fitting route classifier...")
            self.route_classifier = RouteStemClassifier(self.n_route_clusters)
            receiver_trajs = []
            receivers = df_input[df_input['player_role'] == 'Targeted Receiver']
            for (gid, pid, nid), player in receivers.groupby(['game_id', 'play_id', 'nfl_id']):
                traj = player[['x', 'y']].values
                if len(traj) >= 5:
                    receiver_trajs.append(traj)
            if len(receiver_trajs) > 100:
                self.route_classifier.fit(receiver_trajs[:5000])
        
        # Prepare samples
        print(f"[{timestamp()}]   Preparing samples...")
        self.samples = self._prepare_samples(df_input, df_output)
        
        total_features = len(self.BASE_FEATURES) + self.n_route_clusters
        print(f"[{timestamp()}]   Created {len(self.samples):,} samples with {total_features} features")
        print(f"[{timestamp()}]     Base: {len(self.BASE_FEATURES)}")
        print(f"[{timestamp()}]     Route probs: {self.n_route_clusters}")
    
    def _prepare_samples(self, df_input: pd.DataFrame, df_output: pd.DataFrame) -> List[Dict]:
        players = df_input[df_input['player_to_predict'] == True].copy()
        unique_players = players.groupby(['game_id', 'play_id', 'nfl_id']).first().reset_index()
        
        samples = []
        
        for i, (_, player) in enumerate(unique_players.iterrows()):
            if i % 2000 == 0 and i > 0:
                print(f"[{timestamp()}]     Processing {i}/{len(unique_players)}")
            
            game_id, play_id, nfl_id = player['game_id'], player['play_id'], player['nfl_id']
            
            player_input = df_input[
                (df_input['game_id'] == game_id) &
                (df_input['play_id'] == play_id) &
                (df_input['nfl_id'] == nfl_id)
            ].sort_values('frame_id')
            
            player_output = df_output[
                (df_output['game_id'] == game_id) &
                (df_output['play_id'] == play_id) &
                (df_output['nfl_id'] == nfl_id)
            ].sort_values('frame_id')
            
            if len(player_output) == 0:
                continue
            
            last_input = player_input.iloc[-1]
            
            # Base features
            base_features = player_input[self.BASE_FEATURES].values.astype(np.float32)
            base_features = np.nan_to_num(base_features, nan=0.0)
            
            # Route probabilities
            if self.route_classifier is not None:
                input_traj = player_input[['x', 'y']].values
                route_probs = self.route_classifier.predict_proba(input_traj)
            else:
                route_probs = np.ones(self.n_route_clusters) / self.n_route_clusters
            
            # Broadcast route probs to all frames
            route_probs_expanded = np.tile(route_probs, (len(base_features), 1))
            
            # Combine all features
            all_features = np.hstack([base_features, route_probs_expanded]).astype(np.float32)
            
            # Target trajectory - get absolute positions first
            actual_traj = player_output[['x', 'y']].values.astype(np.float32)
            
            # Get reference points for relative targets
            last_pos = np.array([last_input['x'], last_input['y']], dtype=np.float32)
            # Handle missing ball_land with fallback to last position + average throw distance
            ball_land_x = last_input.get('ball_land_x', last_input['x'] + 10.0)
            ball_land_y = last_input.get('ball_land_y', last_input['y'])
            if np.isnan(ball_land_x) or np.isnan(ball_land_y):
                ball_land_x = last_input['x'] + 10.0
                ball_land_y = last_input['y']
            ball_land = np.array([ball_land_x, ball_land_y], dtype=np.float32)
            
            # QUICK WIN 1 & 2: Convert to displacement targets relative to ball landing
            if CONFIG.get('use_displacement_targets', False):
                # Displacement from last known position
                displacement_traj = actual_traj - last_pos
                
                if CONFIG.get('use_relative_targets', False):
                    # Normalize by distance to ball (makes different plays comparable)
                    # Add small epsilon to avoid division by zero
                    ball_dist = np.linalg.norm(ball_land - last_pos) + 1e-6
                    # Scale factor: normalize so typical displacements are ~1
                    scale = max(ball_dist, 5.0)  # Minimum 5 yards to avoid extreme scaling
                    # Guard against NaN from any remaining edge cases
                    if np.isnan(scale) or np.isinf(scale):
                        scale = 10.0  # Default scale
                    displacement_traj = displacement_traj / scale
                    # Clean any NaN values that might have crept in
                    displacement_traj = np.nan_to_num(displacement_traj, nan=0.0)
                    target_scale = scale  # Store for inverse transform
                else:
                    target_scale = 1.0
            else:
                displacement_traj = actual_traj
                target_scale = 1.0
            
            # Role encoding
            role = player['player_role']
            role_encoding = [
                1 if role == 'Passer' else 0,
                1 if role == 'Targeted Receiver' else 0,
                1 if role == 'Defensive Coverage' else 0,
            ]
            
            # Field and coverage context at ball release
            field_context = np.array([
                last_input['yards_to_goal_norm'],
                last_input['red_zone'],
                last_input['goal_line'],
                last_input['sideline_constrained'],
                last_input['coverage_zone_prob'],
                last_input['is_pressed_coverage'],
            ], dtype=np.float32)
            
            # Duration conditioning - normalized play length
            # Adjusted thresholds to match evaluation buckets
            n_output = len(actual_traj)
            duration_context = np.array([
                n_output / CONFIG['max_output_frames'],  # Normalized duration (0-1)
                1.0 if n_output < 20 else 0.0,           # Short play flag (<2s)
                1.0 if n_output >= 40 else 0.0,          # Long play flag (>4s)
            ], dtype=np.float32)
            
            sample = {
                'input_features': all_features,
                'actual_trajectory': displacement_traj if CONFIG.get('use_displacement_targets', False) else actual_traj,
                'absolute_trajectory': actual_traj,  # Keep original for evaluation
                'num_output_frames': n_output,
                'role': np.array(role_encoding, dtype=np.float32),
                'last_position': last_pos,
                'last_velocity': np.array([last_input['vx'], last_input['vy']], dtype=np.float32),
                'ball_landing': ball_land,
                'field_context': field_context,
                'duration_context': duration_context,
                'target_scale': np.array([target_scale], dtype=np.float32),  # For inverse transform
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    max_input_len = max(s['input_features'].shape[0] for s in batch)
    max_output_len = max(s['num_output_frames'] for s in batch)
    n_features = batch[0]['input_features'].shape[1]
    
    input_seqs, input_lengths = [], []
    trajectories, output_lengths = [], []
    absolute_trajectories = []  # Keep absolute for evaluation
    roles, last_positions, last_velocities = [], [], []
    ball_landings, field_contexts, duration_contexts = [], [], []
    target_scales = []  # For inverse transform
    
    for sample in batch:
        seq = sample['input_features']
        seq_len = seq.shape[0]
        if seq_len < max_input_len:
            padding = np.zeros((max_input_len - seq_len, n_features), dtype=np.float32)
            seq = np.vstack([seq, padding])
        input_seqs.append(seq)
        input_lengths.append(seq_len)
        
        traj = sample['actual_trajectory']
        abs_traj = sample.get('absolute_trajectory', traj)  # Fallback if not present
        out_len = sample['num_output_frames']
        if out_len < max_output_len:
            padding = np.zeros((max_output_len - out_len, 2), dtype=np.float32)
            traj = np.vstack([traj, padding])
            abs_traj = np.vstack([abs_traj, padding])
        trajectories.append(traj)
        absolute_trajectories.append(abs_traj)
        output_lengths.append(out_len)
        
        roles.append(sample['role'])
        last_positions.append(sample['last_position'])
        last_velocities.append(sample['last_velocity'])
        ball_landings.append(sample['ball_landing'])
        field_contexts.append(sample['field_context'])
        duration_contexts.append(sample['duration_context'])
        target_scales.append(sample.get('target_scale', np.array([1.0], dtype=np.float32)))
    
    return {
        'input_features': torch.FloatTensor(np.array(input_seqs)),
        'input_lengths': torch.LongTensor(input_lengths),
        'trajectories': torch.FloatTensor(np.array(trajectories)),
        'absolute_trajectories': torch.FloatTensor(np.array(absolute_trajectories)),
        'output_lengths': torch.LongTensor(output_lengths),
        'roles': torch.FloatTensor(np.array(roles)),
        'last_positions': torch.FloatTensor(np.array(last_positions)),
        'last_velocities': torch.FloatTensor(np.array(last_velocities)),
        'ball_landings': torch.FloatTensor(np.array(ball_landings)),
        'field_contexts': torch.FloatTensor(np.array(field_contexts)),
        'duration_contexts': torch.FloatTensor(np.array(duration_contexts)),
        'target_scales': torch.FloatTensor(np.array(target_scales)),
    }


# =============================================================================
# MODEL - EXACT same architecture as field_aware (proven to work!)
# =============================================================================

class CombinedLSTM(nn.Module):
    """
    Combined LSTM with all features + duration conditioning:
    - 11 motion (x, y, vx, vy, s, a, ax, ay, normalized_frame, orientation_diff, facing_ball)
    - 5 ball-relative
    - 11 field position
    - 17 defender->receiver features (motion awareness + per-defender coverage role + scheme context)
    - 4 receiver->defender features (how receivers see defenders)
    - 5 coverage scheme
    = 53 base features + 10 route clusters = 63 total
    - Duration context: 3 features (normalized duration, short flag, long flag)
    """
    
    def __init__(self,
                 input_dim: int = 63,  # 53 base (11 motion + 5 ball + 11 field + 17 defender + 4 receiver_def + 5 coverage) + 10 route
                 hidden_dim: int = 128,
                 role_dim: int = 3,
                 field_dim: int = 6,  # yards_to_goal, red_zone, goal_line, sideline, coverage_zone, is_pressed
                 duration_dim: int = 3,  # normalized duration, short flag, long flag
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 max_output_frames: int = 94):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_output_frames = max_output_frames
        
        # Input encoder
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Role embedding
        self.role_embed = nn.Linear(role_dim, hidden_dim // 4)
        
        # Field context embedding
        self.field_embed = nn.Sequential(
            nn.Linear(field_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Duration conditioning embedding (NEW)
        self.duration_embed = nn.Sequential(
            nn.Linear(duration_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Autoregressive decoder - now includes duration embedding
        decoder_input_dim = hidden_dim * 2 + hidden_dim // 4 + hidden_dim // 4 + hidden_dim // 4 + 2 + 2 + 2
        self.decoder_cell = nn.LSTMCell(decoder_input_dim, hidden_dim)
        
        # Output head - now includes duration context
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4 + hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, input_seq, input_lengths, roles, last_positions, last_velocities, 
                ball_landings, field_contexts, duration_contexts=None, target_trajectory=None, teacher_forcing_ratio=0.0):
        batch_size = input_seq.size(0)
        device = input_seq.device
        
        # Encode input sequence
        x = self.input_proj(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        
        # Attention context
        attn_weights = F.softmax(self.attention(encoded), dim=1)
        context = torch.sum(attn_weights * encoded, dim=1)
        
        # Role embedding
        role_embed = self.role_embed(roles)
        
        # Field context embedding
        field_embed = self.field_embed(field_contexts)
        
        # Duration conditioning embedding (NEW)
        if duration_contexts is not None:
            duration_embed = self.duration_embed(duration_contexts)
        else:
            # Default to medium duration if not provided
            duration_embed = self.duration_embed(torch.tensor([[0.5, 0.0, 0.0]], device=device).expand(batch_size, -1))
        
        # Initialize decoder state from attention context (better than zeros)
        # This transfers encoder knowledge to decoder initialization
        h = context[:, :self.hidden_dim]  # Take first half (forward direction)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        current_pos = last_positions.clone()
        current_vel = last_velocities.clone()
        
        predictions = []
        
        for t in range(self.max_output_frames):
            # Build decoder input - now includes duration embedding
            decoder_input = torch.cat([
                context,
                role_embed,
                field_embed,
                duration_embed,
                current_pos,
                ball_landings,
                current_vel,
            ], dim=1)
            
            h, c = self.decoder_cell(decoder_input, (h, c))
            
            # Output with field AND duration conditioning
            output_input = torch.cat([h, field_embed, duration_embed], dim=1)
            displacement = self.output_head(output_input)
            
            next_pos = current_pos + displacement
            predictions.append(next_pos)
            
            if target_trajectory is not None and t < target_trajectory.size(1):
                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher:
                    current_pos = target_trajectory[:, t, :]
                else:
                    current_pos = next_pos
            else:
                current_pos = next_pos
            
            current_vel = displacement / DT
        
        return torch.stack(predictions, dim=1)


# =============================================================================
# TRAINER - SAME as field_aware
# =============================================================================

class Trainer:
    def __init__(self, model: nn.Module, device: str):
        self.model = model.to(device)
        self.device = device
    
    def loss_fn(self, predictions, targets, output_lengths, last_positions=None, target_scales=None):
        """
        Compute loss between predictions and targets.
        
        If using displacement targets:
        - predictions are in absolute coordinates (model output)
        - targets are in displacement/relative space
        - We convert predictions to displacement space for loss computation
        """
        batch_size = predictions.size(0)
        max_frames = min(predictions.size(1), targets.size(1))
        
        pred = predictions[:, :max_frames, :]
        target = targets[:, :max_frames, :]
        
        # Convert predictions to displacement space if using displacement targets
        if CONFIG.get('use_displacement_targets', False) and last_positions is not None:
            # Convert absolute predictions to displacement from last position
            last_pos = last_positions.unsqueeze(1)  # (B, 1, 2)
            pred_disp = pred - last_pos  # (B, T, 2)
            
            # Apply scaling if using relative targets
            if CONFIG.get('use_relative_targets', False) and target_scales is not None:
                scales = target_scales.unsqueeze(1)  # (B, 1, 1)
                pred_disp = pred_disp / scales
            
            pred = pred_disp
        
        # Basic mask for valid frames
        mask = torch.arange(max_frames).unsqueeze(0).to(self.device) < output_lengths.unsqueeze(1)
        mask = mask.float().unsqueeze(2)
        
        if CONFIG.get('use_frame_weighted_loss', False):
            # Frame-weighted loss: later frames weighted MORE heavily
            # This helps the model focus on reducing error compounding
            frame_weights = torch.zeros(batch_size, max_frames, 1, device=self.device)
            for i in range(batch_size):
                n_frames = output_lengths[i].item()
                if n_frames > 0:
                    # Linear weight from 1.0 to frame_weight_max
                    weights = 1.0 + (CONFIG['frame_weight_max'] - 1.0) * torch.arange(n_frames, device=self.device) / n_frames
                    frame_weights[i, :n_frames, 0] = weights
            
            weighted_mask = mask * frame_weights
            loss = ((pred - target) ** 2 * weighted_mask).sum() / (weighted_mask.sum() * 2 + 1e-6)
        else:
            loss = ((pred - target) ** 2 * mask).sum() / (mask.sum() * 2 + 1e-6)
        
        return loss
    
    def train_epoch(self, train_loader, optimizer, teacher_forcing_ratio):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_feat = batch['input_features'].to(self.device)
            input_lengths = batch['input_lengths']
            targets = batch['trajectories'].to(self.device)
            output_lengths = batch['output_lengths'].to(self.device)
            roles = batch['roles'].to(self.device)
            last_pos = batch['last_positions'].to(self.device)
            last_vel = batch['last_velocities'].to(self.device)
            ball_land = batch['ball_landings'].to(self.device)
            field_ctx = batch['field_contexts'].to(self.device)
            duration_ctx = batch['duration_contexts'].to(self.device)
            target_scales = batch['target_scales'].to(self.device)
            
            # For teacher forcing with displacement targets, use absolute trajectories
            teacher_traj = batch['absolute_trajectories'].to(self.device) if CONFIG.get('use_displacement_targets', False) else targets
            
            predictions = self.model(
                input_feat, input_lengths, roles, last_pos, last_vel, ball_land, field_ctx,
                duration_contexts=duration_ctx, target_trajectory=teacher_traj, teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            loss = self.loss_fn(predictions, targets, output_lengths, last_pos, target_scales)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"[{timestamp()}]   Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}")
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader, return_by_zone=False):
        self.model.eval()
        all_errors = []
        zone_errors = {'red_zone': [], 'midfield': [], 'deep': []}
        role_errors = {'Receiver': [], 'Defender': [], 'Passer': []}
        
        # Duration buckets (in frames, 10 frames = 1 second)
        # Adjusted thresholds: most plays are 2-4 seconds
        duration_errors = {
            'short': [],      # < 20 frames (< 2 sec) - quick plays
            'medium': [],     # 20-40 frames (2-4 sec) - typical plays
            'long': [],       # > 40 frames (> 4 sec) - extended plays
        }
        
        # Frame position errors (early vs late in play)
        # Adjusted: most plays have 20-40 frames total
        frame_errors = {
            'early': [],      # First 10 frames (first second)
            'middle': [],     # Frames 10-30 (middle seconds)
            'late': [],       # Frames 30+ (late in play)
        }
        
        with torch.no_grad():
            for batch in val_loader:
                input_feat = batch['input_features'].to(self.device)
                input_lengths = batch['input_lengths']
                # Use absolute trajectories for evaluation (not displacement targets)
                abs_targets = batch.get('absolute_trajectories', batch['trajectories']).to(self.device)
                output_lengths = batch['output_lengths'].to(self.device)
                roles = batch['roles'].to(self.device)
                last_pos = batch['last_positions'].to(self.device)
                last_vel = batch['last_velocities'].to(self.device)
                ball_land = batch['ball_landings'].to(self.device)
                field_ctx = batch['field_contexts'].to(self.device)
                duration_ctx = batch['duration_contexts'].to(self.device)
                
                predictions = self.model(
                    input_feat, input_lengths, roles, last_pos, last_vel, ball_land, field_ctx,
                    duration_contexts=duration_ctx, target_trajectory=None, teacher_forcing_ratio=0.0
                )
                
                for i in range(predictions.size(0)):
                    length = output_lengths[i].item()
                    pred = predictions[i, :length].cpu().numpy()
                    target = abs_targets[i, :length].cpu().numpy()  # Always compare to absolute
                    errors = np.sqrt(((pred - target) ** 2).sum(axis=1))
                    all_errors.extend(errors)
                    
                    # By role
                    role_vec = roles[i].cpu().numpy()
                    if role_vec[1] > 0.5:
                        role_errors['Receiver'].extend(errors)
                    elif role_vec[2] > 0.5:
                        role_errors['Defender'].extend(errors)
                    else:
                        role_errors['Passer'].extend(errors)
                    
                    # By zone
                    if return_by_zone:
                        ctx = field_ctx[i].cpu().numpy()
                        if ctx[1] > 0.5:
                            zone_errors['red_zone'].extend(errors)
                        elif ctx[0] > 0.6:
                            zone_errors['deep'].extend(errors)
                        else:
                            zone_errors['midfield'].extend(errors)
                    
                    # By play duration (adjusted thresholds)
                    if length < 20:
                        duration_errors['short'].extend(errors)
                    elif length < 40:
                        duration_errors['medium'].extend(errors)
                    else:
                        duration_errors['long'].extend(errors)
                    
                    # By frame position within play (adjusted thresholds)
                    for frame_idx, err in enumerate(errors):
                        if frame_idx < 10:
                            frame_errors['early'].append(err)
                        elif frame_idx < 30:
                            frame_errors['middle'].append(err)
                        else:
                            frame_errors['late'].append(err)
        
        all_errors = np.array(all_errors)
        results = {
            'rmse': np.sqrt(np.mean(all_errors ** 2)),
            'mae': np.mean(all_errors),
            'median': np.median(all_errors),
            'p90': np.percentile(all_errors, 90),
            'p95': np.percentile(all_errors, 95),
        }
        
        for role, errs in role_errors.items():
            if len(errs) > 0:
                results[f'{role.lower()}_rmse'] = np.sqrt(np.mean(np.array(errs) ** 2))
                results[f'{role.lower()}_count'] = len(errs)
        
        if return_by_zone:
            for zone, errs in zone_errors.items():
                if len(errs) > 0:
                    results[f'{zone}_rmse'] = np.sqrt(np.mean(np.array(errs) ** 2))
        
        # Duration metrics
        for dur, errs in duration_errors.items():
            if len(errs) > 0:
                results[f'dur_{dur}_rmse'] = np.sqrt(np.mean(np.array(errs) ** 2))
                results[f'dur_{dur}_count'] = len(errs)
        
        # Frame position metrics (with counts)
        for pos, errs in frame_errors.items():
            if len(errs) > 0:
                results[f'frame_{pos}_rmse'] = np.sqrt(np.mean(np.array(errs) ** 2))
                results[f'frame_{pos}_count'] = len(errs)
        
        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n[{timestamp()}] " + "=" * 80)
    print(f"[{timestamp()}] COMBINED MODEL TRAINING")
    print(f"[{timestamp()}] Field-Aware (1.147) + Defender (1.252) + Route Stem (1.240)")
    print(f"[{timestamp()}] Expected: ~1.061 RMSE")
    print(f"[{timestamp()}] " + "=" * 80)
    
    # Load data
    print(f"\n[{timestamp()}] Loading data...")
    loader = NFLDataLoader()
    
    print(f"[{timestamp()}] Training weeks:")
    df_train_in, df_train_out = loader.load_weeks(CONFIG['train_weeks'])
    
    print(f"[{timestamp()}] Validation weeks:")
    df_val_in, df_val_out = loader.load_weeks(CONFIG['val_weeks'])
    
    # Create datasets
    print(f"\n[{timestamp()}] Creating training dataset...")
    train_dataset = CombinedDataset(df_train_in, df_train_out, fit_routes=True)
    
    print(f"\n[{timestamp()}] Creating validation dataset...")
    val_dataset = CombinedDataset(df_val_in, df_val_out, 
                                   route_classifier=train_dataset.route_classifier)
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    # Create model
    print(f"\n[{timestamp()}] " + "=" * 80)
    print(f"[{timestamp()}] TRAINING")
    print(f"[{timestamp()}] " + "=" * 80)
    
    input_dim = len(CombinedDataset.BASE_FEATURES) + CONFIG['n_route_clusters']
    print(f"[{timestamp()}] Input features: {input_dim}")
    
    model = CombinedLSTM(
        input_dim=input_dim,
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        max_output_frames=CONFIG['max_output_frames']
    )
    
    print(f"[{timestamp()}] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(model, CONFIG['device'])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    
    best_rmse = float('inf')
    patience_counter = 0
    
    # Store route classifier for submission
    route_classifier = train_dataset.route_classifier
    
    for epoch in range(CONFIG['num_epochs']):
        start = time.time()
        
        # Slower teacher forcing decay - helps with long sequences
        tf_min = CONFIG.get('teacher_forcing_min', 0.0)
        tf_decay = (CONFIG['teacher_forcing_ratio'] - tf_min) * (1 - epoch / CONFIG['num_epochs'])
        teacher_forcing = max(tf_min, tf_min + tf_decay)
        
        train_loss = trainer.train_epoch(train_loader, optimizer, teacher_forcing)
        metrics = trainer.evaluate(val_loader, return_by_zone=True)
        
        elapsed = time.time() - start
        scheduler.step(metrics['rmse'])
        
        print(f"\n[{timestamp()}] Epoch {epoch+1}/{CONFIG['num_epochs']} ({elapsed:.1f}s)")
        print(f"[{timestamp()}]   Teacher forcing: {teacher_forcing:.1%}")
        print(f"[{timestamp()}]   Train Loss: {train_loss:.4f}")
        print(f"[{timestamp()}]   Val RMSE:   {metrics['rmse']:.3f} yards | MAE: {metrics['mae']:.3f} | P90: {metrics.get('p90', 0):.3f} | P95: {metrics.get('p95', 0):.3f}")
        
        # Role breakdown
        if 'receiver_rmse' in metrics:
            recv_rmse = metrics.get('receiver_rmse', 0)
            recv_n = metrics.get('receiver_count', 0)
            def_rmse = metrics.get('defender_rmse', 0)
            def_n = metrics.get('defender_count', 0)
            pass_rmse = metrics.get('passer_rmse', 0)
            pass_n = metrics.get('passer_count', 0)
            print(f"[{timestamp()}]   By Role    - Recv: {recv_rmse:.3f} (n={recv_n}), "
                  f"Def: {def_rmse:.3f} (n={def_n}), "
                  f"Pass: {pass_rmse:.3f} (n={pass_n})")
        
        # Duration breakdown (CRITICAL for understanding long play performance)
        if 'dur_short_rmse' in metrics or 'dur_medium_rmse' in metrics or 'dur_long_rmse' in metrics:
            short_rmse = metrics.get('dur_short_rmse', 0)
            short_n = metrics.get('dur_short_count', 0)
            med_rmse = metrics.get('dur_medium_rmse', 0)
            med_n = metrics.get('dur_medium_count', 0)
            long_rmse = metrics.get('dur_long_rmse', 0)
            long_n = metrics.get('dur_long_count', 0)
            print(f"[{timestamp()}]   By Duration- Short(<2s): {short_rmse:.3f} (n={short_n}), "
                  f"Med(2-4s): {med_rmse:.3f} (n={med_n}), "
                  f"Long(>4s): {long_rmse:.3f} (n={long_n})")
        
        # Frame position breakdown (shows error compounding)
        if 'frame_early_rmse' in metrics or 'frame_middle_rmse' in metrics or 'frame_late_rmse' in metrics:
            early_rmse = metrics.get('frame_early_rmse', 0)
            early_n = metrics.get('frame_early_count', 0)
            mid_rmse = metrics.get('frame_middle_rmse', 0)
            mid_n = metrics.get('frame_middle_count', 0)
            late_rmse = metrics.get('frame_late_rmse', 0)
            late_n = metrics.get('frame_late_count', 0)
            print(f"[{timestamp()}]   By Frame   - Early(0-10): {early_rmse:.3f} (n={early_n}), "
                  f"Mid(10-30): {mid_rmse:.3f} (n={mid_n}), "
                  f"Late(30+): {late_rmse:.3f} (n={late_n})")
        
        # Zone breakdown
        if 'red_zone_rmse' in metrics:
            print(f"[{timestamp()}]   By Zone    - Red: {metrics.get('red_zone_rmse', 0):.3f}, "
                  f"Mid: {metrics.get('midfield_rmse', 0):.3f}")
        
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pth')
            print(f"[{timestamp()}]   NEW BEST! Saved to {OUTPUT_DIR / 'best_model.pth'}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[{timestamp()}]   No improvement ({patience_counter}/{CONFIG['patience']})")
        
        # Save checkpoint every 5 epochs for safety
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rmse': best_rmse,
            }, OUTPUT_DIR / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"[{timestamp()}]   Checkpoint saved")
        
        if patience_counter >= CONFIG['patience']:
            print(f"\n[{timestamp()}] Early stopping!")
            break
    
    print(f"\n[{timestamp()}] " + "=" * 80)
    print(f"[{timestamp()}] MODEL 1 TRAINING COMPLETE - Best RMSE: {best_rmse:.3f} yards")
    print(f"[{timestamp()}] " + "=" * 80)
    
    # Load best model weights for model 1
    model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pth', map_location=CONFIG['device']))
    
    # QUICK WIN 3: Ensemble training - train additional models with different seeds
    n_ensemble = CONFIG.get('n_ensemble', 1)
    ensemble_models = [model]  # First model already trained
    ensemble_rmses = [best_rmse]
    
    if n_ensemble > 1:
        print(f"\n[{timestamp()}] " + "=" * 80)
        print(f"[{timestamp()}] TRAINING ENSEMBLE ({n_ensemble} total models)")
        print(f"[{timestamp()}] " + "=" * 80)
        
        for model_idx in range(1, n_ensemble):
            print(f"\n[{timestamp()}] Training ensemble model {model_idx + 1}/{n_ensemble}...")
            
            # Set different random seed for each model
            seed = 42 + model_idx * 17
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create new model
            ensemble_model = CombinedLSTM(
                input_dim=input_dim,
                hidden_dim=CONFIG['hidden_dim'],
                num_layers=CONFIG['num_layers'],
                dropout=CONFIG['dropout'],
                max_output_frames=CONFIG['max_output_frames']
            )
            
            ens_trainer = Trainer(ensemble_model, CONFIG['device'])
            ens_optimizer = torch.optim.AdamW(
                ensemble_model.parameters(),
                lr=CONFIG['learning_rate'],
                weight_decay=CONFIG['weight_decay']
            )
            ens_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                ens_optimizer, mode='min', factor=0.5, patience=4
            )
            
            ens_best_rmse = float('inf')
            ens_patience_counter = 0
            
            # Train with fewer epochs for ensemble members (they converge faster)
            ens_epochs = max(CONFIG['num_epochs'] // 2, 20)
            
            for epoch in range(ens_epochs):
                start = time.time()
                
                tf_min = CONFIG.get('teacher_forcing_min', 0.0)
                tf_decay = (CONFIG['teacher_forcing_ratio'] - tf_min) * (1 - epoch / ens_epochs)
                teacher_forcing = max(tf_min, tf_min + tf_decay)
                
                train_loss = ens_trainer.train_epoch(train_loader, ens_optimizer, teacher_forcing)
                metrics = ens_trainer.evaluate(val_loader, return_by_zone=False)
                
                elapsed = time.time() - start
                ens_scheduler.step(metrics['rmse'])
                
                print(f"[{timestamp()}]   [M{model_idx+1}] Epoch {epoch+1}/{ens_epochs} ({elapsed:.1f}s) - "
                      f"RMSE: {metrics['rmse']:.3f}", end="")
                
                if metrics['rmse'] < ens_best_rmse:
                    ens_best_rmse = metrics['rmse']
                    torch.save(ensemble_model.state_dict(), OUTPUT_DIR / f'best_model_{model_idx+1}.pth')
                    print(f" - NEW BEST!")
                    ens_patience_counter = 0
                else:
                    ens_patience_counter += 1
                    print(f" ({ens_patience_counter}/{CONFIG['patience'] // 2})")
                
                if ens_patience_counter >= CONFIG['patience'] // 2:
                    print(f"[{timestamp()}]   [M{model_idx+1}] Early stopping")
                    break
            
            # Load best weights for this ensemble member
            ensemble_model.load_state_dict(torch.load(OUTPUT_DIR / f'best_model_{model_idx+1}.pth', 
                                                       map_location=CONFIG['device']))
            ensemble_models.append(ensemble_model)
            ensemble_rmses.append(ens_best_rmse)
            print(f"[{timestamp()}]   [M{model_idx+1}] Best RMSE: {ens_best_rmse:.3f}")
            
            # Clean up optimizer/scheduler to free memory
            del ens_trainer, ens_optimizer, ens_scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Summary
    print(f"\n[{timestamp()}] " + "=" * 80)
    print(f"[{timestamp()}] TRAINING COMPLETE")
    print(f"[{timestamp()}] " + "=" * 80)
    print(f"[{timestamp()}] Ensemble of {len(ensemble_models)} models:")
    for i, rmse in enumerate(ensemble_rmses):
        print(f"[{timestamp()}]   Model {i+1}: {rmse:.3f} yards")
    avg_rmse = sum(ensemble_rmses) / len(ensemble_rmses)
    print(f"[{timestamp()}]   Avg:      {avg_rmse:.3f} yards")
    print(f"[{timestamp()}] Comparison:")
    print(f"[{timestamp()}]   field_aware only:     1.147 yards")
    print(f"[{timestamp()}]   Best single model:    {min(ensemble_rmses):.3f} yards")
    if min(ensemble_rmses) < 1.147:
        print(f"[{timestamp()}]   Improvement:          {(1.147 - min(ensemble_rmses)) / 1.147 * 100:.1f}%")
    
    return ensemble_models, min(ensemble_rmses), route_classifier


# =============================================================================
# SUBMISSION GENERATION - Uses trained model for predictions
# =============================================================================

def generate_submission(model, route_classifier=None):
    """Generate submission file using the trained LSTM model."""
    print(f"\n[{timestamp()}] Generating submission...")
    
    # Load test data
    test_input = pd.read_csv(DATA_DIR / 'test_input.csv')
    print(f"[{timestamp()}] Test data loaded: {len(test_input):,} rows")
    
    # Add base features (motion, ball, field)
    test_input = add_base_features(test_input)
    
    # Add coverage features (zone vs man detection)
    print(f"[{timestamp()}] Computing coverage features for test data...")
    try:
        test_input = add_coverage_features(test_input)
        print(f"[{timestamp()}] Coverage features computed successfully")
    except Exception as e:
        print(f"[{timestamp()}] WARNING: Coverage feature computation failed: {e}")
        for col in CombinedDataset.COVERAGE_FEATURES:
            if col not in test_input.columns:
                test_input[col] = 0.5
    
    # Add defender features - test data HAS receiver positions, so we can compute these!
    print(f"[{timestamp()}] Computing defender features for test data...")
    try:
        test_input = add_defender_features(test_input)
        print(f"[{timestamp()}] Defender features computed successfully")
    except Exception as e:
        print(f"[{timestamp()}] WARNING: Defender feature computation failed: {e}")
        for col in CombinedDataset.DEFENDER_FEATURES:
            if col not in test_input.columns:
                test_input[col] = 0.0
        # Also initialize receiver-relative defender features
        for col in CombinedDataset.RECEIVER_DEFENDER_FEATURES:
            if col not in test_input.columns:
                if col == 'nearest_defender_dist':
                    test_input[col] = 0.5  # Default: 10 yards normalized
                else:
                    test_input[col] = 0.0
    
    # Load best model weights
    model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pth', map_location=CONFIG['device']))
    model.eval()
    model.to(CONFIG['device'])
    
    # Get players to predict
    test_players = test_input[test_input['player_to_predict'] == True].copy()
    unique_players = test_players.groupby(['game_id', 'play_id', 'nfl_id']).first().reset_index()
    
    print(f"[{timestamp()}] Players to predict: {len(unique_players)}")
    
    predictions = []
    n_route_clusters = CONFIG['n_route_clusters']
    
    with torch.no_grad():
        for i, (_, player) in enumerate(unique_players.iterrows()):
            if i % 500 == 0 and i > 0:
                print(f"[{timestamp()}]   Predicting {i}/{len(unique_players)}")
            
            game_id = player['game_id']
            play_id = player['play_id']
            nfl_id = player['nfl_id']
            n_frames = int(player['num_frames_output'])
            
            # Get input data for this player
            mask = (test_input['game_id'] == game_id) & \
                   (test_input['play_id'] == play_id) & \
                   (test_input['nfl_id'] == nfl_id)
            player_input = test_input[mask].sort_values('frame_id')
            
            if len(player_input) == 0:
                continue
                
            last = player_input.iloc[-1]
            
            # Extract base features
            base_features = player_input[CombinedDataset.BASE_FEATURES].values.astype(np.float32)
            base_features = np.nan_to_num(base_features, nan=0.0)
            
            # Route probabilities
            if route_classifier is not None:
                input_traj = player_input[['x', 'y']].values
                route_probs = route_classifier.predict_proba(input_traj)
            else:
                route_probs = np.ones(n_route_clusters) / n_route_clusters
            
            # Broadcast route probs to all frames
            route_probs_expanded = np.tile(route_probs, (len(base_features), 1))
            
            # Combine features
            all_features = np.hstack([base_features, route_probs_expanded]).astype(np.float32)
            
            # Role encoding
            role = player['player_role']
            role_vec = np.array([
                1 if role == 'Passer' else 0,
                1 if role == 'Targeted Receiver' else 0,
                1 if role == 'Defensive Coverage' else 0,
            ], dtype=np.float32)
            
            # Field context
            field_ctx = np.array([
                last['yards_to_goal_norm'],
                last['red_zone'],
                last['goal_line'],
                last['sideline_constrained'],
                last['coverage_zone_prob'],
                last['is_pressed_coverage'],
            ], dtype=np.float32)
            
            # Duration context - we know n_frames from test data
            duration_ctx = np.array([
                n_frames / CONFIG['max_output_frames'],  # Normalized duration
                1.0 if n_frames < 20 else 0.0,           # Short play flag (<2s)
                1.0 if n_frames >= 40 else 0.0,          # Long play flag (>4s)
            ], dtype=np.float32)
            
            # Prepare tensors (batch size 1)
            input_tensor = torch.FloatTensor(all_features).unsqueeze(0).to(CONFIG['device'])
            input_lengths = torch.LongTensor([len(all_features)])
            roles = torch.FloatTensor(role_vec).unsqueeze(0).to(CONFIG['device'])
            last_pos = torch.FloatTensor([last['x'], last['y']]).unsqueeze(0).to(CONFIG['device'])
            last_vel = torch.FloatTensor([last['vx'], last['vy']]).unsqueeze(0).to(CONFIG['device'])
            ball_land = torch.FloatTensor([last['ball_land_x'], last['ball_land_y']]).unsqueeze(0).to(CONFIG['device'])
            field_contexts = torch.FloatTensor(field_ctx).unsqueeze(0).to(CONFIG['device'])
            duration_contexts = torch.FloatTensor(duration_ctx).unsqueeze(0).to(CONFIG['device'])
            
            # Run model
            pred_traj = model(
                input_tensor, input_lengths, roles, last_pos, last_vel, ball_land, field_contexts,
                duration_contexts=duration_contexts, target_trajectory=None, teacher_forcing_ratio=0.0
            )
            
            # Extract predictions for required frames
            lstm_pred = pred_traj[0, :n_frames, :].cpu().numpy()
            
            # Get numpy versions for post-processing
            last_pos_np = np.array([last['x'], last['y']])
            last_vel_np = np.array([last['vx'], last['vy']])
            ball_land_np = np.array([last['ball_land_x'], last['ball_land_y']])
            
            # Compute physics baseline
            physics_pred = compute_physics_baseline(last_pos_np, last_vel_np, ball_land_np, role, n_frames)
            
            # Blend LSTM with physics based on role
            blended_pred = blend_with_physics(lstm_pred, physics_pred, role, n_frames)
            
            # Apply physics constraints (speed limits, field bounds)
            final_pred = apply_physics_constraints(blended_pred, last_pos_np, last_vel_np, role)
            
            for frame in range(n_frames):
                predictions.append({
                    'id': f"{game_id}_{play_id}_{nfl_id}_{frame + 1}",
                    'x': float(final_pred[frame, 0]),
                    'y': float(final_pred[frame, 1])
                })
    
    # Save submission
    submission = pd.DataFrame(predictions)
    submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
    
    print(f"[{timestamp()}] Submission saved: {len(submission)} predictions")
    print(f"[{timestamp()}] Sample predictions:")
    print(submission.head(10))
    
    return submission


# =============================================================================
# INFERENCE SERVER SETUP (Required for Competition)
# =============================================================================

import polars as pl
import kaggle_evaluation.nfl_inference_server

# Global model(s) and classifier - loaded once
# Support for ensemble of models (QUICK WIN 3)
LOADED_MODELS = []  # List of models for ensemble
LOADED_ROUTE_CLASSIFIER = None
MODEL_LOADED = False
INFERENCE_BATCH_COUNT = 0  # Track inference progress

def load_model_for_inference():
    """Load trained model(s) for inference. Called once on first prediction."""
    global LOADED_MODELS, LOADED_ROUTE_CLASSIFIER, MODEL_LOADED
    
    if MODEL_LOADED:
        return
    
    print(f"[{timestamp()}] Loading model(s) for inference...")
    
    # Check if we have a pre-trained model
    model_path = OUTPUT_DIR / 'best_model.pth'
    input_dim = len(CombinedDataset.BASE_FEATURES) + CONFIG['n_route_clusters']
    
    if model_path.exists():
        print(f"[{timestamp()}] Found pre-trained model at {model_path}")
        
        # Load primary model
        model = CombinedLSTM(
            input_dim=input_dim,
            hidden_dim=CONFIG['hidden_dim'],
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout'],
            max_output_frames=CONFIG['max_output_frames']
        )
        model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        LOADED_MODELS.append(model)
        
        # Load additional ensemble models if they exist
        n_ensemble = CONFIG.get('n_ensemble', 1)
        for i in range(1, n_ensemble):
            ens_path = OUTPUT_DIR / f'best_model_{i+1}.pth'
            if ens_path.exists():
                print(f"[{timestamp()}] Found ensemble model {i+1} at {ens_path}")
                ens_model = CombinedLSTM(
                    input_dim=input_dim,
                    hidden_dim=CONFIG['hidden_dim'],
                    num_layers=CONFIG['num_layers'],
                    dropout=CONFIG['dropout'],
                    max_output_frames=CONFIG['max_output_frames']
                )
                ens_model.load_state_dict(torch.load(ens_path, map_location=CONFIG['device']))
                ens_model.to(CONFIG['device'])
                ens_model.eval()
                LOADED_MODELS.append(ens_model)
        
        # Create a basic route classifier (will use uniform probs if not fitted)
        LOADED_ROUTE_CLASSIFIER = RouteStemClassifier(CONFIG['n_route_clusters'])
        
    else:
        print(f"[{timestamp()}] No pre-trained model found. Training now...")
        # Train the model first - returns list of models for ensemble
        trained_models, _, LOADED_ROUTE_CLASSIFIER = main()
        
        # Handle both single model and list of models
        if isinstance(trained_models, list):
            LOADED_MODELS = trained_models
        else:
            LOADED_MODELS = [trained_models]
        
        for m in LOADED_MODELS:
            m.eval()
    
    MODEL_LOADED = True
    print(f"[{timestamp()}] {len(LOADED_MODELS)} model(s) ready for inference")


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame:
    """
    Inference function called by the competition server.
    
    Uses ensemble averaging if multiple models are available.
    
    Args:
        test: DataFrame with columns [game_id, play_id, nfl_id, step] - what to predict
        test_input: DataFrame with input features for the current timestep
    
    Returns:
        DataFrame with columns [x, y] - predicted positions
    """
    global LOADED_MODELS, LOADED_ROUTE_CLASSIFIER, INFERENCE_BATCH_COUNT
    
    # Load model on first call
    load_model_for_inference()
    
    # Track inference batches
    INFERENCE_BATCH_COUNT += 1
    
    # Only log every 100 batches to reduce noise
    verbose = (INFERENCE_BATCH_COUNT == 1) or (INFERENCE_BATCH_COUNT % 100 == 0)
    
    if verbose:
        print(f"[{timestamp()}] Inference batch {INFERENCE_BATCH_COUNT} ({len(test)} predictions)")
    
    # Convert polars to pandas for compatibility with our functions
    test_pd = test.to_pandas()
    test_input_pd = test_input.to_pandas()
    
    # Add features to input data (silent for most batches)
    test_input_pd = add_base_features(test_input_pd, verbose=verbose)
    
    # Add coverage features
    try:
        test_input_pd = add_coverage_features(test_input_pd, verbose=verbose)
    except:
        for col in CombinedDataset.COVERAGE_FEATURES:
            if col not in test_input_pd.columns:
                test_input_pd[col] = 0.5
    
    # Add defender features (includes receiver-relative features)
    try:
        test_input_pd = add_defender_features(test_input_pd, verbose=verbose)
    except:
        # Fallback for defender features
        for col in CombinedDataset.DEFENDER_FEATURES:
            if col not in test_input_pd.columns:
                test_input_pd[col] = 0.0
        # Fallback for receiver-relative defender features
        for col in CombinedDataset.RECEIVER_DEFENDER_FEATURES:
            if col not in test_input_pd.columns:
                if col == 'nearest_defender_dist':
                    test_input_pd[col] = 0.5  # Default: 10 yards normalized
                else:
                    test_input_pd[col] = 0.0
    
    predictions = []
    
    # Debug: print test columns on FIRST call only
    if INFERENCE_BATCH_COUNT == 1:
        print(f"[{timestamp()}] Test columns: {list(test_pd.columns)}")
        print(f"[{timestamp()}] Test input columns: {list(test_input_pd.columns)[:20]}...")
    
    with torch.no_grad():
        for idx, row in test_pd.iterrows():
            game_id = row['game_id']
            play_id = row['play_id']
            nfl_id = row['nfl_id']
            
            # Handle different column names for frame/step
            if 'step' in test_pd.columns:
                step = row['step']
            elif 'frame_id' in test_pd.columns:
                step = row['frame_id']
            else:
                # Default to 1 if no step/frame info
                step = 1
            
            # Get input data for this player
            mask = (test_input_pd['game_id'] == game_id) & \
                   (test_input_pd['play_id'] == play_id) & \
                   (test_input_pd['nfl_id'] == nfl_id)
            player_input = test_input_pd[mask].sort_values('frame_id')
            
            if len(player_input) == 0:
                # Fallback prediction - use any available position info from test
                if 'x' in row and 'y' in row:
                    predictions.append({'x': float(row['x']), 'y': float(row['y'])})
                else:
                    predictions.append({'x': 50.0, 'y': 26.65})  # Center of field
                continue
            
            last = player_input.iloc[-1]
            
            # Get number of frames to predict
            n_frames = max(int(step), 1)
            
            # Extract features
            base_features = player_input[CombinedDataset.BASE_FEATURES].values.astype(np.float32)
            base_features = np.nan_to_num(base_features, nan=0.0)
            
            # Route probabilities
            if LOADED_ROUTE_CLASSIFIER is not None:
                input_traj = player_input[['x', 'y']].values
                route_probs = LOADED_ROUTE_CLASSIFIER.predict_proba(input_traj)
            else:
                route_probs = np.ones(CONFIG['n_route_clusters']) / CONFIG['n_route_clusters']
            
            route_probs_expanded = np.tile(route_probs, (len(base_features), 1))
            all_features = np.hstack([base_features, route_probs_expanded]).astype(np.float32)
            
            # Role encoding
            role = player_input.iloc[0].get('player_role', 'Defensive Coverage')
            role_vec = np.array([
                1 if role == 'Passer' else 0,
                1 if role == 'Targeted Receiver' else 0,
                1 if role == 'Defensive Coverage' else 0,
            ], dtype=np.float32)
            
            # Field context
            field_ctx = np.array([
                last.get('yards_to_goal_norm', 0.5),
                last.get('red_zone', 0),
                last.get('goal_line', 0),
                last.get('sideline_constrained', 0),
                last.get('coverage_zone_prob', 0.5),
                last.get('is_pressed_coverage', 0),
            ], dtype=np.float32)
            
            # Duration context
            duration_ctx = np.array([
                n_frames / CONFIG['max_output_frames'],
                1.0 if n_frames < 20 else 0.0,           # Short play flag (<2s)
                1.0 if n_frames >= 40 else 0.0,          # Long play flag (>4s)
            ], dtype=np.float32)
            
            # Prepare tensors
            input_tensor = torch.FloatTensor(all_features).unsqueeze(0).to(CONFIG['device'])
            input_lengths = torch.LongTensor([len(all_features)])
            roles_t = torch.FloatTensor(role_vec).unsqueeze(0).to(CONFIG['device'])
            last_pos = torch.FloatTensor([last['x'], last['y']]).unsqueeze(0).to(CONFIG['device'])
            last_vel = torch.FloatTensor([last.get('vx', 0), last.get('vy', 0)]).unsqueeze(0).to(CONFIG['device'])
            ball_land = torch.FloatTensor([last.get('ball_land_x', last['x']), 
                                           last.get('ball_land_y', last['y'])]).unsqueeze(0).to(CONFIG['device'])
            field_contexts_t = torch.FloatTensor(field_ctx).unsqueeze(0).to(CONFIG['device'])
            duration_contexts_t = torch.FloatTensor(duration_ctx).unsqueeze(0).to(CONFIG['device'])
            
            # ENSEMBLE: Run all models and average predictions
            ensemble_preds = []
            for model in LOADED_MODELS:
                pred_traj = model(
                    input_tensor, input_lengths, roles_t, last_pos, last_vel, ball_land, field_contexts_t,
                    duration_contexts=duration_contexts_t, target_trajectory=None, teacher_forcing_ratio=0.0
                )
                
                # Get prediction for requested step
                step_idx = min(step - 1, pred_traj.size(1) - 1)
                pred_x = pred_traj[0, step_idx, 0].cpu().item()
                pred_y = pred_traj[0, step_idx, 1].cpu().item()
                ensemble_preds.append((pred_x, pred_y))
            
            # Average ensemble predictions
            pred_x = np.mean([p[0] for p in ensemble_preds])
            pred_y = np.mean([p[1] for p in ensemble_preds])
            
            # Apply physics constraints
            pred_x = np.clip(pred_x, FIELD_X_MIN, FIELD_X_MAX)
            pred_y = np.clip(pred_y, 0, FIELD_WIDTH)
            
            predictions.append({'x': float(pred_x), 'y': float(pred_y)})
    
    # Return as polars DataFrame
    return pl.DataFrame(predictions)


# =============================================================================
# ENTRY POINT
# =============================================================================

# Create inference server
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Competition evaluation - just serve predictions
    print(f"[{timestamp()}] Competition mode - starting inference server")
    inference_server.serve()
else:
    # Development mode - train model first, then test with local gateway
    print(f"[{timestamp()}] Development mode - training model")
    
    # Train the model(s) - returns list of models for ensemble
    trained_models, best_rmse, route_classifier = main()
    
    # Handle both single model and list of models
    if isinstance(trained_models, list):
        LOADED_MODELS = trained_models
    else:
        LOADED_MODELS = [trained_models]
    
    LOADED_ROUTE_CLASSIFIER = route_classifier
    MODEL_LOADED = True
    
    print(f"[{timestamp()}] Training complete. Best RMSE: {best_rmse:.3f}")
    print(f"[{timestamp()}] Ensemble size: {len(LOADED_MODELS)} models")
    print(f"[{timestamp()}] Running local gateway for testing...")
    print(f"[{timestamp()}] NOTE: Inference server will log every 100 batches to reduce noise")
    
    # Run local test
    try:
        inference_server.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))
        print(f"[{timestamp()}] ============================================================")
        print(f"[{timestamp()}] INFERENCE COMPLETE - Total batches: {INFERENCE_BATCH_COUNT}")
        print(f"[{timestamp()}] ============================================================")
    except Exception as e:
        print(f"[{timestamp()}] Inference error: {e}")
        print(f"[{timestamp()}] Batches completed before error: {INFERENCE_BATCH_COUNT}")
