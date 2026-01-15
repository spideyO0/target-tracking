import cv2
import numpy as np
import time
try:
    from ultralytics import YOLO
except ImportError:
    print("\n[ERROR] 'ultralytics' library not found.")
    print("Please run: pip install ultralytics")
    print("Then try running this script again.\n")
    exit(1)

# --- CONFIGURATION ---
SAFE_ZONES = [
    # Format: (x_start_norm, y_start_norm, x_end_norm, y_end_norm)
    # These are normalized coordinates (0.0 to 1.0)
    (0.0, 0.0, 0.15, 1.0), # Left edge safety margin (e.g. "Doorway")
    (0.85, 0.0, 1.0, 1.0)  # Right edge safety margin
]

TURRET_LIMITS = {
    'MAX_PAN_SPEED': 2.0,   # Max degrees per frame
    'MAX_TILT_SPEED': 1.5,
    'PAN_RANGE': (-90, 90), # Mechanical limit
    'TILT_RANGE': (-45, 45)
}

AIM_MODES = {
    1: "HEAD",
    2: "UPPER_BODY",
    3: "NON_LETHAL"
}

class TurretController:
    """
    Simulates a 2-axis gimbal/turret with PID control.
    Physical Model: Converts pixel error -> servo angle adjustments.
    """
    def __init__(self, kp=0.1, ki=0.01, kd=0.05):
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        
        # PID constants
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # State
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.integral_x = 0
        self.integral_y = 0

    def update(self, error_x, error_y):
        """
        Calculate new servo angles based on pixel error.
        error_x, error_y: Pixel difference from center.
        """
        # --- PID: X-axis (Pan) ---
        self.integral_x += error_x
        derivative_x = error_x - self.prev_error_x
        output_x = (self.kp * error_x) + (self.ki * self.integral_x) + (self.kd * derivative_x)
        self.prev_error_x = error_x
        
        # --- PID: Y-axis (Tilt) ---
        self.integral_y += error_y
        derivative_y = error_y - self.prev_error_y
        output_y = (self.kp * error_y) + (self.ki * self.integral_y) + (self.kd * derivative_y)
        self.prev_error_y = error_y
        
        # --- Physical Limiting ---
        # 1. Speed Limit (Simulate servo max speed)
        # Note: We scale output by 0.1 to convert "PID unit" to "Degrees" approximately
        delta_pan = np.clip(output_x * 0.1, -TURRET_LIMITS['MAX_PAN_SPEED'], TURRET_LIMITS['MAX_PAN_SPEED'])
        delta_tilt = np.clip(output_y * 0.1, -TURRET_LIMITS['MAX_TILT_SPEED'], TURRET_LIMITS['MAX_TILT_SPEED'])
        
        # 2. Angle Limit (Mechanical stops)
        self.pan_angle = np.clip(self.pan_angle + delta_pan, TURRET_LIMITS['PAN_RANGE'][0], TURRET_LIMITS['PAN_RANGE'][1])
        # Invert tilt because +Y pixel (down) usually means -Tilt (down) or vice versa depending on mount.
        # Here: Box is below center (+error) -> Camera needs to look down.
        # If looking down is negative angle:
        self.tilt_angle = np.clip(self.tilt_angle - delta_tilt, TURRET_LIMITS['TILT_RANGE'][0], TURRET_LIMITS['TILT_RANGE'][1])
        
        return self.pan_angle, self.tilt_angle

class TargetManager:
    """
    Handles Multi-Target logic, Locking, and Safety Zones.
    """
    def __init__(self, frame_width, frame_height):
        self.W = frame_width
        self.H = frame_height
        self.cx = frame_width // 2
        self.cy = frame_height // 2
        
        self.locked_ids = set()
        self.primary_target = None
        
    def is_safe(self, box):
        """
        Check if target is in a Safe Zone (Ignore Zone).
        box: (x1, y1, x2, y2)
        """
        # Calculate box center normalized
        bx_cx = ((box[0] + box[2]) / 2) / self.W
        bx_cy = ((box[1] + box[3]) / 2) / self.H
        
        for zone in SAFE_ZONES:
            zx1, zy1, zx2, zy2 = zone
            # If center of target is inside the safe rect
            if zx1 <= bx_cx <= zx2 and zy1 <= bx_cy <= zy2:
                return True # In safe zone
        return False

    def select_targets(self, results, aim_mode):
        """
        Process tracker output, filter unsafe, and select Primary.
        Accepts full YOLO Results object to access Keypoints.
        """
        valid_targets = []
        
        boxes = results.boxes
        keypoints = results.keypoints
        
        if boxes is None or boxes.id is None:
            self.primary_target = None
            return []

        # --- 1. Filter & Parse ---
        for i, box in enumerate(boxes):
            # Check Class (Strict Person Only)
            cls_id = int(box.cls[0])
            if cls_id != 0: 
                continue 
            
            # Check Safety
            xyxy = box.xyxy[0].cpu().numpy()
            if self.is_safe(xyxy): 
                continue
            
            # Extract ID
            tid = int(box.id[0]) if box.id is not None else -1
            
            # --- PRECISE TARGETING LOGIC (Keypoints) ---
            x1, y1, x2, y2 = map(int, xyxy)
            aim_x, aim_y = (x1 + x2) // 2, (y1 + y2) // 2 # Default to center
            
            if keypoints is not None and len(keypoints) > i:
                # Keypoints for this person
                kps = keypoints[i].xy[0].cpu().numpy() # Shape (17, 2)
                # Kps: 0=Nose, 5=LSh, 6=RSh, 11=LHip, 12=RHip, 13=LKnee, 14=RKnee
                
                # Check confidence of relevant keypoints (simplified: if not [0,0])
                def is_valid(kp_idx):
                    return kps.shape[0] > kp_idx and kps[kp_idx][0] != 0 and kps[kp_idx][1] != 0

                if aim_mode == 1: # HEAD
                    if is_valid(0): # Nose
                        aim_x, aim_y = kps[0]
                    elif is_valid(1) and is_valid(2): # Midpoint Eyes
                        aim_x = (kps[1][0] + kps[2][0]) / 2
                        aim_y = (kps[1][1] + kps[2][1]) / 2
                    else:
                        # Fallback
                        h = y2 - y1
                        aim_y = y1 + (h * 0.12)

                elif aim_mode == 3: # NON_LETHAL (Legs)
                    if is_valid(13) and is_valid(14): # Knees Midpoint
                        aim_x = (kps[13][0] + kps[14][0]) / 2
                        aim_y = (kps[13][1] + kps[14][1]) / 2
                    elif is_valid(11) and is_valid(12): # Hips (aim lower than hips)
                        mid_x = (kps[11][0] + kps[12][0]) / 2
                        mid_y = (kps[11][1] + kps[12][1]) / 2
                        aim_x = mid_x
                        aim_y = mid_y + (y2 - mid_y) * 0.5 # Halfway from hips to feet
                    else:
                        # Fallback
                        h = y2 - y1
                        aim_y = y1 + (h * 0.75)

                else: # UPPER_BODY
                    if is_valid(5) and is_valid(6): # Shoulders Midpoint
                        aim_x = (kps[5][0] + kps[6][0]) / 2
                        aim_y = (kps[5][1] + kps[6][1]) / 2
                        # Adjust slightly down for chest center
                        aim_y += (y2 - y1) * 0.05 
                    else:
                        # Fallback
                        h = y2 - y1
                        aim_y = y1 + (h * 0.35)

            # Store Data
            target_data = {
                'id': tid,
                'box': (x1, y1, x2, y2),
                'center': ((x1+x2)//2, (y1+y2)//2), # Box Center
                'aim_point': (int(aim_x), int(aim_y)), # Precise Aim Point
                'dist_to_center': np.sqrt(((x1+x2)//2 - self.cx)**2 + ((y1+y2)//2 - self.cy)**2),
                'locked': False
            }
            valid_targets.append(target_data)

        # --- 2. Select Primary (Strategy: Closest to Center) ---
        best_dist = float('inf')
        best_t_data = None
        
        for t in valid_targets:
            if t['dist_to_center'] < best_dist:
                best_dist = t['dist_to_center']
                best_t_data = t
        
        # --- 3. Update State ---
        if best_t_data:
            self.primary_target = best_t_data
            self.locked_ids.add(best_t_data['id'])
        else:
            self.primary_target = None

        # Sync 'locked' status
        for ft in valid_targets:
            if ft['id'] in self.locked_ids:
                ft['locked'] = True
                
        return valid_targets

def draw_hud(frame, turret, targets, primary, aim_mode_idx):
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2
    
    # --- 1. Safe Zones (Visualized) ---
    for zone in SAFE_ZONES:
        x1 = int(zone[0] * W)
        y1 = int(zone[1] * H)
        x2 = int(zone[2] * W)
        y2 = int(zone[3] * H)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 50), -1) 
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, "SAFE ZONE", (x1+10, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- 2. Turret Crosshair ---
    color_cross = (0, 255, 0) if primary is None else (0, 0, 255)
    cv2.circle(frame, (cx, cy), 15, color_cross, 1)
    cv2.line(frame, (cx - 25, cy), (cx + 25, cy), color_cross, 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy + 25), color_cross, 1)

    # --- 3. Targets (ALL) ---
    for t in targets:
        x1, y1, x2, y2 = t['box']
        
        if t['locked'] and t == primary:
            color = (0, 0, 255) 
            thick = 2
            # Line to Aim Point
            cv2.line(frame, (cx, cy), t['aim_point'], color, 1)
        else:
            color = (0, 255, 255) 
            thick = 1
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.putText(frame, f"ID:{t['id']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Visualize Aim Point for EVERYONE
        ax, ay = t['aim_point']
        cv2.circle(frame, (ax, ay), 4, (0, 0, 255), -1)

    # --- 4. System Info Panel ---
    panel_w, panel_h = 240, 180 
    panel_x = W - panel_w - 10
    panel_y = 10
    
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 0), 1)
    
    sx = panel_x + 10
    sy = panel_y + 25
    line_h = 20
    
    cv2.putText(frame, "TURRET CONTROL SYS", (sx, sy), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (0, 255, 0), 1)
    sy += 10
    cv2.line(frame, (sx, sy), (sx+200, sy), (50, 50, 50), 1)
    sy += 20
    
    cv2.putText(frame, f"PAN ANGLE : {turret.pan_angle:+.1f}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    sy += line_h
    cv2.putText(frame, f"TILT ANGLE: {turret.tilt_angle:+.1f}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    sy += line_h
    
    mode_str = AIM_MODES.get(aim_mode_idx, "UNKNOWN")
    cv2.putText(frame, f"MODE      : {mode_str}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    sy += line_h

    status = "ENGAGED" if primary else "SCANNING"
    cv2.putText(frame, f"STATUS    : {status}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if primary else (0, 255, 0), 1)
    sy += line_h
    
    cv2.putText(frame, f"TARGETS   : {len(targets)}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    print("[SYSTEM] Initializing Safe Turret System...")
    
    # Load POSE Model for precise keypoint targeting
    try:
        model = YOLO("yolo11n-pose.pt") 
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    # Set resolution to 1920x1080 (Full HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Init Subsystems
    turret = TurretController(kp=0.1, ki=0.01, kd=0.05)
    manager = TargetManager(W, H)
    
    aim_mode = 2 # Default: UPPER_BODY
    
    print(f"[SYSTEM] Cam: {W}x{H}")
    print("[SYSTEM] Mode: PRECISE POSE TRACKING")
    print("[CONTROL] Keys: '1'=Head, '2'=Upper Body, '3'=Non-Lethal (Legs)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Track with ByteTrack for ID consistency
        # classes=[0] enforces Person only filter
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0])
        
        # --- LOGIC UPDATE ---
        targets = []
        primary = None
        
        if results:
             # Pass the FULL results object (contains keypoints)
            targets = manager.select_targets(results[0], aim_mode)
            primary = manager.primary_target

        # --- TURRET PID UPDATE ---
        if primary:
            # Use the calculated AIM POINT
            target_x, target_y = primary['aim_point']
            
            # Error = Target Point - Frame Center
            error_x = target_x - (W // 2)
            error_y = target_y - (H // 2)
            turret.update(error_x, error_y)
        
        # --- RENDER ---
        draw_hud(frame, turret, targets, primary, aim_mode)

        cv2.imshow("Safe Turret Sim", frame)
        
        # Input Handling
        key = cv2.waitKey(1)
        if key == ord('q'): 
            break
        elif key == ord('1'):
            aim_mode = 1
        elif key == ord('2'):
            aim_mode = 2
        elif key == ord('3'):
            aim_mode = 3

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
