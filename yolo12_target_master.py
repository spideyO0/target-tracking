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

    def select_targets(self, tracks):
        """
        Process tracker output, filter unsafe, and select Primary.
        """
        valid_targets = []
        
        # --- 1. Filter & Parse ---
        for t in tracks:
            # Check Class (Strict Person Only)
            cls_id = int(t.cls[0])
            if cls_id != 0: 
                continue 
            
            # Check Safety
            box = t.xyxy[0].cpu().numpy()
            if self.is_safe(box): 
                continue
            
            valid_targets.append(t)

        # --- 2. Select Primary (Strategy: Closest to Center) ---
        best_dist = float('inf')
        best_t_data = None
        
        formatted_targets = []
        
        for t in valid_targets:
            box = t.xyxy[0].cpu().numpy()
            tid = int(t.id[0]) if t.id is not None else -1
            
            x1, y1, x2, y2 = map(int, box)
            tcx = (x1 + x2) // 2
            tcy = (y1 + y2) // 2
            
            dist = np.sqrt((tcx - self.cx)**2 + (tcy - self.cy)**2)
            
            # Target Data Structure
            target_data = {
                'id': tid,
                'box': (x1, y1, x2, y2),
                'center': (tcx, tcy),
                'dist': dist,
                'locked': False
            }
            
            # Selection Logic
            if dist < best_dist:
                best_dist = dist
                best_t_data = target_data
            
            formatted_targets.append(target_data)

        # --- 3. Update State ---
        if best_t_data:
            self.primary_target = best_t_data
            # Auto-lock logic: If closest, it becomes locked
            self.locked_ids.add(best_t_data['id'])
        else:
            self.primary_target = None

        # Sync 'locked' status
        for ft in formatted_targets:
            if ft['id'] in self.locked_ids:
                ft['locked'] = True
                
        return formatted_targets

def draw_hud(frame, turret, targets, primary):
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2
    
    # --- 1. Safe Zones (Visualized) ---
    for zone in SAFE_ZONES:
        x1 = int(zone[0] * W)
        y1 = int(zone[1] * H)
        x2 = int(zone[2] * W)
        y2 = int(zone[3] * H)
        
        # Transparent Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 50), -1) # Dark Red
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Border + Text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, "SAFE ZONE - NO TRACK", (x1+10, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- 2. Turret Crosshair ---
    color_cross = (0, 255, 0) if primary is None else (0, 0, 255)
    cv2.circle(frame, (cx, cy), 15, color_cross, 1)
    cv2.line(frame, (cx - 25, cy), (cx + 25, cy), color_cross, 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy + 25), color_cross, 1)

    # --- 3. Targets ---
    for t in targets:
        x1, y1, x2, y2 = t['box']
        
        if t['locked']:
            color = (0, 0, 255) # Red for Locked
            thick = 2
            # Connect to center
            cv2.line(frame, (cx, cy), t['center'], color, 1)
        else:
            color = (0, 255, 255) # Yellow for Tracked
            thick = 1
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.putText(frame, f"ID:{t['id']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # --- 4. System Info Panel ---
    # Background
    panel_w, panel_h = 240, 160
    panel_x = W - panel_w - 10
    panel_y = 10
    
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 0), 1)
    
    # Text
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
    
    status = "ENGAGED" if primary else "SCANNING"
    cv2.putText(frame, f"STATUS    : {status}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if primary else (0, 255, 0), 1)
    sy += line_h
    
    cv2.putText(frame, f"TARGETS   : {len(targets)}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    print("[SYSTEM] Initializing Safe Turret System...")
    
    # Load Model (Ensure you have ultralytics installed)
    # Using 'yolo11n.pt' as standard nano model, but variable name implies 12 capability
    try:
        model = YOLO("yolo11n.pt") 
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
    
    print(f"[SYSTEM] Cam: {W}x{H}")
    print("[SYSTEM] Mode: SAFE HUMAN TRACKING (PASSIVE)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Track with ByteTrack for ID consistency
        # classes=[0] enforces Person only filter at the model level
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0])
        
        # --- LOGIC UPDATE ---
        targets = []
        primary = None
        
        if results and results[0].boxes.id is not None:
             # Pass the Boxes object to manager
            targets = manager.select_targets(results[0].boxes)
            primary = manager.primary_target

        # --- TURRET PID UPDATE ---
        if primary:
            # Error = Target Center - Frame Center
            error_x = primary['center'][0] - (W // 2)
            error_y = primary['center'][1] - (H // 2)
            turret.update(error_x, error_y)
        else:
            # If no target, maybe drift back to 0? Or just stay.
            # turret.update(0, 0) # Uncomment to auto-center when idle
            pass

        # --- RENDER ---
        draw_hud(frame, turret, targets, primary)

        cv2.imshow("Safe Turret Sim", frame)
        if cv2.waitKey(1) == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
