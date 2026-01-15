import cv2
import numpy as np
import time
import os
import pickle
import warnings

# Suppress DeprecationWarning from face_recognition_models
warnings.filterwarnings("ignore", category=UserWarning, module='face_recognition_models')

try:
    from ultralytics import YOLO
except ImportError:
    print("\n[ERROR] 'ultralytics' library not found.")
    print("Please run: pip install ultralytics")
    print("Then try running this script again.\n")
    exit(1)

# --- OPTIONAL: Face Recognition for Persistence ---
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
    print("[SYSTEM] Face Recognition Module: ACTIVE")
except ImportError:
    FACE_REC_AVAILABLE = False
    print("\n[WARNING] 'face_recognition' library not found.")
    print("Persistent identity tracking will be limited to active sessions.")
    print("To enable full Re-ID, run: pip install face_recognition\n")

# --- CONFIGURATION ---
SAFE_ZONES = [
    # Format: (x_start_norm, y_start_norm, x_end_norm, y_end_norm)
    (0.0, 0.0, 0.15, 1.0), # Left edge
    (0.85, 0.0, 1.0, 1.0)  # Right edge
]

TURRET_LIMITS = {
    'MAX_PAN_SPEED': 2.0,   
    'MAX_TILT_SPEED': 1.5,
    'PAN_RANGE': (-90, 90), 
    'TILT_RANGE': (-45, 45)
}

AIM_MODES = {
    1: "HEAD (Forehead)",
    2: "UPPER_BODY",
    3: "NON_LETHAL"
}

DB_PATH = "data/faces/identity_db.pkl"

class IdentityManager:
    def __init__(self):
        # Database Schema: 
        # { 'pid': int, 'encodings': [list of encodings], 'images': [list of face crops], 'last_seen': timestamp }
        self.database = [] 
        self.next_pid = 1
        
        # Cache: Map YOLO_ID (current session) -> Persistent_ID
        self.session_map = {}
        
        self.load_database()

    def load_database(self):
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.database = data.get('database', [])
                    self.next_pid = data.get('next_pid', 1)
                print(f"[SYSTEM] Loaded {len(self.database)} identities from {DB_PATH}")
            except Exception as e:
                print(f"[ERROR] Failed to load database: {e}")
        else:
            print("[SYSTEM] No existing database found. Starting fresh.")

    def save_database(self):
        # Create directory if not exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        try:
            with open(DB_PATH, 'wb') as f:
                data = {
                    'database': self.database,
                    'next_pid': self.next_pid
                }
                pickle.dump(data, f)
            # print("[SYSTEM] Database saved.") # Too verbose for loop
        except Exception as e:
            print(f"[ERROR] Failed to save database: {e}")

    def get_face_encoding_and_crop(self, frame, kps):
        if not FACE_REC_AVAILABLE:
            return None, None

        face_pts = []
        for i in [0, 1, 2, 3, 4]:
            if len(kps) > i and kps[i][0] != 0:
                face_pts.append(kps[i])
        
        if len(face_pts) < 3:
            return None, None 

        face_pts = np.array(face_pts)
        x_min, y_min = np.min(face_pts, axis=0)
        x_max, y_max = np.max(face_pts, axis=0)
        
        w = x_max - x_min
        h = y_max - y_min
        pad_x = int(w * 0.5)
        pad_y = int(h * 0.5)
        
        H, W = frame.shape[:2]
        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(W, int(x_max + pad_x))
        y2 = min(H, int(y_max + pad_y))
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None, None
            
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        try:
            h_c, w_c = rgb_crop.shape[:2]
            known_loc = [(0, w_c, h_c, 0)]
            
            encodings = face_recognition.face_encodings(rgb_crop, known_face_locations=known_loc)
            if encodings:
                # Return the original BGR crop (for storage/display) and the encoding
                return encodings[0], face_crop
        except Exception:
            pass
        return None, None

    def resolve_identity(self, frame, yolo_id, keypoints):
        # 1. Fast Path: Session Cache
        if yolo_id in self.session_map:
            return self.session_map[yolo_id]
            
        # 2. Slow Path: Face Recognition
        encoding, face_crop = self.get_face_encoding_and_crop(frame, keypoints)
        
        if encoding is not None:
            best_match_pid = None
            
            # Search Database
            for record in self.database:
                matches = face_recognition.compare_faces(record['encodings'], encoding, tolerance=0.6)
                if True in matches:
                    best_match_pid = record['pid']
                    
                    # Update Existing Record
                    # Limit encodings to 50 for performance
                    if len(record['encodings']) < 50:
                        record['encodings'].append(encoding)
                        
                        # Save Image to Disk
                        if 'image_paths' not in record:
                            record['image_paths'] = []
                            
                        if len(record['image_paths']) < 10:
                            filename = f"data/faces/images/pid_{best_match_pid}_{int(time.time()*1000)}.jpg"
                            cv2.imwrite(filename, face_crop)
                            record['image_paths'].append(filename)
                            
                        self.save_database() # Auto-save on update
                    break
            
            if best_match_pid:
                self.session_map[yolo_id] = best_match_pid
                return best_match_pid
            else:
                # Create New Identity
                new_pid = self.next_pid
                self.next_pid += 1
                
                # Save Image to Disk
                filename = f"data/faces/images/pid_{new_pid}_{int(time.time()*1000)}.jpg"
                cv2.imwrite(filename, face_crop)
                
                # Create record
                new_record = {
                    'pid': new_pid, 
                    'encodings': [encoding], 
                    'image_paths': [filename], # Store path, not raw image
                    'created_at': time.time()
                }
                
                self.database.append(new_record)
                self.session_map[yolo_id] = new_pid
                self.save_database() # Auto-save on creation
                return new_pid
        
        # 3. Fallback: Temporary Session ID
        temp_pid = 1000 + yolo_id 
        self.session_map[yolo_id] = temp_pid
        return temp_pid

class TurretController:
    def __init__(self, kp=0.1, ki=0.01, kd=0.05):
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        self.kp = kp; self.ki = ki; self.kd = kd
        self.prev_error_x = 0; self.prev_error_y = 0
        self.integral_x = 0; self.integral_y = 0

    def update(self, error_x, error_y):
        self.integral_x += error_x
        derivative_x = error_x - self.prev_error_x
        output_x = (self.kp * error_x) + (self.ki * self.integral_x) + (self.kd * derivative_x)
        self.prev_error_x = error_x
        
        self.integral_y += error_y
        derivative_y = error_y - self.prev_error_y
        output_y = (self.kp * error_y) + (self.ki * self.integral_y) + (self.kd * derivative_y)
        self.prev_error_y = error_y
        
        delta_pan = np.clip(output_x * 0.1, -TURRET_LIMITS['MAX_PAN_SPEED'], TURRET_LIMITS['MAX_PAN_SPEED'])
        delta_tilt = np.clip(output_y * 0.1, -TURRET_LIMITS['MAX_TILT_SPEED'], TURRET_LIMITS['MAX_TILT_SPEED'])
        
        self.pan_angle = np.clip(self.pan_angle + delta_pan, TURRET_LIMITS['PAN_RANGE'][0], TURRET_LIMITS['PAN_RANGE'][1])
        self.tilt_angle = np.clip(self.tilt_angle - delta_tilt, TURRET_LIMITS['TILT_RANGE'][0], TURRET_LIMITS['TILT_RANGE'][1])
        return self.pan_angle, self.tilt_angle

class TargetManager:
    def __init__(self, frame_width, frame_height):
        self.W = frame_width
        self.H = frame_height
        self.cx = frame_width // 2
        self.cy = frame_height // 2
        
        self.locked_ids = set() # Set of PIDs
        self.primary_target = None
        
        # DEFAULT MODE: Manual
        self.manual_mode = True 
        self.selected_pid = None # We now select by Persistent ID
        
        self.identity_manager = IdentityManager()
        
    def is_safe(self, box):
        bx_cx = ((box[0] + box[2]) / 2) / self.W
        bx_cy = ((box[1] + box[3]) / 2) / self.H
        for zone in SAFE_ZONES:
            zx1, zy1, zx2, zy2 = zone
            if zx1 <= bx_cx <= zx2 and zy1 <= bx_cy <= zy2:
                return True 
        return False

    def select_targets(self, frame, results, aim_mode):
        valid_targets = []
        
        boxes = results.boxes
        keypoints = results.keypoints
        
        if boxes is None or boxes.id is None:
            self.primary_target = None
            return []

        for i, box in enumerate(boxes):
            if int(box.cls[0]) != 0: continue 
            xyxy = box.xyxy[0].cpu().numpy()
            is_target_safe = self.is_safe(xyxy)
            
            yolo_id = int(box.id[0]) if box.id is not None else -1
            
            kps = keypoints[i].xy[0].cpu().numpy() if (keypoints is not None and len(keypoints) > i) else []
            
            pid = self.identity_manager.resolve_identity(frame, yolo_id, kps)
            
            x1, y1, x2, y2 = map(int, xyxy)
            aim_x, aim_y = (x1 + x2) // 2, (y1 + y2) // 2 
            
            def is_valid(kp_idx):
                return kps.shape[0] > kp_idx and kps[kp_idx][0] != 0

            if len(kps) > 0:
                if aim_mode == 1: # HEAD (Forehead)
                    if is_valid(1) and is_valid(2):
                        mid_x = (kps[1][0] + kps[2][0]) / 2
                        mid_y = (kps[1][1] + kps[2][1]) / 2
                        if is_valid(0):
                            vec_x, vec_y = mid_x - kps[0][0], mid_y - kps[0][1]
                            aim_x, aim_y = mid_x + (vec_x * 1.2), mid_y + (vec_y * 1.2)
                        else:
                            eye_dist = np.sqrt((kps[1][0] - kps[2][0])**2 + (kps[1][1] - kps[2][1])**2)
                            aim_x, aim_y = mid_x, mid_y - (eye_dist * 0.8)
                    elif is_valid(0):
                        h = y2 - y1
                        aim_x, aim_y = kps[0][0], kps[0][1] - (h * 0.15)
                    else:
                        h = y2 - y1
                        aim_y = y1 + (h * 0.08)
                elif aim_mode == 3: # NON_LETHAL
                    if is_valid(13) and is_valid(14):
                        aim_x, aim_y = (kps[13][0] + kps[14][0]) / 2, (kps[13][1] + kps[14][1]) / 2
                    elif is_valid(11) and is_valid(12):
                        mid_x, mid_y = (kps[11][0] + kps[12][0]) / 2, (kps[11][1] + kps[12][1]) / 2
                        aim_x, aim_y = mid_x, mid_y + (y2 - mid_y) * 0.5
                    else:
                        h = y2 - y1
                        aim_y = y1 + (h * 0.75)
                else: # UPPER_BODY
                    if is_valid(5) and is_valid(6):
                        aim_x, aim_y = (kps[5][0] + kps[6][0]) / 2, (kps[5][1] + kps[6][1]) / 2
                        aim_y += (y2 - y1) * 0.05 
                    else:
                        h = y2 - y1
                        aim_y = y1 + (h * 0.35)

            target_data = {
                'pid': pid,
                'yolo_id': yolo_id,
                'box': (x1, y1, x2, y2),
                'aim_point': (int(aim_x), int(aim_y)),
                'dist_to_center': np.sqrt(((x1+x2)//2 - self.cx)**2 + ((y1+y2)//2 - self.cy)**2),
                'locked': False,
                'safe': is_target_safe
            }
            valid_targets.append(target_data)

        best_t_data = None
        
        # In Manual Mode (Default), we ONLY lock if selected_pid is set
        if self.selected_pid is not None:
            for t in valid_targets:
                if t['pid'] == self.selected_pid:
                    # Allow selection even if safe? 
                    # Let's say yes for manual, but maybe warn?
                    # For now, yes, allow locking.
                    best_t_data = t
                    break
        
        # Auto-Fallthrough only if Manual Mode is OFF
        if best_t_data is None and not self.manual_mode:
            best_dist = float('inf')
            for t in valid_targets:
                # Ignore Safe targets for AUTO selection
                if t['safe']: continue
                
                if t['dist_to_center'] < best_dist:
                    best_dist = t['dist_to_center']
                    best_t_data = t
            
        if best_t_data:
            self.primary_target = best_t_data
            self.locked_ids.add(best_t_data['pid'])
            if self.manual_mode:
                self.selected_pid = best_t_data['pid']
        else:
            self.primary_target = None

        for ft in valid_targets:
            if ft['pid'] in self.locked_ids:
                ft['locked'] = True
                
        return valid_targets

def draw_hud(frame, turret, targets, primary, aim_mode_idx, manager):
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2
    
    for zone in SAFE_ZONES:
        x1, y1, x2, y2 = int(zone[0]*W), int(zone[1]*H), int(zone[2]*W), int(zone[3]*H)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 50), -1) 
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    color_cross = (0, 255, 0) if primary is None else (0, 0, 255)
    cv2.circle(frame, (cx, cy), 15, color_cross, 1)
    cv2.line(frame, (cx - 25, cy), (cx + 25, cy), color_cross, 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy + 25), color_cross, 1)

    for t in targets:
        x1, y1, x2, y2 = t['box']
        is_engaged = (t == primary)
        is_safe = t.get('safe', False)
        
        if is_safe:
            color = (0, 255, 0) # Green for Safe
            label_suffix = " (SAFE)"
        elif is_engaged:
            color = (0, 0, 255) # Red for Engaged
            label_suffix = ""
        else:
            color = (0, 255, 255) # Yellow for Tracked
            label_suffix = ""
            
        thick = 2 if is_engaged else 1
        
        if is_engaged: cv2.line(frame, (cx, cy), t['aim_point'], color, 1)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.putText(frame, f"P-ID:{t['pid']}{label_suffix}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if not is_safe:
            cv2.circle(frame, t['aim_point'], 4, (0, 0, 255), -1)

    panel_w, panel_h = 260, 280
    panel_x, panel_y = W - panel_w - 10, 10
    
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 0), 1)
    
    sx, sy = panel_x + 10, panel_y + 25
    line_h = 20
    
    cv2.putText(frame, "TURRET CONTROL SYS", (sx, sy), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (0, 255, 0), 1)
    sy += 30
    cv2.putText(frame, f"PAN : {turret.pan_angle:+.1f}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    sy += line_h
    cv2.putText(frame, f"TILT: {turret.tilt_angle:+.1f}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    sy += line_h
    
    mode_str = AIM_MODES.get(aim_mode_idx, "UNKNOWN")
    cv2.putText(frame, f"MODE: {mode_str}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    sy += line_h

    trk_mode = "MANUAL LOCK" if manager.manual_mode else "AUTO"
    if manager.selected_pid and not manager.manual_mode:
        trk_mode = "AUTO (LOCKED)"
        
    cv2.putText(frame, f"LOGIC: {trk_mode}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    sy += line_h
    
    db_count = len(manager.identity_manager.database)
    cv2.putText(frame, f"DB SIZE: {db_count} Identities", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    sy += line_h

    status = "ENGAGED" if primary else "SCANNING"
    cv2.putText(frame, f"STATUS : {status}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if primary else (0, 255, 0), 1)
    sy += line_h
    
    cv2.line(frame, (sx, sy), (sx+200, sy), (50, 50, 50), 1)
    sy += 15
    cv2.putText(frame, "ENGAGEMENT LIST", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    sy += 15
    
    sorted_targets = sorted(targets, key=lambda x: x['pid'])
    for i, t in enumerate(sorted_targets):
        if i > 4: 
            cv2.putText(frame, "...", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            break
        
        is_sel = (t == primary)
        is_safe = t.get('safe', False)
        prefix = ">>" if is_sel else "  "
        
        if is_safe:
            color_t = (0, 255, 0)
            status_txt = "[SAFE]"
        elif is_sel:
            color_t = (0, 255, 0)
            status_txt = "[ENGAGED]"
        else:
            color_t = (150, 150, 150)
            status_txt = ""
            
        text = f"{prefix} PID:{t['pid']} {status_txt}"
        cv2.putText(frame, text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_t, 1)
        sy += line_h

def main():
    print("[SYSTEM] Initializing Safe Turret System...")
    
    try:
        model = YOLO("yolo11n-pose.pt") 
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    turret = TurretController(kp=0.1, ki=0.01, kd=0.05)
    manager = TargetManager(W, H)
    aim_mode = 2 
    
    print(f"[SYSTEM] Cam: {W}x{H}")
    print("[CONTROL] Keys: '1-3'=Aim Mode, 'm'=Manual Lock, 'TAB'=Cycle Target")
    print("[CONTROL] 'r'=Reset Engagement (Un-stick)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0])
        
        targets = []
        primary = None
        
        if results:
            targets = manager.select_targets(frame, results[0], aim_mode)
            primary = manager.primary_target

        if primary:
            target_x, target_y = primary['aim_point']
            error_x = target_x - (W // 2)
            error_y = target_y - (H // 2)
            turret.update(error_x, error_y)
        
        draw_hud(frame, turret, targets, primary, aim_mode, manager)

        cv2.imshow("Safe Turret Sim", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'): 
            break
        elif key == ord('1'): aim_mode = 1
        elif key == ord('2'): aim_mode = 2
        elif key == ord('3'): aim_mode = 3
        elif key == ord('m'):
            manager.manual_mode = not manager.manual_mode
            if manager.primary_target:
                manager.selected_pid = manager.primary_target['pid']
        elif key == ord('r'):
             manager.selected_pid = None
             manager.manual_mode = True # Default to Manual when reset (or False? User said default is Manual)
             # User said: "default target mode should be manual"
             # So reset probably just clears selection but keeps manual mode?
        elif key == 9 or key == ord('\t'): #TAB
            if len(targets) > 0:
                pids = sorted([t['pid'] for t in targets])
                
                # If nothing selected, select first
                if manager.selected_pid is None:
                     manager.selected_pid = pids[0]
                elif manager.selected_pid in pids:
                    # Cycle
                    idx = pids.index(manager.selected_pid)
                    manager.selected_pid = pids[(idx + 1) % len(pids)]
                else:
                    # Current selection lost/not in frame, start at 0
                    manager.selected_pid = pids[0]
                
                # Cycling implies manual intent
                manager.manual_mode = True 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
