#!/usr/bin/env python3
"""
Aetherion
Improvements: Multi-threading, Better Detection, Alert System, Medical AI Integration
"""

import sys
import cv2
import math
import queue
import json
import threading
import time
import logging
import numpy as np
import pyttsx3
import sounddevice as sd
from ultralytics import YOLO
from vosk import Model, KaldiRecognizer
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer

# -----------------------------
# AUTOMATION CONFIGURATION
# -----------------------------
AUTOMATION_HOST = "0.0.0.0"
AUTOMATION_PORT = 9001
SERVER_IP = "172.16.5.237"   # Replace with your PC (server) IP
SERVER_PORT = 8000

# Dictionary of sensors
SENSORS = {
    "gas_sensor_1": "172.16.4.20",   # Replace with sensor laptop/ESP32 IP
    # "smoke_sensor_1": "192.168.1.11",
    # add more as needed...
}

from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QTextEdit, QFileDialog, QFrame, QSizePolicy, QStackedLayout, 
    QTabWidget, QProgressBar, QCheckBox, QSpinBox, QGroupBox, QGridLayout,
    QSlider, QComboBox, QMessageBox, QScrollArea, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt, QThread, QMutex, QWaitCondition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('space_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DetectionResult:
    """Structure to hold detection results"""
    def __init__(self):
        self.timestamp = datetime.now()
        self.objects_detected = []
        self.people_detected = []
        self.unconscious_people = []
        self.stress_levels = {}
        self.breathing_detected = False
        self.face_emotions = {}

class AlertLevel:
    """Alert severity levels"""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3

class EnhancedMedicalAI:
    """Enhanced Medical AI with structured reporting"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.conversation_history = []

class GeneralAI:
    """General AI assistant for voice commands and queries"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.conversation_history = []
    
    def get_response(self, query: str) -> str:
        """Get AI response for general queries"""
        try:
            if not self.api_key or self.api_key == 'your-api-key-here':
                return "AI assistant not configured. Please set your Groq API key in Settings."

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Try multiple compatible Groq chat models in order
            model_candidates = [
                "gemma2-9b-it",
                "llama-3.3-70b-versatile",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768"
            ]
            last_err = None
            for model_name in model_candidates:
                data = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are Aetherion, an advanced space assistant AI. You help astronauts with various tasks including system monitoring, emergency procedures, and general assistance. Keep responses concise and helpful."},
                        {"role": "user", "content": query}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
                response = requests.post(self.base_url, headers=headers, json=data, timeout=15)
                if response.status_code == 200:
                    ai_response = response.json()['choices'][0]['message']['content']
                    return ai_response
                else:
                    try:
                        last_err = response.json()
                    except Exception:
                        last_err = {"error": response.text}
                    continue
            logger.error(f"General AI API error after trying models {model_candidates}: {last_err}")
            return f"AI assistant temporarily unavailable. Error: {last_err}"
                
        except Exception as e:
            logger.error(f"General AI error: {e}")
            return f"AI assistant error: {str(e)}"
        
    def get_medical_assessment(self, symptoms: Dict, detection_data: DetectionResult) -> Dict:
        """Get structured medical assessment"""
        
        # Structure the symptom report
        symptom_report = self._format_symptoms(symptoms, detection_data)
        
        prompt = f"""
        SPACE MEDICAL EMERGENCY ASSESSMENT
        
        Situation: {symptom_report}
        
        Please provide a structured response with:
        1. IMMEDIATE_ACTIONS: 3-5 critical steps to take right now
        2. SEVERITY_ASSESSMENT: Scale 1-10 and reasoning
        3. POTENTIAL_CAUSES: Most likely medical causes
        4. MONITORING_INSTRUCTIONS: What to watch for
        5. MISSION_CONTROL_ALERT: Should mission control be contacted? (YES/NO)
        
        Keep responses concise but thorough. This is a real emergency situation.
        """
        
        try:
            if not self.api_key or self.api_key == 'your-api-key-here':
                raise RuntimeError("Missing or invalid GROQ API key. Set it in Settings.")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Try multiple compatible Groq chat models in order
            model_candidates = [
                "gemma2-9b-it",
                "llama-3.3-70b-versatile",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768"
            ]
            last_err = None
            for model_name in model_candidates:
                data = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a space medicine specialist providing emergency guidance to astronauts."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
                response = requests.post(self.base_url, headers=headers, json=data, timeout=20)
                if response.status_code == 200:
                    ai_response = response.json()['choices'][0]['message']['content']
                    return self._parse_structured_response(ai_response)
                else:
                    try:
                        last_err = response.json()
                    except Exception:
                        last_err = {"error": response.text}
                    # Try next model
                    continue
            logger.error(f"Medical AI API error after trying models {model_candidates}: {last_err}")
            return self._get_error_response(f"API error: {last_err}")
                
        except Exception as e:
            logger.error(f"Medical AI error: {e}")
            return self._get_error_response(str(e))
    
    def _format_symptoms(self, symptoms: Dict, detection_data: DetectionResult) -> str:
        """Format symptoms into structured report"""
        report = []
        
        if symptoms.get('unconscious'):
            report.append(f"UNCONSCIOUS ASTRONAUT DETECTED")
            report.append(f"- Duration: {symptoms.get('unconscious_duration', 'Unknown')}")
            report.append(f"- Head position: {'Slumped' if symptoms.get('head_slumped') else 'Normal'}")
            report.append(f"- Body position: {'Horizontal' if symptoms.get('torso_flat') else 'Upright'}")
        
        if symptoms.get('stress_detected'):
            report.append(f"HIGH STRESS DETECTED: {symptoms['stress_level']}/10")
        
        if not symptoms.get('breathing_detected'):
            report.append("NO BREATHING MOVEMENT DETECTED")
        
        if detection_data.face_emotions:
            emotions = ", ".join([f"{k}: {v}" for k, v in detection_data.face_emotions.items()])
            report.append(f"Facial emotions: {emotions}")
        
        return "\n".join(report)
    
    def _parse_structured_response(self, response: str) -> Dict:
        """Parse AI response into structured format"""
        sections = {
            'immediate_actions': [],
            'severity': 5,
            'causes': [],
            'monitoring': [],
            'contact_mission_control': False,
            'full_response': response
        }
        
        # Simple parsing - could be enhanced with more sophisticated NLP
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'IMMEDIATE_ACTIONS' in line.upper():
                current_section = 'immediate_actions'
            elif 'SEVERITY' in line.upper():
                current_section = 'severity'
            elif 'POTENTIAL_CAUSES' in line.upper():
                current_section = 'causes'
            elif 'MONITORING' in line.upper():
                current_section = 'monitoring'
            elif 'MISSION_CONTROL' in line.upper():
                if 'YES' in line.upper():
                    sections['contact_mission_control'] = True
            elif line and current_section and line.startswith(('-', '•', '1.', '2.')):
                sections[current_section].append(line)
        
        return sections
    
    def _get_error_response(self, message: str) -> Dict:
        """Structured error response when AI is unavailable. No hardcoded guidance."""
        return {
            'immediate_actions': [],
            'severity': 'Unknown',
            'causes': [],
            'monitoring': [],
            'contact_mission_control': False,
            'full_response': f"Medical AI unavailable: {message}"
        }

class EnhancedDetectionEngine(QThread):
    """Enhanced detection engine with multiple AI capabilities"""
    
    detection_ready = pyqtSignal(DetectionResult)
    alert_triggered = pyqtSignal(int, str)  # alert_level, message
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.mutex = QMutex()
        
        # Detection models
        self.pose_model = None
        self.pretrained_model = None
        self.custom_model = None
        self.face_cascade = None
        
        # Detection parameters
        self.unconscious_threshold = 3.0  # seconds
        self.stress_threshold = 7.0  # 1-10 scale
        self.breathing_check_interval = 5.0  # seconds
        
        # Tracking variables
        self.unconscious_start_time = None
        self.last_breathing_check = time.time()
        self.person_history = []  # Store recent detections for analysis
        self.prev_pose_keypoints = []  # previous frame keypoints per person index
        self.unconscious_score_threshold = 0.8
        self.static_movement_threshold = 1.5  # px average keypoint movement considered static
        self.unconscious_confirm_frames = 4  # temporal confirmation frames
        
        # Alert state tracking to prevent spam
        self.alert_states = {
            'unconscious_alerted': False,
            'stress_alerted': False,
            'last_alert_time': 0,
            'alert_cooldown': 10.0  # seconds between same type alerts
        }
        self.per_person_counters = {}  # person_index -> consecutive unconscious frames
        self.last_breathing_status = True  # cache last breathing state to avoid flicker
        
        # Movement tracking for unconscious detection (live feed only)
        self.movement_history = {}  # person_id -> list of recent keypoint positions
        self.movement_history_length = 10  # frames to track for movement analysis
        
        # Initialize models
        self.load_models()
    
    def load_models(self):
        """Load all AI models with error handling"""
        try:
            logger.info("Loading detection models...")
            
            # YOLO models
            self.pose_model = YOLO("yolov8s-pose.pt")
            self.pretrained_model = YOLO("yolov8n.pt")
            
            # Custom model (with fallback)
            custom_path = r"C:\Users\Yajat\runs\detect\train3\weights\best.pt"
            if os.path.exists(custom_path):
                self.custom_model = YOLO(custom_path)
            else:
                logger.warning("Custom model not found, using pretrained only")
                self.custom_model = self.pretrained_model
            
            # Face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def add_frame(self, frame):
        """Add frame to processing queue"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
    
    def run(self):
        """Main detection loop"""
        self.running = True
        logger.info("Detection engine started")
        
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    result = self.process_frame(frame)
                    self.detection_ready.emit(result)
                else:
                    self.msleep(50)  # Wait for frames
                    
            except Exception as e:
                logger.error(f"Detection error: {e}")
                self.msleep(1000)  # Wait before retrying
    
    def process_frame(self, frame) -> DetectionResult:
        """Process single frame with all detection algorithms"""
        result = DetectionResult()
        
        try:
            # Object detection
            result.objects_detected = self.detect_objects(frame)
            
            # Pose detection and unconsciousness analysis
            pose_results = self.pose_model(frame)[0]
            result.people_detected, result.unconscious_people = self.analyze_poses(pose_results)
            
            # Face and emotion detection
            result.face_emotions = self.detect_face_emotions(frame)
            
            # Breathing detection (every interval) with cached status to prevent warnings between checks
            if time.time() - self.last_breathing_check > self.breathing_check_interval:
                result.breathing_detected = self.detect_breathing(frame)
                self.last_breathing_status = result.breathing_detected
                self.last_breathing_check = time.time()
            else:
                result.breathing_detected = self.last_breathing_status
            
            # Check for alerts
            self.check_alerts(result)
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
        
        return result
    
    def detect_objects(self, frame) -> List[str]:
        """Enhanced object detection"""
        detected = []
        
        try:
            # Custom model detection (with class-specific filtering)
            if self.custom_model:
                results_c = self.custom_model(frame)[0]
                if results_c and results_c.boxes is not None:
                    for cls, conf, box in zip(results_c.boxes.cls, results_c.boxes.conf, results_c.boxes.xyxy):
                        name = self.custom_model.names[int(cls)]
                        conf_val = float(conf)
                        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                        if name.lower() in ["fire extinguisher", "extinguisher", "fire_extinguisher"]:
                            # Apply stricter gate for extinguishers
                            if conf_val < 0.6:
                                continue
                            if self._validate_extinguisher(frame, (x1, y1, x2, y2)):
                                detected.append("fire extinguisher")
                            else:
                                # If very red but fails shape checks, mark generic red object once
                                if self._red_ratio(frame, (x1, y1, x2, y2)) > 0.25:
                                    detected.append("red object")
                        else:
                            if conf_val > 0.4:
                                detected.append(name)
            
            # Pretrained model detection
            results_p = self.pretrained_model(frame)[0]
            pretrained_objects = [self.pretrained_model.names[int(cls)] 
                                for cls, conf in zip(results_p.boxes.cls, results_p.boxes.conf) 
                                if conf > 0.4]
            detected.extend(pretrained_objects)
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
        
        return list(set(detected))

    def _validate_extinguisher(self, frame, bbox: Tuple[int, int, int, int]) -> bool:
        """Heuristic to validate fire extinguisher candidates.
        Checks aspect ratio, area, red coverage, and vertical edge dominance.
        """
        try:
            x1, y1, x2, y2 = bbox
            h = max(1, y2 - y1)
            w = max(1, x2 - x1)
            area = h * w
            # Minimum size to avoid tiny red blobs
            if area < 800:
                return False
            aspect = h / w
            # Extinguishers are typically tall cylinders
            if aspect < 1.4 or aspect > 6.0:
                return False
            red_ratio = self._red_ratio(frame, bbox)
            if red_ratio < 0.2:
                return False
            # Vertical edge dominance
            vert_ratio = self._vertical_edge_ratio(frame, bbox)
            if vert_ratio < 0.55:
                return False
            return True
        except Exception as e:
            logger.error(f"Extinguisher validation error: {e}")
            return False

    def _red_ratio(self, frame, bbox: Tuple[int, int, int, int]) -> float:
        """Compute proportion of red pixels in the bbox using HSV thresholds."""
        x1, y1, x2, y2 = bbox
        roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # lower and upper red ranges
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        red_pixels = float(np.count_nonzero(mask))
        total = float(mask.size)
        return red_pixels / max(1.0, total)

    def _vertical_edge_ratio(self, frame, bbox: Tuple[int, int, int, int]) -> float:
        """Estimate dominance of vertical edges inside ROI using Sobel gradients."""
        x1, y1, x2, y2 = bbox
        roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magx = np.abs(gx)
        magy = np.abs(gy)
        sumx = float(np.sum(magx))
        sumy = float(np.sum(magy))
        denom = max(1e-6, sumx + sumy)
        return sumx / denom
    
    def analyze_poses(self, pose_results, is_live_feed: bool = True) -> Tuple[int, List[Dict]]:
        """Enhanced pose analysis with better unconsciousness detection"""
        people_count = 0
        unconscious_people = []
        
        try:
            for i, pose in enumerate(pose_results.keypoints):
                keypoints = pose.data[0].cpu().numpy()
                if len(keypoints) < 13 or np.isnan(keypoints).any():
                    continue
                
                people_count += 1
                xy = keypoints[:, :2]
                
                # Use different detection methods based on source
                if is_live_feed:
                    # Enhanced unconsciousness detection with movement analysis for live feed
                    is_unconscious, debug_info = self.enhanced_unconscious_detection(xy, i)
                else:
                    # Simple unconsciousness detection for image analysis (no movement tracking)
                    is_unconscious, debug_info = self.simple_unconscious_detection(xy)
                
                if is_unconscious:
                    unconscious_person = {
                        'person_id': i,
                        'debug_info': debug_info,
                        'confidence': debug_info.get('confidence', 0.5),
                        'timestamp': datetime.now()
                    }
                    unconscious_people.append(unconscious_person)
                
        except Exception as e:
            logger.error(f"Pose analysis error: {e}")
        
        return people_count, unconscious_people
    
    def enhanced_unconscious_detection(self, keypoints, person_id: int = 0) -> Tuple[bool, Dict]:
        """Enhanced unconsciousness detection with multiple criteria and movement analysis"""
        try:
            nose = keypoints[0]
            l_eye, r_eye = keypoints[1], keypoints[2]
            l_shoulder, r_shoulder = keypoints[5], keypoints[6]
            l_hip, r_hip = keypoints[11], keypoints[12]
            
            # Calculate key metrics
            shoulder_mid = np.mean([l_shoulder, r_shoulder], axis=0)
            hip_mid = np.mean([l_hip, r_hip], axis=0)
            
            # Head tilt analysis
            eye_line_angle = math.degrees(math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))
            head_tilt = abs(eye_line_angle)
            
            # Body position analysis
            torso_angle = math.degrees(math.atan2(hip_mid[1] - shoulder_mid[1], hip_mid[0] - shoulder_mid[0]))
            torso_horizontal = 75 < abs(torso_angle) < 105
            
            # Head position relative to body
            head_below_shoulders = nose[1] > shoulder_mid[1]
            head_drop_distance = nose[1] - shoulder_mid[1] if head_below_shoulders else 0
            
            # Shoulder levelness
            shoulder_level_diff = abs(l_shoulder[1] - r_shoulder[1])
            shoulders_level = shoulder_level_diff < 25
            
            # Movement analysis - track keypoint movement over time
            movement_score = self.analyze_movement(keypoints, person_id)
            
            # Scoring system (0-1)
            unconscious_score = 0
            
            # Head criteria (30% of score)
            if head_tilt > 45:
                unconscious_score += 0.15
            if head_below_shoulders:
                unconscious_score += 0.08
            if head_drop_distance > 30:
                unconscious_score += 0.07
            
            # Body criteria (30% of score)
            if torso_horizontal:
                unconscious_score += 0.2
            if shoulders_level:
                unconscious_score += 0.1
            
            # Movement criteria (40% of score) - now properly implemented
            # Low movement indicates unconsciousness
            unconscious_score += movement_score * 0.4
            
            debug_info = {
                'head_tilt': head_tilt,
                'torso_angle': torso_angle,
                'torso_horizontal': torso_horizontal,
                'head_below_shoulders': head_below_shoulders,
                'head_drop_distance': head_drop_distance,
                'shoulders_level': shoulders_level,
                'movement_score': movement_score,
                'unconscious_score': unconscious_score,
                'confidence': unconscious_score
            }
            
            return unconscious_score > 0.7, debug_info
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def analyze_movement(self, keypoints, person_id: int) -> float:
        """Analyze movement patterns to determine if person is unconscious"""
        try:
            # Store current keypoints in movement history
            if person_id not in self.movement_history:
                self.movement_history[person_id] = []
            
            # Add current keypoints to history
            self.movement_history[person_id].append(keypoints.copy())
            
            # Keep only recent history
            if len(self.movement_history[person_id]) > self.movement_history_length:
                self.movement_history[person_id].pop(0)
            
            # Need at least 3 frames to analyze movement
            if len(self.movement_history[person_id]) < 3:
                return 0.5  # Neutral score if not enough data
            
            # Calculate movement between consecutive frames
            movements = []
            history = self.movement_history[person_id]
            
            for i in range(1, len(history)):
                prev_keypoints = history[i-1]
                curr_keypoints = history[i]
                
                # Calculate average movement of key body points
                key_body_points = [0, 1, 2, 5, 6, 11, 12]  # nose, eyes, shoulders, hips
                frame_movement = 0
                valid_points = 0
                
                for point_idx in key_body_points:
                    if (point_idx < len(prev_keypoints) and point_idx < len(curr_keypoints) and
                        not np.isnan(prev_keypoints[point_idx]).any() and 
                        not np.isnan(curr_keypoints[point_idx]).any()):
                        
                        # Calculate Euclidean distance between points
                        prev_point = prev_keypoints[point_idx]
                        curr_point = curr_keypoints[point_idx]
                        distance = np.sqrt(np.sum((curr_point - prev_point) ** 2))
                        frame_movement += distance
                        valid_points += 1
                
                if valid_points > 0:
                    avg_movement = frame_movement / valid_points
                    movements.append(avg_movement)
            
            if not movements:
                return 0.5  # Neutral score if no valid movement data
            
            # Calculate average movement over recent frames
            avg_movement = np.mean(movements)
            
            # Calculate movement variance (consistency of movement)
            movement_variance = np.var(movements) if len(movements) > 1 else 0
            
            # Determine unconsciousness based on movement patterns
            # Low movement + low variance = likely unconscious
            # High movement + high variance = likely conscious
            
            # Normalize movement score (0-1, where 1 = very still/unconscious)
            movement_score = 0
            
            # Base score on average movement
            if avg_movement < 1.0:  # Very still
                movement_score += 0.6
            elif avg_movement < 2.0:  # Slightly moving
                movement_score += 0.4
            elif avg_movement < 4.0:  # Moderate movement
                movement_score += 0.2
            # High movement gets 0 score (person is conscious)
            
            # Adjust based on movement consistency
            if movement_variance < 0.5:  # Very consistent (likely unconscious)
                movement_score += 0.3
            elif movement_variance < 1.0:  # Somewhat consistent
                movement_score += 0.15
            # High variance gets no bonus (person is moving actively)
            
            # Ensure score is between 0 and 1
            movement_score = max(0, min(1, movement_score))
            
            return movement_score
            
        except Exception as e:
            logger.error(f"Movement analysis error: {e}")
            return 0.5  # Neutral score on error
    
    def simple_unconscious_detection(self, keypoints) -> Tuple[bool, Dict]:
        """Simple unconscious detection for image analysis (no movement tracking)"""
        try:
            nose = keypoints[0]
            l_eye, r_eye = keypoints[1], keypoints[2]
            l_shoulder, r_shoulder = keypoints[5], keypoints[6]
            l_hip, r_hip = keypoints[11], keypoints[12]
            
            # Calculate key metrics
            shoulder_mid = np.mean([l_shoulder, r_shoulder], axis=0)
            hip_mid = np.mean([l_hip, r_hip], axis=0)
            
            # Head tilt analysis
            eye_line_angle = math.degrees(math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))
            head_tilt = abs(eye_line_angle)
            
            # Body position analysis
            torso_angle = math.degrees(math.atan2(hip_mid[1] - shoulder_mid[1], hip_mid[0] - shoulder_mid[0]))
            torso_horizontal = 75 < abs(torso_angle) < 105
            
            # Head position relative to body
            head_below_shoulders = nose[1] > shoulder_mid[1]
            head_drop_distance = nose[1] - shoulder_mid[1] if head_below_shoulders else 0
            
            # Shoulder levelness
            shoulder_level_diff = abs(l_shoulder[1] - r_shoulder[1])
            shoulders_level = shoulder_level_diff < 25
            
            # Simple scoring system (0-1) - no movement analysis
            unconscious_score = 0
            
            # Head criteria (50% of score)
            if head_tilt > 45:
                unconscious_score += 0.25
            if head_below_shoulders:
                unconscious_score += 0.15
            if head_drop_distance > 30:
                unconscious_score += 0.1
            
            # Body criteria (50% of score)
            if torso_horizontal:
                unconscious_score += 0.3
            if shoulders_level:
                unconscious_score += 0.2
            
            debug_info = {
                'head_tilt': head_tilt,
                'torso_angle': torso_angle,
                'torso_horizontal': torso_horizontal,
                'head_below_shoulders': head_below_shoulders,
                'head_drop_distance': head_drop_distance,
                'shoulders_level': shoulders_level,
                'unconscious_score': unconscious_score,
                'confidence': unconscious_score
            }
            
            return unconscious_score > 0.6, debug_info  # Lower threshold for images
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def detect_face_emotions(self, frame) -> Dict[str, float]:
        """Basic emotion detection using facial features"""
        emotions = {}
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Simple stress detection based on face analysis
                # This is a placeholder - could be enhanced with proper emotion recognition models
                face_variance = np.var(face_roi)
                stress_level = min(10, face_variance / 1000)
                
                emotions[f'face_{len(emotions)}'] = {
                    'stress_level': stress_level,
                    'area': w * h
                }
                
        except Exception as e:
            logger.error(f"Face emotion detection error: {e}")
        
        return emotions
    
    def detect_breathing(self, frame) -> bool:
        """Detect subtle breathing movements - lightweight temporal heuristic."""
        try:
            # Use chest/torso ROI derived from average of shoulder and hip keypoints if available
            # If not available (no pose), keep last known status
            if not self.prev_pose_keypoints:
                return self.last_breathing_status
            # Combine all persons' torso regions and measure tiny luminance variance changes
            # as a proxy for motion. This is a coarse heuristic.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motions = []
            for xy in self.prev_pose_keypoints:
                if xy.shape[0] < 13:
                    continue
                l_shoulder, r_shoulder = xy[5], xy[6]
                l_hip, r_hip = xy[11], xy[12]
                shoulder_mid = np.mean([l_shoulder, r_shoulder], axis=0)
                hip_mid = np.mean([l_hip, r_hip], axis=0)
                # Define torso ROI
                min_x = int(max(0, min(shoulder_mid[0], hip_mid[0]) - 20))
                max_x = int(min(gray.shape[1]-1, max(shoulder_mid[0], hip_mid[0]) + 20))
                min_y = int(max(0, min(shoulder_mid[1], hip_mid[1]) - 10))
                max_y = int(min(gray.shape[0]-1, max(shoulder_mid[1], hip_mid[1]) + 10))
                if max_x <= min_x or max_y <= min_y:
                    continue
                roi = gray[min_y:max_y, min_x:max_x]
                if roi.size == 0:
                    continue
                # Simple motion estimate via Laplacian variance (edge fluctuation)
                motions.append(float(cv2.Laplacian(roi, cv2.CV_32F).var()))
            if not motions:
                return self.last_breathing_status
            avg_motion = np.mean(motions)
            # Threshold chosen empirically to detect slight movements
            return avg_motion > 2.0
        except Exception:
            return self.last_breathing_status
    
    def check_alerts(self, result: DetectionResult):
        """Check for alert conditions with spam prevention"""
        try:
            current_time = time.time()
            
            # Check for unconscious people
            if result.unconscious_people:
                if self.unconscious_start_time is None:
                    self.unconscious_start_time = current_time
                elif (current_time - self.unconscious_start_time > self.unconscious_threshold and 
                      not self.alert_states['unconscious_alerted'] and
                      current_time - self.alert_states['last_alert_time'] > self.alert_states['alert_cooldown']):
                    # Trigger emergency alert only once
                    message = f"EMERGENCY: Astronaut unconscious for {current_time - self.unconscious_start_time:.1f} seconds"
                    self.alert_triggered.emit(AlertLevel.EMERGENCY, message)
                    self.alert_states['unconscious_alerted'] = True
                    self.alert_states['last_alert_time'] = current_time
            else:
                # Reset unconscious alert state when no unconscious people detected
                self.unconscious_start_time = None
                self.alert_states['unconscious_alerted'] = False
            
            # Check stress levels with spam prevention
            for face_id, emotions in result.face_emotions.items():
                if (emotions.get('stress_level', 0) > self.stress_threshold and 
                    not self.alert_states['stress_alerted'] and
                    current_time - self.alert_states['last_alert_time'] > self.alert_states['alert_cooldown']):
                    message = f"WARNING: High stress detected (Level: {emotions['stress_level']:.1f}/10)"
                    self.alert_triggered.emit(AlertLevel.WARNING, message)
                    self.alert_states['stress_alerted'] = True
                    self.alert_states['last_alert_time'] = current_time
                    break  # Only alert once per check cycle
            
            # Reset stress alert if stress level drops
            max_stress = max([emotions.get('stress_level', 0) 
                            for emotions in result.face_emotions.values()], default=0)
            if max_stress <= self.stress_threshold:
                self.alert_states['stress_alerted'] = False
            
        except Exception as e:
            logger.error(f"Alert checking error: {e}")
    
    def stop(self):
        """Stop the detection engine"""
        self.running = False
        self.wait()

class EnhancedVoiceAssistant(QThread):
    """Enhanced voice assistant with wake word and continuous listening"""
    
    command_received = pyqtSignal(str)
    listening_status = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.wake_words = ["houston", "assistant", "help"]
        self.listening_for_command = False
        self.listening_started_at = 0.0
        self.listening_timeout_sec = 6.0
        self.wake_cooldown_sec = 2.0
        self.last_wake_time = 0.0
        # Audio configuration
        self.input_device_index = None
        self.audio_gain = 1.0
        self.tts_rate = 170
        self.tts_volume = 1.0
        self.allowed_commands = [
            "start monitoring",
            "stop monitoring",
            "detect objects",
            "check status",
            "emergency alert",
            "emergency",
            "help",
            "get medical help",
            "medical"
        ]
        
        # Audio components
        self.audio_queue = queue.Queue()
        self.vosk_model = None
        self.recognizer = None
        self.grammar_recognizer = None
        self._grammar_json = None
        self.tts_engine = None
        
        # Initialize components
        self.setup_audio()
    
    def setup_audio(self):
        """Setup audio components"""
        try:
            # Vosk setup
            # Prefer larger, more accurate model if available
            large_model = "model/vosk-model-en-us-0.22"
            small_model = "model/vosk-model-small-en-us-0.15"
            model_path = large_model if os.path.exists(large_model) else small_model
            if os.path.exists(model_path):
                logger.info(f"Loading Vosk model: {model_path}")
                self.vosk_model = Model(model_path)
                self.recognizer = KaldiRecognizer(self.vosk_model, 16000)
            else:
                logger.error("Vosk model not found")
            
            # TTS setup
            try:
                self.tts_engine = pyttsx3.init('sapi5')
            except Exception:
                self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', self.tts_rate)
            self.tts_engine.setProperty('volume', self.tts_volume)

            # Build command grammar to improve accuracy when listening for commands
            self._grammar_json = self._build_command_grammar()
            
        except Exception as e:
            logger.error(f"Audio setup error: {e}")
    
    def audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if self.running:
            try:
                arr = np.frombuffer(indata, dtype=np.int16)
                if self.audio_gain != 1.0:
                    arr = np.clip(arr.astype(np.float32) * float(self.audio_gain), -32768, 32767).astype(np.int16)
                self.audio_queue.put(arr.tobytes())
            except Exception:
                self.audio_queue.put(bytes(indata))
    
    def run(self):
        """Main voice processing loop"""
        self.running = True
        logger.info("Voice assistant started")
        
        try:
            stream_kwargs = dict(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=self.audio_callback)
            if self.input_device_index is not None:
                stream_kwargs['device'] = self.input_device_index
            with sd.RawInputStream(**stream_kwargs):
                while self.running:
                    self.process_audio()
                    
        except Exception as e:
            logger.error(f"Voice assistant error: {e}")
    
    def process_audio(self):
        """Process audio for wake word and commands"""
        try:
            if not self.audio_queue.empty():
                data = self.audio_queue.get()
                
                # Choose recognizer based on state: grammar for commands, general for wake word
                active_recognizer = self.grammar_recognizer if self.listening_for_command and self.grammar_recognizer else self.recognizer
                if active_recognizer and active_recognizer.AcceptWaveform(data):
                    result = json.loads(active_recognizer.Result())
                    text = result.get("text", "").lower()
                    
                    if text:
                        # Print final recognized text to terminal
                        try:
                            logger.info(f"HEARD (final): {text}")
                            print(f"[heard final] {text}")
                        except Exception:
                            pass
                        # Manage timeout while listening for a command
                        if self.listening_for_command and (time.time() - self.listening_started_at) > self.listening_timeout_sec:
                            self.listening_for_command = False
                            self.listening_status.emit(False)
                        
                        # Wake word handling (multiple wake words) with cooldown
                        if not self.listening_for_command and any(w in text for w in self.wake_words):
                            if (time.time() - self.last_wake_time) < self.wake_cooldown_sec:
                                return
                            # Accept only if text is exactly the wake word or begins with it
                            tokens = text.split()
                            if len(tokens) == 1 and tokens[0] in self.wake_words:
                                self.listening_for_command = True
                                self.listening_started_at = time.time()
                                self.last_wake_time = self.listening_started_at
                                self.listening_status.emit(True)
                                self.speak("Yes, how can I help?")
                                # Reset and use grammar-based recognizer for better command accuracy
                                try:
                                    if self._grammar_json and self.vosk_model:
                                        self.grammar_recognizer = KaldiRecognizer(self.vosk_model, 16000, self._grammar_json)
                                        logger.info("Switched to grammar recognizer for command listening")
                                except Exception as _:
                                    self.grammar_recognizer = None
                            elif tokens[0] in self.wake_words and len(tokens) > 1:
                                # Treat remaining phrase as the command
                                command_text = " ".join(tokens[1:]).strip()
                                normalized, confidence = self._classify_command(command_text)
                                if normalized and confidence >= 0.7:
                                    logger.info(f"COMMAND (from wake+command): {normalized}")
                                    self.command_received.emit(normalized)
                                else:
                                    self.speak("Please repeat the command clearly.")
                            # If wake word appears elsewhere, ignore to reduce false triggers
                        
                        elif self.listening_for_command:
                            # Strip wake word if user says it again
                            for w in self.wake_words:
                                if text.startswith(w):
                                    text = text[len(w):].strip()
                                    break
                            normalized, confidence = self._classify_command(text)
                            if normalized and confidence >= 0.7:
                                logger.info(f"COMMAND: {normalized}")
                                self.command_received.emit(normalized)
                            else:
                                # Ignore noise/unknown commands
                                pass
                            self.listening_for_command = False
                            self.listening_status.emit(False)
                            # Return to general recognizer after command phase
                            self.grammar_recognizer = None
                        else:
                            # No wake word path: allow direct commands with higher confidence to avoid false triggers
                            normalized, confidence = self._classify_command(text)
                            if normalized and confidence >= 0.9:
                                logger.info(f"COMMAND (no wake word): {normalized}")
                                self.command_received.emit(normalized)
                elif active_recognizer:
                    # Show partial recognition continuously
                    try:
                        partial_json = active_recognizer.PartialResult()
                        partial_text = json.loads(partial_json).get("partial", "").strip().lower()
                        if partial_text:
                            logger.info(f"hearing (partial): {partial_text}")
                            print(f"[hearing] {partial_text}", end='\r', flush=True)
                    except Exception:
                        pass
            else:
                self.msleep(100)
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")

    def _build_command_grammar(self) -> str:
        """Build a Vosk grammar (JSON array of phrases) to bias recognition toward allowed commands and wake words."""
        try:
            phrases = []
            # Wake words alone and with commas are common
            for w in self.wake_words:
                phrases.append(w)
                phrases.append(f"{w} start monitoring")
                phrases.append(f"{w} stop monitoring")
                phrases.append(f"{w} detect objects")
                phrases.append(f"{w} check status")
                phrases.append(f"{w} emergency alert")
                phrases.append(f"{w} emergency")
                phrases.append(f"{w} help")
                phrases.append(f"{w} get medical help")
                phrases.append(f"{w} medical")
            # Also allow bare commands
            phrases.extend(self.allowed_commands)
            # Deduplicate
            phrases = sorted(list(set(phrases)))
            return json.dumps(phrases)
        except Exception as _:
            return json.dumps(self.allowed_commands)

    def _classify_command(self, text: str) -> Tuple[Optional[str], float]:
        """Classify command via pattern matching; returns (command, confidence)."""
        t = text.strip().lower()
        if not t:
            return None, 0.0
        candidates: List[Tuple[str, float]] = []
        # direct matches high confidence
        for cmd in self.allowed_commands:
            if cmd == t:
                candidates.append((cmd, 1.0))
            elif cmd in t:
                candidates.append((cmd, 0.8))
        # pattern-based
        if "start" in t and "monitor" in t:
            candidates.append(("start monitoring", 0.85))
        if ("stop" in t or "end" in t) and "monitor" in t:
            candidates.append(("stop monitoring", 0.85))
        if ("object" in t) and ("detect" in t or "detection" in t):
            candidates.append(("detect objects", 0.8))
        if "status" in t or "system" in t:
            candidates.append(("check status", 0.75))
        if "emergency" in t or ("need" in t and "help" in t):
            candidates.append(("emergency", 0.9))
        if "medical" in t or ("doctor" in t and "help" in t):
            candidates.append(("medical", 0.8))
        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]
    
    def speak(self, text: str):
        """Text-to-speech output"""
        try:
            if self.tts_engine:
                self.tts_engine.setProperty('rate', self.tts_rate)
                self.tts_engine.setProperty('volume', self.tts_volume)
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS error: {e}")

    def list_audio_devices(self) -> List[str]:
        """Return a list of input device names."""
        try:
            devices = sd.query_devices()
            names = []
            for idx, d in enumerate(devices):
                if d.get('max_input_channels', 0) > 0:
                    names.append(f"{idx}: {d.get('name','Unknown')}")
            return names
        except Exception as e:
            logger.error(f"List audio devices error: {e}")
            return []

    def set_input_device(self, index: Optional[int]):
        self.input_device_index = index

    def set_tts_config(self, rate: int, volume: float):
        self.tts_rate = max(120, min(220, int(rate)))
        self.tts_volume = max(0.0, min(1.0, float(volume)))
    
    def stop(self):
        """Stop voice assistant"""
        self.running = False
        self.wait()

class AlertSystem(QObject):
    """Centralized alert management system"""
    
    alert_display = pyqtSignal(int, str, dict)  # level, message, details
    
    def __init__(self):
        super().__init__()
        self.active_alerts = {}
        self.alert_history = []
        self.medical_ai = None
    
    def set_medical_ai(self, medical_ai: EnhancedMedicalAI):
        """Set medical AI reference"""
        self.medical_ai = medical_ai
    
    def handle_alert(self, level: int, message: str, detection_result: Optional[DetectionResult] = None):
        """Handle incoming alerts"""
        alert_id = f"{level}_{hash(message)}_{int(time.time())}"
        
        alert_data = {
            'id': alert_id,
            'level': level,
            'message': message,
            'timestamp': datetime.now(),
            'detection_result': detection_result,
            'resolved': False
        }
        
        self.active_alerts[alert_id] = alert_data
        self.alert_history.append(alert_data)
        
        # Handle based on severity
        if level >= AlertLevel.CRITICAL:
            self.handle_critical_alert(alert_data)
        
        self.alert_display.emit(level, message, alert_data)
    
    def handle_critical_alert(self, alert_data: Dict):
        """Handle critical/emergency alerts"""
        try:
            if self.medical_ai and alert_data['detection_result']:
                # Prepare symptoms for medical AI
                symptoms = self.extract_symptoms(alert_data['detection_result'])
                
                # Get medical assessment
                medical_response = self.medical_ai.get_medical_assessment(symptoms, alert_data['detection_result'])
                alert_data['medical_assessment'] = medical_response
                
                # Update alert with medical guidance
                enhanced_message = f"{alert_data['message']}\n\nIMMEDIATE ACTION REQUIRED:\n"
                for action in medical_response['immediate_actions'][:3]:  # Show top 3 actions
                    enhanced_message += f"• {action}\n"
                
                self.alert_display.emit(alert_data['level'], enhanced_message, alert_data)
                
        except Exception as e:
            logger.error(f"Critical alert handling error: {e}")
    
    def extract_symptoms(self, detection_result: DetectionResult) -> Dict:
        """Extract symptoms from detection result"""
        symptoms = {}
        
        if detection_result.unconscious_people:
            symptoms['unconscious'] = True
            symptoms['unconscious_count'] = len(detection_result.unconscious_people)
            
            # Get details from first unconscious person
            first_person = detection_result.unconscious_people[0]
            debug_info = first_person.get('debug_info', {})
            
            symptoms.update({
                'head_slumped': debug_info.get('head_below_shoulders', False),
                'torso_flat': debug_info.get('torso_horizontal', False),
                'confidence': debug_info.get('confidence', 0)
            })
        
        if detection_result.face_emotions:
            max_stress = max([emotions.get('stress_level', 0) 
                            for emotions in detection_result.face_emotions.values()])
            if max_stress > 7:
                symptoms['stress_detected'] = True
                symptoms['stress_level'] = max_stress
        
        symptoms['breathing_detected'] = detection_result.breathing_detected
        
        return symptoms

# -----------------------------
# AUTOMATION CLASSES
# -----------------------------

class AutomationHandler(BaseHTTPRequestHandler):
    """HTTP handler for receiving sensor data and automation requests"""
    
    def __init__(self, app_instance, *args, **kwargs):
        self.app = app_instance
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        """Handle POST requests from sensors"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            sensor_name = data.get("sensor", "unknown_sensor")
            sensor_status = data.get("status", "unknown_status")
            client_ip = self.client_address[0]

            print(f"📡 Received from {sensor_name} ({client_ip}): {sensor_status}")

            # Handle gas leak detection
            if sensor_status == "leak_detected":
                warning_msg = f"Warning! Gas leak detected from {sensor_name} at {client_ip}. Sending alert to server!"
                print("⚠", warning_msg)
                
                # Create alert data
                alert_data = {
                    'type': 'gas_leak',
                    'severity': 'critical',
                    'message': warning_msg,
                    'sensor_name': sensor_name,
                    'sensor_ip': client_ip,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Emit signal to main app
                if hasattr(self.app, 'gas_leak_detected'):
                    self.app.gas_leak_detected.emit(alert_data)
                
                # Forward to server
                payload = {"action": "open_browser", "query": f"Gas leak detected from {sensor_name}"}
                try:
                    r = requests.post(f"http://{SERVER_IP}:{SERVER_PORT}", json=payload, timeout=5)
                    print("✅ Server response:", r.json())
                except Exception as e:
                    print("❌ Error sending to server:", e)

            # Respond to sensor
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            
        except Exception as e:
            print(f"❌ Error handling sensor data: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Error")

class TaskExecutor(QObject):
    """Handle task execution and server communication"""
    
    task_completed = pyqtSignal(dict)
    task_failed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
    
    def execute_task(self, prompt: str):
        """Execute a task by sending prompt to server"""
        try:
            print(f"🚀 Sending prompt to server: {prompt}")
            
            payload = {
                "action": "execute_prompt", 
                "prompt": prompt
            }
            
            r = requests.post(f"http://{SERVER_IP}:{SERVER_PORT}", json=payload, timeout=30)
            response = r.json()
            
            result = {
                'status': response.get("status", "unknown"),
                'message': response.get("message", "Task completed"),
                'prompt': prompt,
                'timestamp': datetime.now().isoformat()
            }
            
            print("✅ Server response:", result['message'])
            self.task_completed.emit(result)
            
        except Exception as e:
            error_msg = f"Error communicating with server: {e}"
            print("❌", error_msg)
            self.task_failed.emit(error_msg)

class EnhancedSpaceAssistantApp(QWidget):
    """Main application with enhanced UI and functionality"""
    
    # Signals
    gas_leak_detected = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aetherion")
        self.setStyleSheet("""
            QWidget { background-color: #1e1e2f; color: #ffffff; }
            QTabWidget::pane { border: 1px solid #3f3f5f; background-color: #2c2f4a; }
            QTabBar::tab { 
                background-color: #3f3f5f; 
                padding: 8px 16px; 
                margin-right: 2px; 
                border-top-left-radius: 4px; 
                border-top-right-radius: 4px; 
            }
            QTabBar::tab:selected { background-color: #00ffcc; color: #000; }
            QPushButton { 
                background-color: #00d46a; 
                color: white; 
                padding: 10px; 
                font-weight: bold; 
                border-radius: 6px; 
            }
            QPushButton:hover { background-color: #00e676; }
            QPushButton:pressed { background-color: #00c853; }
            QTextEdit { 
                background-color: #2c2f4a; 
                color: #00ff9f; 
                border: 1px solid #00ffcc; 
                border-radius: 6px; 
                padding: 8px; 
            }
            QLabel { color: #ffffff; }
        """)
        self.resize(1400, 900)
        
        # Core components
        self.detection_engine = None
        self.voice_assistant = None
        self.alert_system = AlertSystem()
        self.medical_ai = None
        self.general_ai = None
        
        # Automation components
        self.task_executor = TaskExecutor()
        self.automation_server = None
        self.automation_server_thread = None
        
        # Camera
        self.camera = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera)
        
        # UI setup
        self.setup_ui()
        self.setup_connections()
        
        # Initialize systems
        self.initialize_systems()
    
    def setup_ui(self):
        """Setup the enhanced UI with tabs"""
        main_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Aetherion")
        title_label.setFont(QFont("Orbitron", 20, QFont.Bold))
        title_label.setStyleSheet("color: #00ffcc; margin: 10px; text-align: center;")
        title_label.setAlignment(Qt.AlignCenter)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Live Feed Tab
        self.setup_live_feed_tab()
        
        # Voice Assistant Tab
        self.setup_voice_tab()
        
        # Image Analysis Tab
        self.setup_image_tab()
        
        # Alerts & Medical Tab
        self.setup_alerts_tab()
        
        # Automation Tab
        self.setup_automation_tab()
        
        # Settings Tab
        self.setup_settings_tab()
        
        # Log Tab
        self.setup_log_tab()
        
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.tab_widget)
        
        self.setLayout(main_layout)
    
    def setup_live_feed_tab(self):
        """Setup live camera feed tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("🎥 Start Live Feed")
        self.start_button.clicked.connect(self.start_live_feed)
        
        self.stop_button = QPushButton("⏹️ Stop Feed")
        self.stop_button.clicked.connect(self.stop_live_feed)
        self.stop_button.setEnabled(False)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setFixedHeight(500)
        self.camera_label.setStyleSheet("border: 2px solid #3f3f5f; border-radius: 8px; background-color: #12121c;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("📷 Camera feed will appear here")
        
        # Status display
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        
        layout.addLayout(control_layout)
        layout.addWidget(self.camera_label)
        layout.addWidget(self.status_text)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "🎥 Live Feed")
    
    def setup_voice_tab(self):
        """Setup AI Chatbot Assistant tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Header with status and controls
        header_layout = QHBoxLayout()
        
        # Chatbot status
        self.chatbot_status_label = QLabel("🤖 AI Assistant: Ready")
        self.chatbot_status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.chatbot_status_label.setStyleSheet("color: #00ff9f; margin: 10px;")
        
        # Voice control buttons
        self.start_voice_button = QPushButton("🎙️ Enable Voice")
        self.start_voice_button.clicked.connect(self.start_voice_assistant)
        self.start_voice_button.setStyleSheet("background-color: #00d46a; padding: 8px 16px;")
        
        self.stop_voice_button = QPushButton("⏹️ Disable Voice")
        self.stop_voice_button.clicked.connect(self.stop_voice_assistant)
        self.stop_voice_button.setEnabled(False)
        self.stop_voice_button.setStyleSheet("background-color: #ff6b6b; padding: 8px 16px;")
        
        # Clear chat button
        self.clear_chat_button = QPushButton("🗑️ Clear Chat")
        self.clear_chat_button.clicked.connect(self.clear_chat_history)
        self.clear_chat_button.setStyleSheet("background-color: #ffaa00; padding: 8px 16px;")
        
        header_layout.addWidget(self.chatbot_status_label)
        header_layout.addStretch()
        header_layout.addWidget(self.start_voice_button)
        header_layout.addWidget(self.stop_voice_button)
        header_layout.addWidget(self.clear_chat_button)
        
        # Welcome message
        welcome_label = QLabel("💬 Chat with your AI Space Assistant - Type messages or use voice commands")
        welcome_label.setStyleSheet("color: #00ffcc; font-style: italic; margin: 5px; font-size: 12px;")
        welcome_label.setAlignment(Qt.AlignCenter)
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a2e;
                color: #ffffff;
                border: 2px solid #3f3f5f;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                line-height: 1.4;
            }
        """)
        
        # Add welcome message to chat
        self.add_chat_message("assistant", "👋 Hello! I'm your AI Space Assistant. I can help you with:\n• System monitoring and control\n• Medical guidance and alerts\n• Space mission information\n• General questions and commands\n\nType your message below or use voice commands!")
        
        # Input area
        input_layout = QHBoxLayout()
        
        # Text input
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(60)
        self.chat_input.setPlaceholderText("Type your message here... (Press Ctrl+Enter to send)")
        self.chat_input.setStyleSheet("""
            QTextEdit {
                background-color: #2c2f4a;
                color: #ffffff;
                border: 2px solid #00ffcc;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }
            QTextEdit:focus {
                border-color: #00ff9f;
            }
        """)
        
        # Send button
        self.send_button = QPushButton("📤 Send")
        self.send_button.clicked.connect(self.send_chat_message)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #00ffcc;
                color: #000;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #00ff9f;
            }
            QPushButton:pressed {
                background-color: #00e676;
            }
        """)
        
        # Voice input button
        self.voice_input_button = QPushButton("🎤 Voice")
        self.voice_input_button.clicked.connect(self.start_voice_input)
        self.voice_input_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #ff5252;
            }
            QPushButton:pressed {
                background-color: #f44336;
            }
        """)
        
        input_layout.addWidget(self.chat_input, 1)
        input_layout.addWidget(self.voice_input_button)
        input_layout.addWidget(self.send_button)
        
        # Quick commands
        quick_commands_group = QGroupBox("🚀 Quick Commands")
        quick_commands_layout = QGridLayout()
        
        quick_commands = [
            ("Start Monitoring", "start monitoring"),
            ("Check Status", "check status"),
            ("Emergency Alert", "emergency alert"),
            ("Medical Help", "get medical help"),
            ("Space Info", "tell me about space"),
            ("Weather", "what's the weather like?")
        ]
        
        for i, (label, command) in enumerate(quick_commands):
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, cmd=command: self.send_quick_command(cmd))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3f3f5f;
                    color: #ffffff;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #00ffcc;
                    color: #000;
                }
            """)
            quick_commands_layout.addWidget(btn, i // 3, i % 3)
        
        quick_commands_group.setLayout(quick_commands_layout)
        quick_commands_group.setMaximumHeight(120)
        
        # Add all widgets to layout
        layout.addLayout(header_layout)
        layout.addWidget(welcome_label)
        layout.addWidget(self.chat_display)
        layout.addLayout(input_layout)
        layout.addWidget(quick_commands_group)
        
        # Connect Enter key to send message
        self.chat_input.keyPressEvent = self.chat_input_key_press
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "💬 AI Assistant")
    
    def add_chat_message(self, sender: str, message: str):
        """Add a message to the chat display"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if sender == "user":
                message_html = f"""
                <div style="margin: 10px 0; text-align: right;">
                    <div style="background-color: #00ffcc; color: #000; padding: 8px 12px; border-radius: 12px; display: inline-block; max-width: 70%; word-wrap: break-word;">
                        <strong>You</strong> <span style="font-size: 10px; opacity: 0.7;">[{timestamp}]</span><br>
                        {message}
                    </div>
                </div>
                """
            else:  # assistant
                message_html = f"""
                <div style="margin: 10px 0; text-align: left;">
                    <div style="background-color: #3f3f5f; color: #fff; padding: 8px 12px; border-radius: 12px; display: inline-block; max-width: 70%; word-wrap: break-word;">
                        <strong>🤖 AI Assistant</strong> <span style="font-size: 10px; opacity: 0.7;">[{timestamp}]</span><br>
                        {message}
                    </div>
                </div>
                """
            
            self.chat_display.append(message_html)
            
            # Auto-scroll to bottom
            cursor = self.chat_display.textCursor()
            cursor.movePosition(cursor.End)
            self.chat_display.setTextCursor(cursor)
            
            # Always speak AI assistant messages
            if sender == "assistant" and self.voice_assistant:
                try:
                    # Clean the message for speech (remove HTML and emojis)
                    clean_message = self.clean_message_for_speech(message)
                    self.voice_assistant.speak(clean_message)
                except Exception as e:
                    logger.error(f"Error speaking message: {e}")
            
        except Exception as e:
            logger.error(f"Error adding chat message: {e}")
    
    def clean_message_for_speech(self, message: str) -> str:
        """Clean message for better speech output"""
        try:
            # Remove HTML tags
            import re
            clean_message = re.sub(r'<[^>]+>', '', message)
            
            # Remove or replace emojis with text
            emoji_replacements = {
                '👋': 'Hello',
                '🤖': 'AI Assistant',
                '🎙️': 'Voice',
                '🚀': 'Rocket',
                '✅': 'Check',
                '⚠️': 'Warning',
                '🚨': 'Alert',
                '🏥': 'Medical',
                '🌤️': 'Weather',
                '🕐': 'Time',
                '📹': 'Camera',
                '🔍': 'Detection',
                '📡': 'Sensor',
                '⚙️': 'System',
                '🌌': 'Space',
                '👨‍🚀': 'Astronaut',
                '🛰️': 'Space Station',
                '❌': 'Error',
                '⏹️': 'Stop',
                'ℹ️': 'Info',
                '📤': 'Send',
                '🗑️': 'Clear',
                '💬': 'Chat'
            }
            
            for emoji, replacement in emoji_replacements.items():
                clean_message = clean_message.replace(emoji, replacement)
            
            # Remove extra whitespace
            clean_message = ' '.join(clean_message.split())
            
            return clean_message
            
        except Exception as e:
            logger.error(f"Error cleaning message for speech: {e}")
            return message
    
    def chat_input_key_press(self, event):
        """Handle key press events in chat input"""
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
            self.send_chat_message()
        else:
            # Call the original keyPressEvent for normal behavior
            QTextEdit.keyPressEvent(self.chat_input, event)
    
    def send_chat_message(self):
        """Send a message from the chat input"""
        try:
            message = self.chat_input.toPlainText().strip()
            if not message:
                return
            
            # Add user message to chat
            self.add_chat_message("user", message)
            
            # Clear input
            self.chat_input.clear()
            
            # Process the message
            self.process_chat_message(message)
            
        except Exception as e:
            logger.error(f"Error sending chat message: {e}")
    
    def send_quick_command(self, command: str):
        """Send a quick command"""
        try:
            self.chat_input.setPlainText(command)
            self.send_chat_message()
        except Exception as e:
            logger.error(f"Error sending quick command: {e}")
    
    def start_voice_input(self):
        """Start voice input for chat"""
        try:
            if not self.voice_assistant or not self.voice_assistant.running:
                self.add_chat_message("assistant", "🎤 Voice input is not active. Please enable voice assistant first.")
                return
            
            self.add_chat_message("assistant", "🎤 Listening... Speak your message now.")
            self.voice_assistant.listening_for_command = True
            self.voice_assistant.listening_status.emit(True)
            
        except Exception as e:
            logger.error(f"Error starting voice input: {e}")
    
    def clear_chat_history(self):
        """Clear the chat history"""
        try:
            self.chat_display.clear()
            self.add_chat_message("assistant", "👋 Chat history cleared. How can I help you today?")
        except Exception as e:
            logger.error(f"Error clearing chat history: {e}")
    
    def process_chat_message(self, message: str):
        """Process chat message and generate response"""
        try:
            # Show loading indicator
            self.add_chat_message("assistant", "🤔 Processing your request...")
            
            # Use the existing voice command processor
            response = self.process_voice_command(message)
            
            # Remove the last message (loading indicator) and add the actual response
            self.remove_last_chat_message()
            self.add_chat_message("assistant", response)
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            self.add_chat_message("assistant", "❌ Sorry, I encountered an error processing your message. Please try again.")
    
    def remove_last_chat_message(self):
        """Remove the last message from chat display"""
        try:
            # Get current text
            current_text = self.chat_display.toPlainText()
            
            # Split by lines and remove the last complete message block
            lines = current_text.split('\n')
            
            # Find the last complete message block (look for timestamp pattern)
            last_timestamp_index = -1
            for i in range(len(lines) - 1, -1, -1):
                if '[' in lines[i] and ']' in lines[i] and ':' in lines[i]:
                    last_timestamp_index = i
                    break
            
            if last_timestamp_index != -1:
                # Remove everything from the last timestamp onwards
                new_text = '\n'.join(lines[:last_timestamp_index])
                self.chat_display.setPlainText(new_text)
            
        except Exception as e:
            logger.error(f"Error removing last chat message: {e}")

    def send_manual_command(self):
        """Send a manual command from text box to the voice command handler"""
        try:
            text = self.manual_command_input.toPlainText().strip()
            if not text:
                QMessageBox.warning(self, "Input Required", "Please enter a command, e.g., 'start monitoring'.")
                return
            self.handle_voice_command(text)
            self.manual_command_input.clear()
        except Exception as e:
            self.log_message(f"❌ Manual command error: {e}")
    
    def setup_image_tab(self):
        """Setup image analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Upload section
        upload_layout = QHBoxLayout()
        
        self.upload_button = QPushButton("📁 Upload Image for Analysis")
        self.upload_button.clicked.connect(self.upload_and_analyze_image)
        
        self.batch_upload_button = QPushButton("📂 Batch Upload Images")
        self.batch_upload_button.clicked.connect(self.batch_upload_images)
        
        upload_layout.addWidget(self.upload_button)
        upload_layout.addWidget(self.batch_upload_button)
        upload_layout.addStretch()
        
        # Image display
        self.image_display = QLabel()
        self.image_display.setFixedHeight(400)
        self.image_display.setStyleSheet("border: 2px solid #3f3f5f; border-radius: 8px; background-color: #12121c;")
        self.image_display.setAlignment(Qt.AlignCenter)
        
        # Medical guidance status
        self.medical_status_label = QLabel("")
        self.medical_status_label.setStyleSheet("color: #00ffcc; font-weight: bold; padding: 5px;")
        self.medical_status_label.setAlignment(Qt.AlignCenter)
        self.medical_status_label.hide()
        self.image_display.setText("📷 Upload an image to see analysis")
        
        # Analysis results
        self.analysis_results = QTextEdit()
        self.analysis_results.setReadOnly(True)
        self.analysis_results.setMaximumHeight(200)
        
        layout.addLayout(upload_layout)
        layout.addWidget(self.image_display)
        layout.addWidget(self.medical_status_label)
        layout.addWidget(self.analysis_results)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "📷 Image Analysis")
    
    def setup_alerts_tab(self):
        """Setup alerts and medical guidance tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Active alerts section
        alerts_group = QGroupBox("🚨 Active Alerts")
        alerts_layout = QVBoxLayout()
        
        self.active_alerts_list = QTextEdit()
        self.active_alerts_list.setReadOnly(True)
        self.active_alerts_list.setMaximumHeight(150)
        self.active_alerts_list.setText("No active alerts")
        
        alerts_layout.addWidget(self.active_alerts_list)
        alerts_group.setLayout(alerts_layout)
        
        # Medical guidance section
        medical_group = QGroupBox("🏥 Medical Guidance")
        medical_layout = QVBoxLayout()
        
        self.medical_guidance_text = QTextEdit()
        self.medical_guidance_text.setReadOnly(True)
        self.medical_guidance_text.setText("Medical guidance will appear here when emergencies are detected...")
        
        # Manual medical query
        manual_query_layout = QHBoxLayout()
        self.medical_query_input = QTextEdit()
        self.medical_query_input.setMaximumHeight(60)
        self.medical_query_input.setPlaceholderText("Describe symptoms for medical assistance...")
        
        self.query_medical_button = QPushButton("🩺 Get Medical Help")
        self.query_medical_button.clicked.connect(self.manual_medical_query)
        
        manual_query_layout.addWidget(self.medical_query_input)
        manual_query_layout.addWidget(self.query_medical_button)
        
        medical_layout.addWidget(self.medical_guidance_text)
        medical_layout.addLayout(manual_query_layout)
        medical_group.setLayout(medical_layout)
        
        # Emergency actions
        emergency_group = QGroupBox("🆘 Emergency Actions")
        emergency_layout = QHBoxLayout()
        
        self.emergency_alert_button = QPushButton("🚨 Trigger Emergency Alert")
        self.emergency_alert_button.setStyleSheet("background-color: #ff4444;")
        self.emergency_alert_button.clicked.connect(self.trigger_emergency_alert)
        
        self.contact_mission_button = QPushButton("📡 Contact Mission Control")
        self.contact_mission_button.clicked.connect(self.contact_mission_control)
        
        emergency_layout.addWidget(self.emergency_alert_button)
        emergency_layout.addWidget(self.contact_mission_button)
        emergency_layout.addStretch()
        
        emergency_group.setLayout(emergency_layout)
        
        layout.addWidget(alerts_group)
        layout.addWidget(medical_group)
        layout.addWidget(emergency_group)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "🚨 Alerts & Medical")
    
    def setup_automation_tab(self):
        """Setup automation and task execution tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("🤖 Task Automation")
        title_label.setFont(QFont("Orbitron", 16, QFont.Bold))
        title_label.setStyleSheet("color: #00ffcc; margin: 10px; text-align: center;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Task execution section
        task_group = QGroupBox("📝 Task Execution")
        task_layout = QVBoxLayout()
        
        # Task input
        task_input_layout = QHBoxLayout()
        self.task_input = QTextEdit()
        self.task_input.setMaximumHeight(80)
        self.task_input.setPlaceholderText("Enter your task prompt here, e.g., 'open excel and list all states in India'")
        
        self.execute_task_button = QPushButton("🚀 Execute Task")
        self.execute_task_button.clicked.connect(self.execute_task)
        
        task_input_layout.addWidget(self.task_input)
        task_input_layout.addWidget(self.execute_task_button)
        
        # Task history
        self.task_history = QTextEdit()
        self.task_history.setReadOnly(True)
        self.task_history.setMaximumHeight(200)
        self.task_history.setText("Task execution history will appear here...")
        
        task_layout.addLayout(task_input_layout)
        task_layout.addWidget(self.task_history)
        task_group.setLayout(task_layout)
        
        # Server configuration section
        server_group = QGroupBox("🌐 Server Configuration")
        server_layout = QVBoxLayout()
        
        # Server settings
        server_settings_layout = QGridLayout()
        
        server_settings_layout.addWidget(QLabel("Server IP:"), 0, 0)
        self.server_ip_input = QLineEdit(SERVER_IP)
        self.server_ip_input.setPlaceholderText("Enter server IP address")
        server_settings_layout.addWidget(self.server_ip_input, 0, 1)
        
        server_settings_layout.addWidget(QLabel("Server Port:"), 1, 0)
        self.server_port_input = QLineEdit(str(SERVER_PORT))
        self.server_port_input.setPlaceholderText("Enter server port")
        server_settings_layout.addWidget(self.server_port_input, 1, 1)
        
        server_layout.addLayout(server_settings_layout)
        
        # Test connection button
        self.test_connection_button = QPushButton("🔗 Test Connection")
        self.test_connection_button.clicked.connect(self.test_server_connection)
        server_layout.addWidget(self.test_connection_button)
        
        server_group.setLayout(server_layout)
        
        # Sensor monitoring section
        sensor_group = QGroupBox("📡 Sensor Monitoring")
        sensor_layout = QVBoxLayout()
        
        # Automation server controls
        automation_controls_layout = QHBoxLayout()
        
        self.start_automation_button = QPushButton("🔵 Start Automation Server")
        self.start_automation_button.clicked.connect(self.start_automation_server)
        
        self.stop_automation_button = QPushButton("🔴 Stop Automation Server")
        self.stop_automation_button.clicked.connect(self.stop_automation_server)
        self.stop_automation_button.setEnabled(False)
        
        automation_controls_layout.addWidget(self.start_automation_button)
        automation_controls_layout.addWidget(self.stop_automation_button)
        automation_controls_layout.addStretch()
        
        # Sensor status
        self.sensor_status = QTextEdit()
        self.sensor_status.setReadOnly(True)
        self.sensor_status.setMaximumHeight(150)
        self.sensor_status.setText("Automation server not started. Sensors will be monitored here...")
        
        sensor_layout.addLayout(automation_controls_layout)
        sensor_layout.addWidget(self.sensor_status)
        sensor_group.setLayout(sensor_layout)
        
        # Add all groups to main layout
        layout.addWidget(task_group)
        layout.addWidget(server_group)
        layout.addWidget(sensor_group)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "🤖 Automation")
    
    def setup_settings_tab(self):
        """Setup settings and configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Detection settings
        detection_group = QGroupBox("🔍 Detection Settings")
        detection_layout = QGridLayout()
        
        # Unconscious threshold
        detection_layout.addWidget(QLabel("Unconscious Alert Threshold (seconds):"), 0, 0)
        self.unconscious_threshold_spin = QSpinBox()
        self.unconscious_threshold_spin.setRange(1, 30)
        self.unconscious_threshold_spin.setValue(3)
        detection_layout.addWidget(self.unconscious_threshold_spin, 0, 1)
        
        # Stress threshold
        detection_layout.addWidget(QLabel("Stress Alert Threshold (1-10):"), 1, 0)
        self.stress_threshold_slider = QSlider(Qt.Horizontal)
        self.stress_threshold_slider.setRange(1, 10)
        self.stress_threshold_slider.setValue(7)
        detection_layout.addWidget(self.stress_threshold_slider, 1, 1)
        
        # Detection confidence
        detection_layout.addWidget(QLabel("Detection Confidence Threshold:"), 2, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(30, 90)
        self.confidence_slider.setValue(40)
        detection_layout.addWidget(self.confidence_slider, 2, 1)
        
        detection_group.setLayout(detection_layout)
        
        # System settings
        system_group = QGroupBox("⚙️ System Settings")
        system_layout = QGridLayout()
        
        # Camera selection
        system_layout.addWidget(QLabel("Camera Device:"), 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Default Camera (0)", "USB Camera (1)", "Network Camera"])
        system_layout.addWidget(self.camera_combo, 0, 1)
        
        # Mic input selection
        system_layout.addWidget(QLabel("Microphone Device:"), 1, 0)
        self.mic_combo = QComboBox()
        try:
            temp_va = EnhancedVoiceAssistant()
            for name in temp_va.list_audio_devices():
                self.mic_combo.addItem(name)
            temp_va.stop()
        except Exception:
            self.mic_combo.addItems(["Default (system)"])
        system_layout.addWidget(self.mic_combo, 1, 1)

        # TTS controls
        system_layout.addWidget(QLabel("TTS Rate:"), 2, 0)
        self.tts_rate_slider = QSlider(Qt.Horizontal)
        self.tts_rate_slider.setRange(120, 220)
        self.tts_rate_slider.setValue(170)
        system_layout.addWidget(self.tts_rate_slider, 2, 1)

        system_layout.addWidget(QLabel("TTS Volume:"), 3, 0)
        self.tts_volume_slider = QSlider(Qt.Horizontal)
        self.tts_volume_slider.setRange(0, 100)
        self.tts_volume_slider.setValue(100)
        system_layout.addWidget(self.tts_volume_slider, 3, 1)

        # GPU acceleration
        self.gpu_checkbox = QCheckBox("Enable GPU Acceleration")
        self.gpu_checkbox.setChecked(True)
        system_layout.addWidget(self.gpu_checkbox, 4, 0, 1, 2)
        
        # Auto-start systems
        self.autostart_checkbox = QCheckBox("Auto-start Detection on Launch")
        system_layout.addWidget(self.autostart_checkbox, 5, 0, 1, 2)
        
        system_group.setLayout(system_layout)
        
        # API settings
        api_group = QGroupBox("🌐 API Settings")
        api_layout = QVBoxLayout()
        
        groq_layout = QHBoxLayout()
        groq_layout.addWidget(QLabel("Groq API Key:"))
        self.groq_api_input = QTextEdit()
        self.groq_api_input.setMaximumHeight(30)
        self.groq_api_input.setPlaceholderText("Enter your Groq API key...")
        groq_layout.addWidget(self.groq_api_input)
        
        api_layout.addLayout(groq_layout)
        api_group.setLayout(api_layout)
        
        # Save settings button
        save_button = QPushButton("💾 Save Settings")
        save_button.clicked.connect(self.save_settings)
        
        layout.addWidget(detection_group)
        layout.addWidget(system_group)
        layout.addWidget(api_group)
        layout.addWidget(save_button)
        layout.addStretch()
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "⚙️ Settings")
    
    def setup_log_tab(self):
        """Setup system logs tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Log controls
        log_controls = QHBoxLayout()
        
        self.clear_logs_button = QPushButton("🗑️ Clear Logs")
        self.clear_logs_button.clicked.connect(self.clear_logs)
        
        self.export_logs_button = QPushButton("💾 Export Logs")
        self.export_logs_button.clicked.connect(self.export_logs)
        
        log_controls.addWidget(self.clear_logs_button)
        log_controls.addWidget(self.export_logs_button)
        log_controls.addStretch()
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Courier", 10))
        
        layout.addLayout(log_controls)
        layout.addWidget(self.log_display)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "📋 Logs")
    
    def setup_connections(self):
        """Setup signal connections"""
        # Alert system connections
        self.alert_system.alert_display.connect(self.display_alert)
    
    def initialize_systems(self):
        """Initialize all systems"""
        try:
            # Initialize AI systems
            api_key = os.getenv("GROQ_API_KEY")
            self.medical_ai = EnhancedMedicalAI(api_key)
            self.general_ai = GeneralAI(api_key)
            self.alert_system.set_medical_ai(self.medical_ai)
            
            # Initialize Detection Engine
            self.detection_engine = EnhancedDetectionEngine()
            self.detection_engine.detection_ready.connect(self.handle_detection_result)
            self.detection_engine.alert_triggered.connect(self.alert_system.handle_alert)
            
            # Initialize Voice Assistant
            self.voice_assistant = EnhancedVoiceAssistant()
            self.voice_assistant.command_received.connect(self.handle_voice_command)
            self.voice_assistant.listening_status.connect(self.update_voice_status)
            
            self.log_message("✅ All systems initialized successfully")
            
        except Exception as e:
            self.log_message(f"❌ System initialization error: {e}")
            logger.error(f"System initialization error: {e}")
    
    # Event handlers and system methods
    
    def start_live_feed(self):
        """Start live camera feed and detection"""
        try:
            camera_index = self.camera_combo.currentIndex()
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera")
            
            # Start detection engine
            self.detection_engine.start()
            
            # Start camera timer
            self.camera_timer.start(50)  # 20 FPS
            
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            self.log_message("🎥 Live feed started")
            
        except Exception as e:
            self.log_message(f"❌ Camera error: {e}")
            QMessageBox.critical(self, "Camera Error", f"Failed to start camera: {e}")
    
    def stop_live_feed(self):
        """Stop live camera feed and detection"""
        try:
            self.camera_timer.stop()
            
            if self.detection_engine:
                self.detection_engine.stop()
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            self.camera_label.setText("📷 Camera feed stopped")
            
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            self.log_message("⏹️ Live feed stopped")
            
        except Exception as e:
            self.log_message(f"❌ Stop camera error: {e}")
    
    def update_camera(self):
        """Update camera frame"""
        try:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    # Send frame to detection engine
                    self.detection_engine.add_frame(frame.copy())
                    
                    # Display frame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    # Scale image to fit display
                    pixmap = QPixmap.fromImage(qt_image)
                    scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.camera_label.setPixmap(scaled_pixmap)
                    
        except Exception as e:
            self.log_message(f"❌ Camera update error: {e}")
    
    def handle_detection_result(self, result: DetectionResult):
        """Handle detection results"""
        try:
            status_message = f"[{result.timestamp.strftime('%H:%M:%S')}] "
            
            # Objects detected
            if result.objects_detected:
                status_message += f"Objects: {', '.join(result.objects_detected)} | "
            
            # People status
            if result.people_detected:
                status_message += f"People: {result.people_detected} detected"
                if result.unconscious_people:
                    status_message += f", {len(result.unconscious_people)} UNCONSCIOUS ⚠️"
                status_message += " | "
            
            # Stress levels
            if result.face_emotions:
                max_stress = max([emotions.get('stress_level', 0) 
                                for emotions in result.face_emotions.values()])
                status_message += f"Max Stress: {max_stress:.1f}/10 | "
            
            # Breathing
            breathing_status = "✓" if result.breathing_detected else "⚠️"
            status_message += f"Breathing: {breathing_status}"
            
            self.status_text.append(status_message)
            
            # Auto-scroll to bottom
            cursor = self.status_text.textCursor()
            cursor.movePosition(cursor.End)
            self.status_text.setTextCursor(cursor)
            
        except Exception as e:
            self.log_message(f"❌ Detection result handling error: {e}")
    
    def start_voice_assistant(self):
        """Start voice assistant"""
        try:
            self.voice_assistant.start()
            self.start_voice_button.setEnabled(False)
            self.stop_voice_button.setEnabled(True)
            
            # Update chatbot status
            if hasattr(self, 'chatbot_status_label'):
                self.chatbot_status_label.setText("🤖 AI Assistant: Voice Enabled")
                self.chatbot_status_label.setStyleSheet("color: #00ff9f; margin: 10px;")
                self.add_chat_message("assistant", "🎙️ Voice input enabled! You can now speak your commands or questions.")
            else:
                # Fallback for old interface
                self.conversation_text.append("🤖 Assistant: Voice assistant is now active. Ask me anything or give commands.")
                if self.voice_assistant:
                    self.voice_assistant.speak("Voice assistant active. Ask me anything or give commands.")
            
            self.log_message("🎙️ Voice assistant started")
            
        except Exception as e:
            self.log_message(f"❌ Voice assistant error: {e}")
            if hasattr(self, 'chat_display'):
                self.add_chat_message("assistant", f"❌ Failed to start voice assistant: {e}")
            QMessageBox.critical(self, "Voice Error", f"Failed to start voice assistant: {e}")
    
    def stop_voice_assistant(self):
        """Stop voice assistant"""
        try:
            self.voice_assistant.stop()
            self.start_voice_button.setEnabled(True)
            self.stop_voice_button.setEnabled(False)
            
            # Update chatbot status
            if hasattr(self, 'chatbot_status_label'):
                self.chatbot_status_label.setText("🤖 AI Assistant: Text Only")
                self.chatbot_status_label.setStyleSheet("color: #ffaa00; margin: 10px;")
                self.add_chat_message("assistant", "🎙️ Voice input disabled. You can still type your messages.")
            else:
                # Fallback for old interface
                self.voice_status_label.setText("🎙️ Voice Assistant: Stopped")
                self.voice_status_label.setStyleSheet("color: #ff6b6b;")
            
            self.log_message("⏹️ Voice assistant stopped")
            
        except Exception as e:
            self.log_message(f"❌ Voice stop error: {e}")
            if hasattr(self, 'chat_display'):
                self.add_chat_message("assistant", f"❌ Error stopping voice assistant: {e}")
    
    def update_voice_status(self, listening: bool):
        """Update voice assistant status"""
        if hasattr(self, 'chatbot_status_label'):
            if listening:
                self.chatbot_status_label.setText("🤖 AI Assistant: Listening...")
                self.chatbot_status_label.setStyleSheet("color: #00ff9f; margin: 10px;")
            else:
                self.chatbot_status_label.setText("🤖 AI Assistant: Ready")
                self.chatbot_status_label.setStyleSheet("color: #00ff9f; margin: 10px;")
        else:
            # Fallback for old voice status label
            if listening:
                self.voice_status_label.setText("🎙️ Voice Assistant: Listening for command...")
                self.voice_status_label.setStyleSheet("color: #00ff9f;")
            else:
                self.voice_status_label.setText("🎙️ Voice Assistant: Active (waiting for wake word)")
                self.voice_status_label.setStyleSheet("color: #00ffcc;")
    
    def handle_voice_command(self, command: str):
        """Handle voice commands"""
        try:
            # Add to chat if voice input was used
            if hasattr(self, 'chat_display'):
                self.add_chat_message("user", f"🎤 {command}")
                # Show loading indicator
                self.add_chat_message("assistant", "🤔 Processing your request...")
            
            response = self.process_voice_command(command)
            
            # Add response to chat (speech is handled automatically in add_chat_message)
            if hasattr(self, 'chat_display'):
                # Remove loading message and add actual response
                self.remove_last_chat_message()
                self.add_chat_message("assistant", response)
            
            self.log_message(f"Voice command processed: {command}")
            
        except Exception as e:
            self.log_message(f"❌ Voice command error: {e}")
            if hasattr(self, 'chat_display'):
                self.add_chat_message("assistant", "❌ Sorry, I encountered an error processing your voice command.")
    
    def process_voice_command(self, command: str) -> str:
        """Process voice commands and return response using Groq API"""
        command = command.lower().strip()
        
        # Handle system commands first (these should execute actions, not just respond)
        if "start monitoring" in command or "begin monitoring" in command:
            if not self.camera_timer.isActive():
                self.start_live_feed()
                return "🚀 Starting live monitoring and detection systems. All cameras and AI detection are now active."
            else:
                return "✅ Monitoring is already active. All systems are running normally."
        
        elif "stop monitoring" in command or "end monitoring" in command:
            if self.camera_timer.isActive():
                self.stop_live_feed()
                return "⏹️ Stopping monitoring systems. All cameras and detection have been deactivated."
            else:
                return "ℹ️ Monitoring is not currently active. Use 'start monitoring' to begin."
        
        elif "emergency" in command or "emergency alert" in command or "help" in command:
            self.trigger_emergency_alert()
            return "🚨 EMERGENCY ALERT ACTIVATED! Initiating emergency protocols and contacting medical AI for immediate assistance. All emergency systems are now online."
        
        # For all other queries, use Groq API for natural language responses
        return self.get_groq_response(command)
    
    def get_groq_response(self, query: str) -> str:
        """Get response from Groq API for natural language queries"""
        try:
            # Get API key from settings or environment
            api_key = os.getenv('GROQ_API_KEY', '')
            if api_key == os.getenv('GROQ_API_KEY', ''):
                # Try to get from settings
                if hasattr(self, 'groq_api_input'):
                    api_key = self.groq_api_input.toPlainText().strip()
            
            if not api_key or api_key == os.getenv('GROQ_API_KEY', ''):
                return "❌ Groq API key not configured. Please set your API key in the Settings tab to enable AI responses."
            
            # Prepare the prompt for space assistant context
            system_prompt = """You are an AI Space Assistant for a space station monitoring system. You help astronauts and mission control with:

- Space mission operations and procedures
- Medical guidance and emergency protocols
- System monitoring and technical support
- General questions about space, astronomy, and space technology
- Safety protocols and emergency procedures

You should be helpful, professional, and knowledgeable about space operations. Keep responses concise but informative. If asked about medical procedures, provide general guidance but always recommend consulting medical professionals for serious situations.

Current system capabilities:
- Live monitoring and detection systems
- Unconsciousness detection for astronaut safety
- Medical AI integration
- Voice and text communication
- Real-time alert systems

Respond naturally and conversationally while maintaining the context of being a space station AI assistant."""
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gemma2-9b-it",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": False
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                
                # Add some context-specific emojis based on query content
                if any(word in query.lower() for word in ["medical", "health", "cpr", "emergency", "injury"]):
                    ai_response = f"🏥 {ai_response}"
                elif any(word in query.lower() for word in ["space", "mission", "astronaut", "orbit", "station"]):
                    ai_response = f"🚀 {ai_response}"
                elif any(word in query.lower() for word in ["weather", "climate", "environment"]):
                    ai_response = f"🌤️ {ai_response}"
                elif any(word in query.lower() for word in ["time", "date", "schedule"]):
                    ai_response = f"🕐 {ai_response}"
                elif any(word in query.lower() for word in ["camera", "detection", "monitoring", "system"]):
                    ai_response = f"📹 {ai_response}"
                else:
                    ai_response = f"🤖 {ai_response}"
                
                return ai_response
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logger.error(f"Groq API error: {error_msg}")
                return f"❌ Sorry, I encountered an error with the AI service: {error_msg}"
                
        except requests.exceptions.Timeout:
            return "⏰ Sorry, the AI service is taking too long to respond. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request error: {e}")
            return f"❌ Network error connecting to AI service: {str(e)}"
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"❌ Sorry, I encountered an error: {str(e)}"
    
    
    def upload_and_analyze_image(self):
        """Upload and analyze single image"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image for Analysis", "", 
                "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
            )
            
            if file_path:
                self.analyze_uploaded_image(file_path)
                
        except Exception as e:
            self.log_message(f"❌ Image upload error: {e}")
            QMessageBox.critical(self, "Upload Error", f"Failed to upload image: {e}")
    
    def batch_upload_images(self):
        """Upload and analyze multiple images"""
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Select Images for Batch Analysis", "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
            )
            
            if file_paths:
                for file_path in file_paths:
                    self.analyze_uploaded_image(file_path)
                    
        except Exception as e:
            self.log_message(f"❌ Batch upload error: {e}")
            QMessageBox.critical(self, "Batch Upload Error", f"Failed to process images: {e}")
    
    def analyze_uploaded_image(self, file_path: str):
        """Analyze uploaded image"""
        try:
            # Load and process image
            img = cv2.imread(file_path)
            if img is None:
                raise Exception("Could not load image")
            
            # Create temporary detection result for image analysis
            temp_result = DetectionResult()
            
            # Run detection on image
            if self.detection_engine:
                temp_result.objects_detected = self.detection_engine.detect_objects(img)
                
                # Pose analysis (image analysis - no movement tracking)
                pose_results = self.detection_engine.pose_model(img)[0]
                temp_result.people_detected, temp_result.unconscious_people = self.detection_engine.analyze_poses(pose_results, is_live_feed=False)
                
                # Face analysis
                temp_result.face_emotions = self.detection_engine.detect_face_emotions(img)
            
            # Display analyzed image
            annotated_img = self.draw_annotations(img.copy(), temp_result, pose_results)
            self.display_analyzed_image(annotated_img)
            
            # Display results
            self.display_image_analysis_results(file_path, temp_result)
            
            self.log_message(f"📷 Analyzed image: {os.path.basename(file_path)}")
            
        except Exception as e:
            self.log_message(f"❌ Image analysis error: {e}")
    
    def draw_annotations(self, img, result: DetectionResult, pose_results):
        """Draw annotations on image"""
        try:
            # Draw pose annotations
            if pose_results and pose_results.keypoints is not None:
                for i, pose in enumerate(pose_results.keypoints):
                    keypoints = pose.data[0].cpu().numpy()
                    if len(keypoints) < 13 or np.isnan(keypoints).any():
                        continue
                        
                    xy = keypoints[:, :2]
                    
                    # Check if person is unconscious
                    is_unconscious = any(person['person_id'] == i for person in result.unconscious_people)
                    
                    # Draw bounding box
                    min_x, min_y = int(np.min(xy[:, 0])), int(np.min(xy[:, 1]))
                    max_x, max_y = int(np.max(xy[:, 0])), int(np.max(xy[:, 1]))
                    
                    color = (0, 0, 255) if is_unconscious else (0, 255, 0)
                    label = "⚠ UNCONSCIOUS" if is_unconscious else "Conscious"
                    
                    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color, 2)
                    cv2.putText(img, label, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            return img
            
        except Exception as e:
            logger.error(f"Annotation error: {e}")
            return img
    
    def display_analyzed_image(self, img):
        """Display analyzed image in UI"""
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Image display error: {e}")
    
    def display_image_analysis_results(self, file_path: str, result: DetectionResult):
        """Display analysis results"""
        try:
            analysis_text = f"Analysis Results for: {os.path.basename(file_path)}\n"
            analysis_text += f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            if result.objects_detected:
                analysis_text += f"🔍 Objects Detected: {', '.join(result.objects_detected)}\n"
            
            if result.people_detected:
                analysis_text += f"👥 People Detected: {result.people_detected}\n"
                
                if result.unconscious_people:
                    analysis_text += f"⚠️ Unconscious People: {len(result.unconscious_people)}\n"
                    for person in result.unconscious_people:
                        debug_info = person.get('debug_info', {})
                        analysis_text += f"   - Person {person['person_id']}: Confidence {debug_info.get('confidence', 0):.2f}\n"
            
            if result.face_emotions:
                analysis_text += f"😟 Stress Analysis:\n"
                for face_id, emotions in result.face_emotions.items():
                    stress_level = emotions.get('stress_level', 0)
                    analysis_text += f"   - {face_id}: Stress Level {stress_level:.1f}/10\n"
            
            analysis_text += "\n" + "="*50 + "\n"
            
            self.analysis_results.append(analysis_text)
            
            # Auto-scroll to bottom
            cursor = self.analysis_results.textCursor()
            cursor.movePosition(cursor.End)
            self.analysis_results.setTextCursor(cursor)
            
            # Check for emergencies in uploaded image
            if result.unconscious_people:
                self.alert_system.handle_alert(AlertLevel.CRITICAL, 
                                             f"Unconscious person detected in uploaded image: {os.path.basename(file_path)}", 
                                             result)
                
                # Generate and display medical guidance for unconscious person
                self.generate_unconscious_medical_guidance(file_path, result)
                
        except Exception as e:
            logger.error(f"Results display error: {e}")
    
    def generate_unconscious_medical_guidance(self, file_path: str, result: DetectionResult):
        """Generate medical guidance for unconscious person detection using Groq API"""
        try:
            if not self.general_ai:
                self.analysis_results.append("\n❌ Medical AI not available for guidance generation.\n")
                return
            
            # Prepare context for medical AI
            unconscious_count = len(result.unconscious_people)
            filename = os.path.basename(file_path)
            
            # Show loading indicator
            loading_text = f"\n{'='*60}\n"
            loading_text += f"🚨 MEDICAL EMERGENCY DETECTED\n"
            loading_text += f"Image: {filename}\n"
            loading_text += f"Unconscious people: {unconscious_count}\n"
            loading_text += f"⏳ Generating medical guidance...\n"
            loading_text += f"{'='*60}\n"
            
            self.analysis_results.append(loading_text)
            
            # Show status label
            self.medical_status_label.setText("🚨 MEDICAL EMERGENCY - Generating guidance...")
            self.medical_status_label.setStyleSheet("color: #ff4444; font-weight: bold; padding: 5px; background-color: #2a1a1a; border: 2px solid #ff4444; border-radius: 5px;")
            self.medical_status_label.show()
            
            # Auto-scroll to show loading
            cursor = self.analysis_results.textCursor()
            cursor.movePosition(cursor.End)
            self.analysis_results.setTextCursor(cursor)
            
            medical_prompt = f"""
            CRITICAL MEDICAL SITUATION: Unconscious person detected in image analysis.
            
            Image: {filename}
            Unconscious people detected: {unconscious_count}
            Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Please provide immediate medical guidance in the following format:
            
            POSSIBLE CAUSES:
            - List the most likely causes of unconsciousness
            - Include both medical and environmental factors
            - Consider space/astronaut context if relevant
            
            HOW TO HELP AND ALERT:
            - Immediate first aid steps
            - Emergency response procedures
            - Who to contact and how
            - Safety precautions for responders
            - Monitoring instructions
            
            Please provide clear, actionable medical guidance that can help save this person's life.
            """
            
            # Get medical guidance from AI
            medical_guidance = self.general_ai.get_response(medical_prompt)
            
            # Format and display the guidance
            guidance_text = f"\n{'='*60}\n"
            guidance_text += f"🚨 MEDICAL EMERGENCY GUIDANCE\n"
            guidance_text += f"Image: {filename}\n"
            guidance_text += f"Detected: {unconscious_count} unconscious person(s)\n"
            guidance_text += f"Time: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            guidance_text += f"{'='*60}\n\n"
            guidance_text += medical_guidance
            guidance_text += f"\n\n{'='*60}\n"
            
            # Display in analysis results
            self.analysis_results.append(guidance_text)
            
            # Also display in medical guidance tab if available
            if hasattr(self, 'medical_guidance_text'):
                self.medical_guidance_text.append(guidance_text)
            
            # Update status label to show completion
            self.medical_status_label.setText("✅ Medical guidance generated successfully")
            self.medical_status_label.setStyleSheet("color: #00ff44; font-weight: bold; padding: 5px; background-color: #1a2a1a; border: 2px solid #00ff44; border-radius: 5px;")
            
            # Auto-scroll to bottom
            cursor = self.analysis_results.textCursor()
            cursor.movePosition(cursor.End)
            self.analysis_results.setTextCursor(cursor)
            
            # Speak critical alert if voice assistant is available
            if self.voice_assistant:
                self.voice_assistant.speak(f"Critical medical emergency! Unconscious person detected in {filename}. Medical guidance has been generated.")
            
            self.log_message(f"🚨 Medical guidance generated for unconscious person in {filename}")
            
            # Hide status label after 5 seconds
            QTimer.singleShot(5000, self.medical_status_label.hide)
            
        except Exception as e:
            error_msg = f"❌ Error generating medical guidance: {e}"
            self.analysis_results.append(f"\n{error_msg}\n")
            logger.error(f"Medical guidance generation error: {e}")
    
    def display_alert(self, level: int, message: str, alert_data: Dict):
        """Display alert in UI with automatic tab switching for live feed only"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Color coding by severity
            colors = {
                AlertLevel.INFO: "#00ffcc",
                AlertLevel.WARNING: "#ffaa00", 
                AlertLevel.CRITICAL: "#ff6b6b",
                AlertLevel.EMERGENCY: "#ff0000"
            }
            
            level_names = {
                AlertLevel.INFO: "INFO",
                AlertLevel.WARNING: "WARNING", 
                AlertLevel.CRITICAL: "CRITICAL",
                AlertLevel.EMERGENCY: "EMERGENCY"
            }
            
            color = colors.get(level, "#ffffff")
            level_name = level_names.get(level, "UNKNOWN")
            
            alert_text = f"[{timestamp}] {level_name}: {message}\n"
            
            # Update active alerts display
            self.active_alerts_list.append(alert_text)
            
            # Update medical guidance if available
            medical_assessment = alert_data.get('medical_assessment')
            if medical_assessment:
                self.display_medical_guidance(medical_assessment)
            
            # Auto-switch to medical alerts tab ONLY for live feed unconscious person alerts
            # Check if this is from live feed (not image analysis) by checking if camera is active
            if ("unconscious" in message.lower() and 
                level >= AlertLevel.CRITICAL and 
                self.camera_timer.isActive()):  # Only if live feed is active
                
                # Find the medical alerts tab index
                for i in range(self.tab_widget.count()):
                    if "Alerts" in self.tab_widget.tabText(i) or "Medical" in self.tab_widget.tabText(i):
                        self.tab_widget.setCurrentIndex(i)
                        break
                
                # Ensure medical guidance is ready
                self.prepare_emergency_medical_guidance()
            
            # Show popup for critical/emergency alerts
            if level >= AlertLevel.CRITICAL:
                self.show_critical_alert_popup(level_name, message, alert_data)
            
            self.log_message(f"🚨 {level_name}: {message}")
            
        except Exception as e:
            logger.error(f"Alert display error: {e}")
    
    def show_critical_alert_popup(self, level_name: str, message: str, alert_data: Dict):
        """Show popup for critical alerts"""
        try:
            msg_box = QMessageBox()
            msg_box.setWindowTitle(f"🚨 {level_name} ALERT")
            msg_box.setText(message)
            
            medical_assessment = alert_data.get('medical_assessment')
            if medical_assessment and medical_assessment['immediate_actions']:
                detailed_text = f"\n\nIMMEDIATE ACTIONS REQUIRED:\n"
                for i, action in enumerate(medical_assessment['immediate_actions'][:3], 1):
                    detailed_text += f"{i}. {action}\n"
                
                if medical_assessment['contact_mission_control']:
                    detailed_text += "\n⚠️ CONTACT MISSION CONTROL IMMEDIATELY"
                
                msg_box.setDetailedText(detailed_text)
            
            msg_box.setIcon(QMessageBox.Critical if level_name == "EMERGENCY" else QMessageBox.Warning)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            
        except Exception as e:
            logger.error(f"Critical alert popup error: {e}")
    
    def display_medical_guidance(self, medical_assessment: Dict):
        """Display medical guidance in the medical tab"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            guidance_text = f"\n{'='*60}\n[{timestamp}] MEDICAL ASSESSMENT\n{'='*60}\n\n"
            
            # Severity
            severity = medical_assessment.get('severity', 'Unknown')
            guidance_text += f"🩺 SEVERITY LEVEL: {severity}/10\n\n"
            
            # Immediate actions
            if medical_assessment.get('immediate_actions'):
                guidance_text += "🆘 IMMEDIATE ACTIONS:\n"
                for i, action in enumerate(medical_assessment['immediate_actions'], 1):
                    guidance_text += f"   {i}. {action}\n"
                guidance_text += "\n"
            
            # Potential causes
            if medical_assessment.get('causes'):
                guidance_text += "🔍 POTENTIAL CAUSES:\n"
                for cause in medical_assessment['causes']:
                    guidance_text += f"   • {cause}\n"
                guidance_text += "\n"
            
            # Monitoring instructions
            if medical_assessment.get('monitoring'):
                guidance_text += "👁️ MONITORING INSTRUCTIONS:\n"
                for instruction in medical_assessment['monitoring']:
                    guidance_text += f"   • {instruction}\n"
                guidance_text += "\n"
            
            # Mission control alert
            if medical_assessment.get('contact_mission_control'):
                guidance_text += "📡 MISSION CONTROL: Contact immediately required\n\n"

            # If structured sections are empty, show the full AI text for clarity
            if not any([
                medical_assessment.get('immediate_actions'),
                medical_assessment.get('causes'),
                medical_assessment.get('monitoring')
            ]):
                full_text = medical_assessment.get('full_response', '').strip()
                if full_text:
                    guidance_text += "📝 AI NOTES (unstructured):\n"
                    guidance_text += full_text + "\n\n"
            
            self.medical_guidance_text.append(guidance_text)
            
            # Auto-scroll to bottom
            cursor = self.medical_guidance_text.textCursor()
            cursor.movePosition(cursor.End)
            self.medical_guidance_text.setTextCursor(cursor)
            
        except Exception as e:
            logger.error(f"Medical guidance display error: {e}")
    
    def prepare_emergency_medical_guidance(self):
        """Prepare emergency medical guidance for unconscious person"""
        try:
            # Clear any existing guidance
            self.medical_guidance_text.clear()
            
            # Add immediate emergency guidance
            emergency_guidance = f"""
{'='*80}
🚨 EMERGENCY MEDICAL PROTOCOL ACTIVATED
{'='*80}

🆘 IMMEDIATE ACTIONS REQUIRED:
   1. Check astronaut's breathing and pulse immediately
   2. Ensure airway is clear and unobstructed
   3. Place in recovery position if safe to do so
   4. Check for any visible injuries or obstructions
   5. Contact mission control immediately

🔍 ASSESSMENT CHECKLIST:
   • Breathing: Check for chest movement and breathing sounds
   • Pulse: Check carotid pulse (neck) and radial pulse (wrist)
   • Consciousness: Try to rouse with voice and gentle touch
   • Airway: Look for obstructions, check mouth and throat
   • Circulation: Check for bleeding or signs of shock

📡 MISSION CONTROL CONTACT:
   • Priority communication channel activated
   • Medical telemetry being transmitted
   • Emergency beacon status: ACTIVE
   • Estimated response time: 2-3 minutes

👁️ CONTINUOUS MONITORING:
   • Breathing rate and pattern
   • Heart rate and rhythm
   • Consciousness level changes
   • Any movement or response to stimuli
   • Vital signs every 30 seconds

⚠️ CRITICAL NOTES:
   • Do not move the astronaut unless absolutely necessary
   • Document all observations and actions taken
   • Prepare for potential evacuation procedures
   • Monitor for any environmental hazards

{'='*80}
"""
            
            self.medical_guidance_text.append(emergency_guidance)
            
            # Auto-scroll to top to show emergency guidance
            cursor = self.medical_guidance_text.textCursor()
            cursor.movePosition(cursor.Start)
            self.medical_guidance_text.setTextCursor(cursor)
            
            self.log_message("🚨 Emergency medical guidance prepared and displayed")
            
        except Exception as e:
            logger.error(f"Emergency medical guidance preparation error: {e}")
    
    def manual_medical_query(self):
        """Handle manual medical query"""
        try:
            query_text = self.medical_query_input.toPlainText().strip()
            if not query_text:
                QMessageBox.warning(self, "Input Required", "Please enter a medical query.")
                return
            
            # Create dummy detection result for manual query
            dummy_result = DetectionResult()
            
            # Prepare symptoms from query
            symptoms = {
                'manual_query': True,
                'description': query_text
            }
            
            if self.medical_ai:
                self.medical_guidance_text.append(f"\n🤖 Processing query: {query_text}\n")
                medical_response = self.medical_ai.get_medical_assessment(symptoms, dummy_result)
                self.display_medical_guidance(medical_response)
                
                # Clear input
                self.medical_query_input.clear()
            else:
                QMessageBox.critical(self, "Medical AI Error", "Medical AI system not available.")
            
        except Exception as e:
            self.log_message(f"❌ Manual medical query error: {e}")
    
    # -----------------------------
    # AUTOMATION METHODS
    # -----------------------------
    
    def execute_task(self):
        """Execute a task by sending prompt to server"""
        try:
            prompt = self.task_input.toPlainText().strip()
            if not prompt:
                QMessageBox.warning(self, "Input Required", "Please enter a task prompt.")
                return
            
            # Disable button during execution
            self.execute_task_button.setEnabled(False)
            self.execute_task_button.setText("⏳ Executing...")
            
            # Add to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.task_history.append(f"[{timestamp}] Executing: {prompt}")
            
            # Connect signals
            self.task_executor.task_completed.connect(self.on_task_completed)
            self.task_executor.task_failed.connect(self.on_task_failed)
            
            # Execute task in separate thread
            self.task_executor.execute_task(prompt)
            
        except Exception as e:
            self.log_message(f"❌ Task execution error: {e}")
            self.execute_task_button.setEnabled(True)
            self.execute_task_button.setText("🚀 Execute Task")
    
    def on_task_completed(self, result: dict):
        """Handle task completion"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            status = result.get('status', 'unknown')
            message = result.get('message', 'Task completed')
            
            self.task_history.append(f"[{timestamp}] ✅ {status.upper()}: {message}")
            
            # Speak result if voice assistant is available
            if self.voice_assistant:
                self.voice_assistant.speak(f"Task completed. {message}")
            
            # Re-enable button
            self.execute_task_button.setEnabled(True)
            self.execute_task_button.setText("🚀 Execute Task")
            
            # Clear input
            self.task_input.clear()
            
        except Exception as e:
            self.log_message(f"❌ Task completion handler error: {e}")
    
    def on_task_failed(self, error_msg: str):
        """Handle task failure"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.task_history.append(f"[{timestamp}] ❌ FAILED: {error_msg}")
            
            # Speak error if voice assistant is available
            if self.voice_assistant:
                self.voice_assistant.speak("Task failed. Please check the error message.")
            
            # Re-enable button
            self.execute_task_button.setEnabled(True)
            self.execute_task_button.setText("🚀 Execute Task")
            
        except Exception as e:
            self.log_message(f"❌ Task failure handler error: {e}")
    
    def test_server_connection(self):
        """Test connection to the automation server"""
        try:
            server_ip = self.server_ip_input.text().strip()
            server_port = self.server_port_input.text().strip()
            
            if not server_ip or not server_port:
                QMessageBox.warning(self, "Invalid Input", "Please enter both server IP and port.")
                return
            
            # Test connection
            test_payload = {"action": "test_connection", "message": "Connection test"}
            r = requests.post(f"http://{server_ip}:{server_port}", json=test_payload, timeout=5)
            
            if r.status_code == 200:
                QMessageBox.information(self, "Connection Test", "✅ Server connection successful!")
                self.log_message("✅ Server connection test successful")
            else:
                QMessageBox.warning(self, "Connection Test", f"❌ Server responded with status {r.status_code}")
                self.log_message(f"❌ Server connection test failed: {r.status_code}")
                
        except Exception as e:
            QMessageBox.critical(self, "Connection Test", f"❌ Connection failed: {e}")
            self.log_message(f"❌ Server connection test error: {e}")
    
    def start_automation_server(self):
        """Start the automation HTTP server"""
        try:
            if self.automation_server:
                QMessageBox.warning(self, "Server Already Running", "Automation server is already running.")
                return
            
            # Create server with custom handler
            def handler(*args, **kwargs):
                return AutomationHandler(self, *args, **kwargs)
            
            self.automation_server = HTTPServer((AUTOMATION_HOST, AUTOMATION_PORT), handler)
            
            # Start server in separate thread
            self.automation_server_thread = threading.Thread(target=self.automation_server.serve_forever)
            self.automation_server_thread.daemon = True
            self.automation_server_thread.start()
            
            # Update UI
            self.start_automation_button.setEnabled(False)
            self.stop_automation_button.setEnabled(True)
            
            # Update sensor status
            self.sensor_status.setText(f"🔵 Automation server running on {AUTOMATION_HOST}:{AUTOMATION_PORT}\n\nMonitoring sensors:")
            for sensor_name, sensor_ip in SENSORS.items():
                self.sensor_status.append(f"  • {sensor_name}: {sensor_ip}")
            
            self.log_message(f"🔵 Automation server started on {AUTOMATION_HOST}:{AUTOMATION_PORT}")
            
            # Connect gas leak signal
            self.gas_leak_detected.connect(self.handle_gas_leak_alert)
            
        except Exception as e:
            QMessageBox.critical(self, "Server Start Error", f"Failed to start automation server: {e}")
            self.log_message(f"❌ Automation server start error: {e}")
    
    def stop_automation_server(self):
        """Stop the automation HTTP server"""
        try:
            if not self.automation_server:
                QMessageBox.warning(self, "Server Not Running", "Automation server is not running.")
                return
            
            # Shutdown server
            self.automation_server.shutdown()
            self.automation_server.server_close()
            self.automation_server = None
            self.automation_server_thread = None
            
            # Update UI
            self.start_automation_button.setEnabled(True)
            self.stop_automation_button.setEnabled(False)
            
            # Update sensor status
            self.sensor_status.setText("🔴 Automation server stopped. Sensors not monitored.")
            
            self.log_message("🔴 Automation server stopped")
            
        except Exception as e:
            QMessageBox.critical(self, "Server Stop Error", f"Failed to stop automation server: {e}")
            self.log_message(f"❌ Automation server stop error: {e}")
    
    def handle_gas_leak_alert(self, alert_data: dict):
        """Handle gas leak alert from sensors"""
        try:
            sensor_name = alert_data.get('sensor_name', 'unknown')
            sensor_ip = alert_data.get('sensor_ip', 'unknown')
            message = alert_data.get('message', 'Gas leak detected')
            
            # Update sensor status
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.sensor_status.append(f"\n[{timestamp}] ⚠️ GAS LEAK ALERT!")
            self.sensor_status.append(f"  Sensor: {sensor_name} ({sensor_ip})")
            self.sensor_status.append(f"  Message: {message}")
            
            # Create alert for the alert system
            self.alert_system.handle_alert(
                AlertLevel.CRITICAL,
                f"Gas leak detected from {sensor_name} at {sensor_ip}",
                None
            )
            
            # Display alert in UI
            self.display_alert({
                'type': 'gas_leak',
                'severity': 'critical',
                'message': message,
                'sensor_name': sensor_name,
                'sensor_ip': sensor_ip,
                'timestamp': alert_data.get('timestamp', datetime.now().isoformat())
            })
            
            # Speak alert if voice assistant is available
            if self.voice_assistant:
                self.voice_assistant.speak(f"Critical alert! Gas leak detected from {sensor_name}")
            
            self.log_message(f"⚠️ Gas leak alert processed: {sensor_name}")
            
        except Exception as e:
            self.log_message(f"❌ Gas leak alert handler error: {e}")
    
    def trigger_emergency_alert(self):
        """Trigger manual emergency alert"""
        try:
            emergency_msg = "MANUAL EMERGENCY ALERT ACTIVATED - Immediate assistance required"
            dummy_result = DetectionResult()
            self.alert_system.handle_alert(AlertLevel.EMERGENCY, emergency_msg, dummy_result)
            
        except Exception as e:
            self.log_message(f"❌ Emergency alert error: {e}")
    
    def contact_mission_control(self):
        """Contact mission control (placeholder)"""
        try:
            # This would integrate with actual mission control systems
            msg = "Mission Control contact feature would be implemented here.\n"
            msg += "This would typically involve:\n"
            msg += "• Automatic telemetry transmission\n"
            msg += "• Priority communication channel activation\n"
            msg += "• Emergency beacon activation\n"
            msg += "• Current status and alert data packaging"
            
            QMessageBox.information(self, "Mission Control", msg)
            self.log_message("📡 Mission Control contact initiated (simulation)")
            
        except Exception as e:
            self.log_message(f"❌ Mission Control contact error: {e}")
    
    def save_settings(self):
        """Save application settings"""
        try:
            settings = {
                'unconscious_threshold': self.unconscious_threshold_spin.value(),
                'stress_threshold': self.stress_threshold_slider.value(),
                'confidence_threshold': self.confidence_slider.value() / 100.0,
                'camera_device': self.camera_combo.currentIndex(),
                'mic_device_index': self._parse_selected_device_index(self.mic_combo.currentText()),
                'tts_rate': self.tts_rate_slider.value(),
                'tts_volume': self.tts_volume_slider.value() / 100.0,
                'gpu_acceleration': self.gpu_checkbox.isChecked(),
                'auto_start': self.autostart_checkbox.isChecked(),
                'groq_api_key': self.groq_api_input.toPlainText().strip()
            }
            
            # Save to file
            with open('space_assistant_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            # Apply settings to systems
            if self.detection_engine:
                self.detection_engine.unconscious_threshold = settings['unconscious_threshold']
                self.detection_engine.stress_threshold = settings['stress_threshold']
            
            # Apply audio settings
            if self.voice_assistant:
                self.voice_assistant.set_input_device(settings['mic_device_index'])
                self.voice_assistant.set_tts_config(settings['tts_rate'], settings['tts_volume'])

            # Update AI API keys
            if settings['groq_api_key'] and self.medical_ai:
                self.medical_ai.api_key = settings['groq_api_key']
            if settings['groq_api_key'] and self.general_ai:
                self.general_ai.api_key = settings['groq_api_key']
            
            QMessageBox.information(self, "Settings", "Settings saved successfully!")
            self.log_message("💾 Settings saved")
            
        except Exception as e:
            self.log_message(f"❌ Settings save error: {e}")
            QMessageBox.critical(self, "Settings Error", f"Failed to save settings: {e}")
    
    def load_settings(self):
        """Load application settings"""
        try:
            if os.path.exists('space_assistant_settings.json'):
                with open('space_assistant_settings.json', 'r') as f:
                    settings = json.load(f)
                
                # Apply settings to UI
                self.unconscious_threshold_spin.setValue(settings.get('unconscious_threshold', 3))
                self.stress_threshold_slider.setValue(settings.get('stress_threshold', 7))
                self.confidence_slider.setValue(int(settings.get('confidence_threshold', 0.4) * 100))
                self.camera_combo.setCurrentIndex(settings.get('camera_device', 0))
                # Mic and TTS
                mic_idx = settings.get('mic_device_index')
                if mic_idx is not None:
                    # Try to select matching index text if present
                    for i in range(self.mic_combo.count()):
                        if str(mic_idx) == self.mic_combo.itemText(i).split(':')[0]:
                            self.mic_combo.setCurrentIndex(i)
                            break
                self.tts_rate_slider.setValue(settings.get('tts_rate', 170))
                self.tts_volume_slider.setValue(int(settings.get('tts_volume', 1.0) * 100))
                self.gpu_checkbox.setChecked(settings.get('gpu_acceleration', True))
                self.autostart_checkbox.setChecked(settings.get('auto_start', False))
                
                groq_key = settings.get('groq_api_key', '')
                if groq_key:
                    self.groq_api_input.setPlainText(groq_key)
                
                self.log_message("📥 Settings loaded")
                
        except Exception as e:
            self.log_message(f"❌ Settings load error: {e}")
    
    def clear_logs(self):
        """Clear system logs"""
        try:
            self.log_display.clear()
            self.log_message("🗑️ Logs cleared")
            
        except Exception as e:
            logger.error(f"Clear logs error: {e}")
    
    def export_logs(self):
        """Export logs to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"space_assistant_logs_{timestamp}.txt"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Logs", filename, "Text Files (*.txt)"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.log_display.toPlainText())
                
                QMessageBox.information(self, "Export Complete", f"Logs exported to:\n{file_path}")
                self.log_message(f"💾 Logs exported to {file_path}")
                
        except Exception as e:
            self.log_message(f"❌ Log export error: {e}")
    
    def log_message(self, message: str):
        """Add message to log display"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            
            self.log_display.append(log_entry)
            
            # Auto-scroll to bottom
            cursor = self.log_display.textCursor()
            cursor.movePosition(cursor.End)
            self.log_display.setTextCursor(cursor)
            
            # Also log to file
            logger.info(message)
            
        except Exception as e:
            logger.error(f"Log message error: {e}")

    def _parse_selected_device_index(self, text: str) -> Optional[int]:
        try:
            return int(text.split(':')[0].strip())
        except Exception:
            return None
    
    def closeEvent(self, event):
        """Handle application close event"""
        try:
            # Stop all systems
            if self.camera_timer.isActive():
                self.stop_live_feed()
            
            if self.voice_assistant:
                self.voice_assistant.stop()
            
            if self.detection_engine:
                self.detection_engine.stop()
            
            # Save settings before closing
            self.save_settings()
            
            self.log_message("🛰️ Space Assistant shutting down")
            
            event.accept()
            
        except Exception as e:
            logger.error(f"Close event error: {e}")
            event.accept()

# Additional utility classes and functions

class VideoReplaySystem:
    """System for replaying and analyzing recorded video files"""
    
    def __init__(self, detection_engine: EnhancedDetectionEngine):
        self.detection_engine = detection_engine
        self.video_files = []
        self.current_video = None
        self.replay_results = []
    
    def add_video_file(self, file_path: str):
        """Add video file for replay analysis"""
        if os.path.exists(file_path):
            self.video_files.append(file_path)
            return True
        return False
    
    def analyze_video(self, file_path: str, frame_skip: int = 5) -> List[DetectionResult]:
        """Analyze entire video file"""
        results = []
        
        try:
            cap = cv2.VideoCapture(file_path)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for performance
                if frame_count % frame_skip == 0:
                    result = self.detection_engine.process_frame(frame)
                    results.append(result)
                
                frame_count += 1
            
            cap.release()
            return results
            
        except Exception as e:
            logger.error(f"Video analysis error: {e}")
            return []

class SystemHealthMonitor:
    """Monitor system health and performance"""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.detection_count = 0
        self.error_count = 0
        self.last_fps_check = time.time()
        self.fps = 0.0
    
    def update_frame_count(self):
        """Update frame processing metrics"""
        self.frame_count += 1
        
        # Calculate FPS every second
        current_time = time.time()
        if current_time - self.last_fps_check >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_check)
            self.frame_count = 0
            self.last_fps_check = current_time
    
    def update_detection_count(self):
        """Update detection metrics"""
        self.detection_count += 1
    
    def update_error_count(self):
        """Update error metrics"""
        self.error_count += 1
    
    def get_health_report(self) -> Dict:
        """Get system health report"""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'uptime_formatted': str(timedelta(seconds=int(uptime))),
            'fps': self.fps,
            'total_detections': self.detection_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(1, self.detection_count),
            'status': 'HEALTHY' if self.error_count < 10 and self.fps > 5 else 'DEGRADED'
        }

def main():
    """Main application entry point"""
    try:
        # Set up application
        app = QApplication(sys.argv)
        app.setApplicationName("Aetharion")
        app.setApplicationVersion("2.0")
        
        # Create and show main window
        window = EnhancedSpaceAssistantApp()
        window.load_settings()  # Load saved settings
        window.show()
        
        # Auto-start systems if configured
        if window.autostart_checkbox.isChecked():
            window.start_live_feed()
            window.start_voice_assistant()
        
        logger.info("Aetherion started successfully")
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application startup error: {e}")
        if 'app' in locals():
            QMessageBox.critical(None, "Startup Error", f"Failed to start application:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()