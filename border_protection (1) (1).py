#!/usr/bin/env python3
"""
Border Protection System - Fixed Version
- Camera only activates for 10 seconds after PIR motion
- Auto-refresh every second
- Saves all detections during active window
"""

from flask import Flask, render_template_string, Response, jsonify
import cv2
import RPi.GPIO as GPIO
from ultralytics import YOLO
import threading
import time
from datetime import datetime
import numpy as np

# ======================== CONFIGURATION ========================
PIR_PIN = 17  # PIR sensor GPIO pin
ULTRASONIC_TRIG = 23  # Ultrasonic TRIG pin
ULTRASONIC_ECHO = 24  # Ultrasonic ECHO pin (use voltage divider!)
BUZZER_PIN = 13  # Buzzer GPIO pin
CAMERA_INDEX = 0  # Usually 0 for USB webcam
CONFIDENCE_THRESHOLD = 0.5
CAMERA_ACTIVE_DURATION = 10  # Camera stays on for 10 seconds after motion
ULTRASONIC_THRESHOLD = 300  # Distance threshold in cm (trigger if closer than this)
BUZZER_DURATION = 2  # Buzzer beep duration in seconds for humans

# ======================== GLOBAL VARIABLES ========================
app = Flask(__name__)
model = YOLO('yolov8n.pt')
camera = None
camera_active = False
camera_end_time = 0
motion_detected = False
detection_log = []  # Store all detections
current_detections = []  # Current frame detections
ultrasonic_distance = 0  # Current distance reading
buzzer_active = False  # Buzzer state
motion_trigger_source = ""  # "PIR", "Ultrasonic", or "Both"

# COCO class names
HUMAN_CLASSES = [0]  # person
ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # animals

# ======================== GPIO SETUP ========================
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)
GPIO.setup(ULTRASONIC_TRIG, GPIO.OUT)
GPIO.setup(ULTRASONIC_ECHO, GPIO.IN)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(ULTRASONIC_TRIG, False)
GPIO.output(BUZZER_PIN, False)

# ======================== ULTRASONIC SENSOR ========================
def get_distance():
    """Measure distance using HC-SR04 ultrasonic sensor"""
    try:
        # Send 10us pulse to trigger
        GPIO.output(ULTRASONIC_TRIG, True)
        time.sleep(0.00001)
        GPIO.output(ULTRASONIC_TRIG, False)
        
        # Wait for echo
        pulse_start = time.time()
        pulse_end = time.time()
        timeout = time.time() + 0.1  # 100ms timeout
        
        while GPIO.input(ULTRASONIC_ECHO) == 0:
            pulse_start = time.time()
            if pulse_start > timeout:
                return -1
        
        while GPIO.input(ULTRASONIC_ECHO) == 1:
            pulse_end = time.time()
            if pulse_end > timeout:
                return -1
        
        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Speed of sound = 34300 cm/s
        distance = round(distance, 2)
        
        return distance if distance < 400 else 400  # Max range ~4m
    except:
        return -1

# ======================== BUZZER CONTROL ========================
def activate_buzzer(duration=2):
    """Activate buzzer for specified duration"""
    global buzzer_active
    buzzer_active = True
    GPIO.output(BUZZER_PIN, True)
    print(f"🔊 BUZZER ACTIVATED for {duration} seconds")
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, False)
    buzzer_active = False
    print("🔇 Buzzer deactivated")

def buzzer_alert_thread():
    """Run buzzer in separate thread to not block main program"""
    thread = threading.Thread(target=activate_buzzer, args=(BUZZER_DURATION,), daemon=True)
    thread.start()

# ======================== CAMERA FUNCTIONS ========================
def initialize_camera():
    """Initialize the camera"""
    global camera
    try:
        camera = cv2.VideoCapture(CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(1)  # Camera warm-up
        print("✓ Camera initialized")
        return True
    except Exception as e:
        print(f"✗ Camera error: {e}")
        return False

def release_camera():
    """Release the camera"""
    global camera, camera_active
    if camera is not None:
        camera.release()
        camera = None
    camera_active = False
    print("✓ Camera released")

def analyze_frame(frame):
    """Analyze frame with YOLO and return detections"""
    global current_detections
    
    # Run YOLO detection
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # Parse results
    detections = []
    annotated_frame = frame.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            detection_type = None
            if cls in HUMAN_CLASSES:
                detection_type = "HUMAN"
                color = (0, 0, 255)  # Red for humans
            elif cls in ANIMAL_CLASSES:
                detection_type = "ANIMAL"
                color = (0, 255, 255)  # Yellow for animals
            
            if detection_type:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{detection_type} {conf*100:.1f}%"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                detections.append({
                    "type": detection_type,
                    "confidence": round(conf * 100, 1),
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                })
    
    current_detections = detections
    return annotated_frame, detections

# ======================== PIR MONITORING ========================
def monitor_pir():
    """Monitor PIR sensor and control camera"""
    global motion_detected, camera_active, camera_end_time, detection_log, motion_trigger_source
    
    print("🔍 PIR monitoring started...")
    
    while True:
        pir_state = GPIO.input(PIR_PIN)
        current_time = time.time()
        
        # PIR detected motion
        if pir_state == 1 and not camera_active:
            print(f"\n{'='*60}")
            print(f"🚨 MOTION DETECTED by PIR at {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            
            motion_detected = True
            camera_active = True
            camera_end_time = current_time + CAMERA_ACTIVE_DURATION
            motion_trigger_source = "PIR"
            
            # Initialize camera
            if initialize_camera():
                print(f"📹 Camera activated for {CAMERA_ACTIVE_DURATION} seconds")
                
                # Clear previous detection log for new session
                detection_log = []
        
        # Check if camera should be turned off
        if camera_active and current_time >= camera_end_time:
            print(f"\n⏰ {CAMERA_ACTIVE_DURATION} seconds elapsed")
            
            # Log summary
            if detection_log:
                print(f"📊 Detection Summary:")
                human_count = sum(1 for d in detection_log if d['type'] == 'HUMAN')
                animal_count = sum(1 for d in detection_log if d['type'] == 'ANIMAL')
                
                if human_count > 0:
                    print(f"   ⚠️  HUMANS DETECTED: {human_count} times")
                if animal_count > 0:
                    print(f"   ℹ️  Animals detected: {animal_count} times")
                
                print(f"   Total detections: {len(detection_log)}")
            else:
                print("   No humans or animals detected")
            
            print(f"{'='*60}\n")
            
            motion_detected = False
            motion_trigger_source = ""
            release_camera()
        
        time.sleep(0.1)

# ======================== ULTRASONIC MONITORING ========================
def monitor_ultrasonic():
    """Monitor ultrasonic sensor for proximity detection"""
    global motion_detected, camera_active, camera_end_time, detection_log, ultrasonic_distance, motion_trigger_source
    
    print("📡 Ultrasonic monitoring started...")
    
    while True:
        distance = get_distance()
        ultrasonic_distance = distance
        
        if distance > 0 and distance < ULTRASONIC_THRESHOLD and not camera_active:
            print(f"\n{'='*60}")
            print(f"📡 PROXIMITY ALERT by Ultrasonic at {datetime.now().strftime('%H:%M:%S')}")
            print(f"   Distance: {distance} cm (Threshold: {ULTRASONIC_THRESHOLD} cm)")
            print(f"{'='*60}")
            
            motion_detected = True
            camera_active = True
            camera_end_time = time.time() + CAMERA_ACTIVE_DURATION
            
            if motion_trigger_source == "PIR":
                motion_trigger_source = "Both"
            else:
                motion_trigger_source = "Ultrasonic"
            
            # Initialize camera
            if initialize_camera():
                print(f"📹 Camera activated for {CAMERA_ACTIVE_DURATION} seconds")
                
                # Clear previous detection log for new session
                detection_log = []
        
        time.sleep(0.2)  # Check every 200ms

# ======================== CONTINUOUS DETECTION ========================
def continuous_detection():
    """Continuously analyze frames when camera is active"""
    global camera_active, detection_log
    
    human_detected_in_session = False  # Track if human detected in this session
    
    while True:
        if camera_active and camera is not None:
            ret, frame = camera.read()
            if ret:
                annotated_frame, detections = analyze_frame(frame)
                
                # Add new detections to log
                for detection in detections:
                    detection_log.append(detection)
                    
                    # Print real-time alerts
                    if detection['type'] == 'HUMAN':
                        print(f"⚠️  ALERT: {detection['type']} detected! "
                              f"Confidence: {detection['confidence']}% at {detection['time']}")
                        
                        # Trigger buzzer only once per session when human first detected
                        if not human_detected_in_session and not buzzer_active:
                            human_detected_in_session = True
                            buzzer_alert_thread()
                    else:
                        print(f"ℹ️  {detection['type']} detected. "
                              f"Confidence: {detection['confidence']}% at {detection['time']}")
        else:
            # Reset human detection flag when camera turns off
            human_detected_in_session = False
        
        time.sleep(0.5)  # Analyze every 0.5 seconds

# ======================== VIDEO STREAMING ========================
def generate_frames():
    """Generate frames for video streaming"""
    while True:
        if camera_active and camera is not None:
            ret, frame = camera.read()
            if ret:
                # Analyze and annotate frame
                annotated_frame, _ = analyze_frame(frame)
                
                # Add timer overlay
                remaining_time = max(0, int(camera_end_time - time.time()))
                cv2.putText(annotated_frame, f"Time remaining: {remaining_time}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', annotated_frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Black screen when inactive
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            text = "Waiting for Motion..."
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (640 - text_size[0]) // 2
            text_y = (480 + text_size[1]) // 2
            cv2.putText(black_frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            
            ret, buffer = cv2.imencode('.jpg', black_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

# ======================== WEB INTERFACE ========================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Border Protection System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            text-align: center;
            opacity: 0.9;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .status-card {
            background: rgba(255, 255, 255, 0.15);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            transition: transform 0.2s;
        }
        
        .status-card:hover {
            transform: translateY(-5px);
        }
        
        .status-card h3 {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 10px;
        }
        
        .status-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .pulse {
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .video-section {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .video-section h2 {
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        #video-feed {
            width: 100%;
            max-width: 800px;
            border-radius: 10px;
            display: block;
            margin: 0 auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        
        .detections-section {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .detections-section h2 {
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .detection-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 4px solid;
            transition: all 0.3s;
        }
        
        .detection-item.human {
            border-left-color: #ff6b6b;
            background: rgba(255, 107, 107, 0.2);
        }
        
        .detection-item.animal {
            border-left-color: #ffd43b;
            background: rgba(255, 212, 59, 0.2);
        }
        
        .detection-item:hover {
            transform: translateX(5px);
        }
        
        .detection-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .detection-type {
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .detection-confidence {
            background: rgba(255, 255, 255, 0.2);
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .detection-time {
            opacity: 0.8;
            font-size: 0.9em;
        }
        
        .indicator {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            display: inline-block;
            margin-left: 10px;
        }
        
        .indicator.active {
            background: #51cf66;
            box-shadow: 0 0 15px #51cf66;
        }
        
        .indicator.inactive {
            background: #868e96;
        }
        
        .empty-state {
            text-align: center;
            padding: 40px;
            opacity: 0.6;
        }
        
        .stats-summary {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
        }
        
        .stat-label {
            opacity: 0.8;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Border Protection System</h1>
        <p class="subtitle">AI-Powered Motion Detection & Classification</p>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>CAMERA STATUS</h3>
                <div class="status-value" id="camera-status">
                    <span id="camera-text">Standby</span>
                    <span class="indicator inactive" id="camera-indicator"></span>
                </div>
            </div>
            
            <div class="status-card">
                <h3>MOTION SENSOR</h3>
                <div class="status-value" id="motion-status">
                    <span id="motion-text">Monitoring</span>
                    <span class="indicator inactive" id="motion-indicator"></span>
                </div>
                <small id="trigger-source"></small>
            </div>
            
            <div class="status-card">
                <h3>ULTRASONIC DISTANCE</h3>
                <div class="status-value" id="distance">-- cm</div>
            </div>
            
            <div class="status-card">
                <h3>BUZZER STATUS</h3>
                <div class="status-value" id="buzzer-status">
                    <span id="buzzer-text">Off</span>
                    <span class="indicator inactive" id="buzzer-indicator"></span>
                </div>
            </div>
            
            <div class="status-card">
                <h3>TIME REMAINING</h3>
                <div class="status-value" id="time-remaining">--</div>
            </div>
            
            <div class="status-card">
                <h3>CURRENT DETECTIONS</h3>
                <div class="status-value" id="current-count">0</div>
            </div>
        </div>
        
        <div class="video-section">
            <h2>📹 Live Camera Feed</h2>
            <img id="video-feed" src="{{ url_for('video_feed') }}" alt="Video Feed">
        </div>
        
        <div class="detections-section">
            <h2>📋 Detection Log (This Session)</h2>
            <div id="detections-list">
                <div class="empty-state">
                    <p>Waiting for motion detection...</p>
                </div>
            </div>
            
            <div class="stats-summary">
                <div class="stat-item">
                    <div class="stat-number" id="human-count">0</div>
                    <div class="stat-label">Humans</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="animal-count">0</div>
                    <div class="stat-label">Animals</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="total-count">0</div>
                    <div class="stat-label">Total</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let lastDetectionCount = 0;
        
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    // Update camera status
                    const cameraText = document.getElementById('camera-text');
                    const cameraIndicator = document.getElementById('camera-indicator');
                    
                    if (data.camera_active) {
                        cameraText.textContent = 'ACTIVE';
                        cameraText.className = 'pulse';
                        cameraIndicator.className = 'indicator active';
                    } else {
                        cameraText.textContent = 'Standby';
                        cameraText.className = '';
                        cameraIndicator.className = 'indicator inactive';
                    }
                    
                    // Update motion status
                    const motionText = document.getElementById('motion-text');
                    const motionIndicator = document.getElementById('motion-indicator');
                    const triggerSource = document.getElementById('trigger-source');
                    
                    if (data.motion_detected) {
                        motionText.textContent = 'DETECTED';
                        motionText.className = 'pulse';
                        motionIndicator.className = 'indicator active';
                        triggerSource.textContent = data.trigger_source ? `via ${data.trigger_source}` : '';
                    } else {
                        motionText.textContent = 'Monitoring';
                        motionText.className = '';
                        motionIndicator.className = 'indicator inactive';
                        triggerSource.textContent = '';
                    }
                    
                    // Update ultrasonic distance
                    const distanceElem = document.getElementById('distance');
                    if (data.distance > 0) {
                        distanceElem.textContent = data.distance + ' cm';
                        if (data.distance < 100) {
                            distanceElem.style.color = '#ff6b6b';
                        } else if (data.distance < 200) {
                            distanceElem.style.color = '#ffd43b';
                        } else {
                            distanceElem.style.color = 'white';
                        }
                    } else {
                        distanceElem.textContent = '-- cm';
                        distanceElem.style.color = 'white';
                    }
                    
                    // Update buzzer status
                    const buzzerText = document.getElementById('buzzer-text');
                    const buzzerIndicator = document.getElementById('buzzer-indicator');
                    
                    if (data.buzzer_active) {
                        buzzerText.textContent = 'ACTIVE';
                        buzzerText.className = 'pulse';
                        buzzerIndicator.className = 'indicator active';
                    } else {
                        buzzerText.textContent = 'Off';
                        buzzerText.className = '';
                        buzzerIndicator.className = 'indicator inactive';
                    }
                    
                    // Update time remaining
                    const timeRemaining = document.getElementById('time-remaining');
                    if (data.camera_active) {
                        timeRemaining.textContent = data.remaining_time + 's';
                    } else {
                        timeRemaining.textContent = '--';
                    }
                    
                    // Update current detections count
                    document.getElementById('current-count').textContent = data.current_detections;
                    
                    // Update detection log
                    if (data.detections.length > 0) {
                        if (data.detections.length !== lastDetectionCount) {
                            const detectionsHtml = data.detections.map(det => `
                                <div class="detection-item ${det.type.toLowerCase()}">
                                    <div class="detection-header">
                                        <span class="detection-type">${det.type === 'HUMAN' ? '⚠️ HUMAN' : '🦌 ANIMAL'}</span>
                                        <span class="detection-confidence">${det.confidence}%</span>
                                    </div>
                                    <div class="detection-time">🕐 ${det.time}</div>
                                </div>
                            `).reverse().join('');
                            
                            document.getElementById('detections-list').innerHTML = detectionsHtml;
                            lastDetectionCount = data.detections.length;
                        }
                        
                        // Update stats
                        document.getElementById('human-count').textContent = data.human_count;
                        document.getElementById('animal-count').textContent = data.animal_count;
                        document.getElementById('total-count').textContent = data.detections.length;
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        // Update every 500ms for smooth refresh
        setInterval(updateStatus, 500);
        updateStatus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Return current system status"""
    remaining_time = max(0, int(camera_end_time - time.time())) if camera_active else 0
    
    # Count humans and animals
    human_count = sum(1 for d in detection_log if d['type'] == 'HUMAN')
    animal_count = sum(1 for d in detection_log if d['type'] == 'ANIMAL')
    
    return jsonify({
        'camera_active': camera_active,
        'motion_detected': motion_detected,
        'remaining_time': remaining_time,
        'detections': detection_log,
        'current_detections': len(current_detections),
        'human_count': human_count,
        'animal_count': animal_count,
        'distance': ultrasonic_distance,
        'buzzer_active': buzzer_active,
        'trigger_source': motion_trigger_source
    })

# ======================== MAIN ========================
if __name__ == '__main__':
    import socket
    
    print("\n" + "="*60)
    print("🛡️  BORDER PROTECTION SYSTEM - ENHANCED")
    print("="*60)
    print(f"PIR Sensor Pin: GPIO {PIR_PIN}")
    print(f"Ultrasonic TRIG Pin: GPIO {ULTRASONIC_TRIG}")
    print(f"Ultrasonic ECHO Pin: GPIO {ULTRASONIC_ECHO}")
    print(f"Buzzer Pin: GPIO {BUZZER_PIN}")
    print(f"Camera Index: {CAMERA_INDEX}")
    print(f"Camera Active Duration: {CAMERA_ACTIVE_DURATION} seconds")
    print(f"Ultrasonic Threshold: {ULTRASONIC_THRESHOLD} cm")
    print(f"Buzzer Duration: {BUZZER_DURATION} seconds")
    print("Loading YOLOv8 model...")
    print("="*60 + "\n")
    
    # Start PIR monitoring thread
    pir_thread = threading.Thread(target=monitor_pir, daemon=True)
    pir_thread.start()
    
    # Start ultrasonic monitoring thread
    ultrasonic_thread = threading.Thread(target=monitor_ultrasonic, daemon=True)
    ultrasonic_thread.start()
    
    # Start continuous detection thread
    detection_thread = threading.Thread(target=continuous_detection, daemon=True)
    detection_thread.start()
    
    # Get IP address
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
    except:
        ip_address = "localhost"
    
    print(f"🌐 Web Interface Available at:")
    print(f"   → http://{ip_address}:5000")
    print(f"   → http://localhost:5000\n")
    print("Press Ctrl+C to stop\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Shutting down gracefully...")
        print("="*60)
        GPIO.cleanup()
        release_camera()
        print("✓ System stopped\n")
