# Aetherion - Advanced Space Assistant AI
<img width="1385" height="939" alt="image" src="https://github.com/user-attachments/assets/3d6715c5-cc0d-4ff4-87cd-2e7158bb3a6d" />
<img width="1391" height="948" alt="image" src="https://github.com/user-attachments/assets/8381ca49-c603-415c-8f56-04353cb6fcbd" />
<img width="1395" height="952" alt="image" src="https://github.com/user-attachments/assets/e8a02a49-42a4-4625-979f-99945d05218b" />
<img width="1395" height="953" alt="image" src="https://github.com/user-attachments/assets/55702c32-984a-4393-b584-b6373005e38e" />
<img width="1386" height="950" alt="image" src="https://github.com/user-attachments/assets/8f3897aa-e717-4bcb-83ae-c1dfa3355d94" />

Video demo (AI):
https://docs.google.com/videos/d/1nUyW7i3wTwzIyGSHB6F7lL0VHsXKAxl3KQ9_Nu6pJeU/edit?usp=sharing

Video demo (Server):
https://drive.google.com/file/d/1nzUIjKKAAcBH4zMQ6_3gMrfogxRRdmuX/view?usp=sharing 


**Aetherion** is an advanced AI-powered space assistant designed for astronaut monitoring, emergency detection, and automated task execution. The system provides real-time health monitoring, unconsciousness detection, voice interaction, and emergency medical guidance.

## 🚀 Features

### Core Monitoring
- **Real-time Video Analysis**: Live camera feed with AI-powered pose detection
- **Enhanced Unconsciousness Detection**: Advanced movement analysis with temporal confirmation
- **Object Recognition**: Custom-trained models for space equipment detection
- **Stress Level Monitoring**: Facial emotion analysis and stress detection
- **Breathing Detection**: Subtle movement analysis for respiratory monitoring

### Voice Assistant
- **Wake Word Activation**: Multi-wake word support ("houston", "assistant", "help")
- **Natural Language Processing**: Powered by Groq AI for intelligent responses
- **Voice Commands**: System control and general queries
- **Text-to-Speech**: Audio feedback and alerts

### Emergency Response
- **Intelligent Alert System**: Single alert per incident with cooldown management
- **Medical AI Guidance**: Structured emergency medical assessments
- **Auto Tab Switching**: Automatic UI navigation to medical guidance for live emergencies
- **Mission Control Integration**: Emergency communication protocols

### Automation & Integration
- **Task Execution**: Server-based automation for complex tasks
- **Sensor Integration**: HTTP server for IoT sensor data collection
- **Gas Leak Detection**: Automated sensor monitoring and alerting
- **Batch Processing**: Multiple image analysis capabilities

## 🛠️ Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required system packages (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-pip python3-tk portaudio19-dev

# For Windows users, install Visual C++ Build Tools
```

### Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

**requirements.txt:**
```txt
torch>=1.11.0
torchvision>=0.12.0
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
PyQt5>=5.15.0
pyttsx3>=2.90
sounddevice>=0.4.4
vosk>=0.3.32
requests>=2.26.0
```

### Model Setup

1. **Download YOLO Models** (automatic on first run):
   - YOLOv8 pose model
   - YOLOv8 object detection model

2. **Vosk Speech Recognition Models**:
   ```bash
   # Download and extract Vosk models
   mkdir model
   cd model
   
   # For better accuracy (recommended)
   wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
   unzip vosk-model-en-us-0.22.zip
   
   # For faster processing
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   unzip vosk-model-small-en-us-0.15.zip
   ```

3. **Custom Models** (optional):
   - Place custom YOLO models in specified path
   - Update model path in configuration

## ⚙️ Configuration

### API Setup

1. **Groq API Key** (required for AI features):
   - Sign up at [Groq Console](https://console.groq.com/)
   - Get your API key
   - Enter in Settings tab or update code directly

### System Configuration

```python
# Update these in the code for your setup
SERVER_IP = "your-server-ip"    # Automation server IP
SERVER_PORT = 8000              # Automation server port
SENSORS = {
    "gas_sensor_1": "sensor-ip" # Add your sensor IPs
}
```

## 🎯 Usage

### Starting Aetherion

```bash
python aetherion.py
```

### Basic Workflow

1. **Launch Application**: Run the Python script
2. **Configure Settings**: Set API keys and device preferences
3. **Start Live Feed**: Begin camera monitoring
4. **Activate Voice**: Start voice assistant for commands
5. **Monitor Alerts**: Check Alerts & Medical tab for emergencies

### Voice Commands

| Command | Function |
|---------|----------|
| `"houston start monitoring"` | Begin live camera feed |
| `"houston stop monitoring"` | Stop monitoring systems |
| `"houston check status"` | System status report |
| `"houston emergency"` | Trigger emergency alert |
| `"houston medical help"` | Get medical assistance |

### Tab Overview

- **🎥 Live Feed**: Real-time camera monitoring
- **🎙️ Voice Assistant**: Voice interaction and commands  
- **📷 Image Analysis**: Static image upload and analysis
- **🚨 Alerts & Medical**: Emergency alerts and medical guidance
- **🤖 Automation**: Task execution and sensor monitoring
- **⚙️ Settings**: Configuration and preferences
- **📋 Logs**: System logs and debugging

## 🔬 Technical Architecture

### Detection Engine
- **Movement Analysis**: 15-frame keypoint tracking for unconsciousness detection
- **Multi-model Approach**: YOLO pose + object detection + face analysis
- **Temporal Confirmation**: Multi-frame validation to reduce false positives
- **Alert Management**: Cooldown system prevents alert spam

### AI Integration
- **Medical AI**: Groq-powered medical assessment and guidance
- **General AI**: Natural language processing for voice commands
- **Model Fallback**: Multiple model candidates for reliability

### Performance Features
- **Multi-threading**: Separate threads for detection, voice, and UI
- **Queue Management**: Frame buffering for smooth processing
- **Resource Optimization**: GPU acceleration when available

## 🚨 Emergency Detection

### Unconsciousness Detection Algorithm

1. **Pose Analysis**: Head tilt, body position, shoulder alignment
2. **Movement Tracking**: 15-frame keypoint displacement analysis
3. **Temporal Validation**: Multi-frame confirmation (4+ frames)
4. **Scoring System**: Weighted combination of pose + movement factors

### Alert Workflow

```
Live Feed Detection → Movement Analysis → Confidence Check → 
Single Alert → Auto Tab Switch → Medical AI Assessment → 
Emergency Guidance Display
```

## 📊 Medical AI Features

### Emergency Assessment Structure
- **Immediate Actions**: Critical steps to take now
- **Severity Assessment**: 1-10 scale with reasoning
- **Potential Causes**: Most likely medical causes
- **Monitoring Instructions**: What to watch for
- **Mission Control Alert**: Contact recommendations

### Response Time
- Typical medical assessment: 3-10 seconds
- Live feed alert trigger: < 1 second
- Voice command processing: 1-3 seconds

## 🔧 Customization

### Detection Thresholds
```python
# Adjust in Settings tab or code
unconscious_threshold = 3.0      # seconds before alert
stress_threshold = 7.0           # 1-10 scale
movement_threshold = 3.0         # pixels minimum movement
```

### Adding Custom Sensors
```python
SENSORS = {
    "gas_sensor_1": "192.168.1.20",
    "smoke_detector": "192.168.1.21",
    "temperature": "192.168.1.22"
}
```

## 🐛 Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# Check available cameras
python -c "import cv2; print([cv2.VideoCapture(i).read()[0] for i in range(4)])"
```

**Voice recognition not working:**
- Verify microphone permissions
- Check audio device selection in Settings
- Ensure Vosk models are downloaded

**AI responses failing:**
- Verify Groq API key is valid
- Check internet connection
- Try different model candidates in code

**High CPU usage:**
- Enable GPU acceleration in Settings
- Reduce frame processing rate
- Use smaller YOLO models

### Debug Mode
```bash
# Run with verbose logging
