# Indoor Navigation Demo - Complete Setup Guide

## 🎯 What This Creates

A **dedicated indoor navigation system** with two separate scripts:

1. **Pi**: `navigation_camera_service.py` - Only captures + uploads images
2. **Mac**: `indoor_navigation_agent.py` - Voice interaction + position detection

## 📋 Setup Steps

### 1. **Convex Setup** (Add Navigation Functions)
```javascript
// Add the functions from convex_functions/navigation.js to your Convex project
// This creates dedicated navigation endpoints separate from your main sensor data
```

### 2. **Pi Setup** (Lightweight Camera Service)
```bash
# On your Raspberry Pi
cd /path/to/vision-assistant/pi-service

# Set environment variable
export CONVEX_DEPLOYMENT_URL="https://your-deployment.convex.cloud"

# Test camera first
python3 navigation_camera_service.py --test

# Run the service
python3 navigation_camera_service.py
```

### 3. **Mac Setup** (Navigation Agent)
```bash
# On your Mac
cd /Users/radhikadanda/vision-assistant

# Install dependencies
pip install -r requirements_navigation.txt

# Set up environment
cp .env.example .env
# Edit .env with your Convex URL and API keys

# Add your reference photos
mkdir -p reference_images/tan_oak
# Copy your 20-25 TAN OAK room photos here

# Process photos (one-time)
python photo_analyzer.py

# Run navigation demo
python indoor_navigation_agent.py
```

## 🔄 How It Works

### **Simple Flow**:
```
Pi Camera → navigation_camera_service.py → Convex → indoor_navigation_agent.py → Voice Response
```

### **Detailed Flow**:
1. **Pi captures image** every 2 seconds
2. **Uploads to Convex** via `upload_navigation_image`
3. **Mac agent gets latest image** via `get_latest_navigation_image`
4. **Finds similar reference photo** using CLIP embeddings
5. **Responds with position** via LiveKit voice

## 🎙️ Voice Commands

**Position Detection:**
- "Where am I?"
- "What's my location?"
- "Where am I in the room?"

**Exit Directions:**
- "How do I exit?"
- "How do I get out?"
- "Where's the door?"

## 📸 Expected Results

### **High Confidence (Photo Match)**
```
User: "Where am I?"
Agent: "You're sitting at the front left chair facing the projector screen. 
        Confidence: High (89% similarity match)"
```

### **Exit Directions**
```
User: "How do I exit?"
Agent: "From your current position, turn right 90 degrees and walk straight 
        toward the door. It should be about 6 steps ahead."
```

## 🛠️ Benefits of Separate Scripts

### **Pi Script** (`navigation_camera_service.py`)
✅ **Lightweight** - Only camera capture + upload  
✅ **No AI processing** - Saves Pi resources  
✅ **Dedicated endpoints** - Clean separation from sensor data  
✅ **Simple debugging** - Easy to test camera independently  

### **Mac Script** (`indoor_navigation_agent.py`)  
✅ **Full AI power** - CLIP + GPT-4V for accurate positioning  
✅ **Voice interaction** - LiveKit STT/TTS  
✅ **Photo similarity** - Uses your actual room photos  
✅ **Fallback systems** - Multiple positioning methods  

## 🧪 Testing

### **Test Pi Camera**
```bash
# Test camera capture
python3 navigation_camera_service.py --test

# Check service status
curl -X POST https://your-deployment.convex.cloud/get_navigation_service_status
```

### **Test Mac Navigation**
```bash
# Test photo processing
python photo_analyzer.py

# Check reference photos loaded
ls reference_images/tan_oak/

# Run navigation agent
python indoor_navigation_agent.py
```

## 🔧 Troubleshooting

### **Pi Issues**
- **Camera not working**: Try `--test` flag first
- **Upload failing**: Check `CONVEX_DEPLOYMENT_URL`
- **No images in Convex**: Verify navigation functions deployed

### **Mac Issues**  
- **No recent images**: Check Pi service is running
- **Low confidence**: Add more reference photos
- **Voice not working**: Verify LiveKit credentials

## 🚀 Demo Script

### **Setup**
1. Start Pi service: `python3 navigation_camera_service.py`
2. Start Mac agent: `python indoor_navigation_agent.py`
3. Join LiveKit room
4. Test voice commands

### **Demo Points**
- **Dedicated navigation system** - Separate from general sensors
- **Photo-based positioning** - Uses actual room photos for accuracy
- **Voice-first interface** - Natural speech interaction
- **Real-time analysis** - Immediate position feedback
- **Scalable design** - Easy to add more rooms

This creates a **clean, dedicated navigation demo** that's completely separate from your existing sensor services!
