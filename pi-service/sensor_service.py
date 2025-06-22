#!/usr/bin/env python3
"""
Simple SentientSight Camera AI Service with TTS - Testing Only (FIXED VERSION)
Just camera → AI inference → alerts → TTS for hazards

FIXES APPLIED:
- Added aiohttp.ClientSession for Cartesia TTS plugin
- Replaced deprecated FastAPI on_event with lifespan context manager
- Proper HTTP session cleanup
- Fixed TTS HTTP session context error
"""

import asyncio
import time
import io
import os
import json
import wave
import tempfile
import subprocess
from datetime import datetime
from contextlib import asynccontextmanager

# Environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not installed - install with: pip install python-dotenv")

# FastAPI imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# HTTP client for TTS
import aiohttp

# Ultrasonic sensor for distance measurements
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    print("✅ RPi.GPIO imported successfully")
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️ RPi.GPIO not available - ultrasonic sensor disabled")

# AI imports
from google import genai
from google.genai import types

# Camera import
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    print("⚠️ picamera2 not available - camera disabled")
    CAMERA_AVAILABLE = False

# LiveKit TTS imports
try:
    from livekit import agents, rtc
    from livekit.agents.tts import SynthesizedAudio
    from livekit.plugins import cartesia
    from typing import AsyncIterable
    LIVEKIT_AVAILABLE = True
except ImportError:
    print("⚠️ LiveKit not available - TTS disabled")
    print("💡 Install: pip install livekit-agents livekit-plugins-cartesia")
    LIVEKIT_AVAILABLE = False

# Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
AI_ANALYSIS_INTERVAL = 3.0  # Analyze every 3 seconds

# AI Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = "gemini-2.5-flash"

# TTS Configuration
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')
LIVEKIT_URL = os.getenv('LIVEKIT_URL', 'wss://your-livekit-server.com')
TTS_MODEL = "sonic-english"

# Ultrasonic Sensor Configuration (HC-SR04)
TRIG_PIN = 23  # GPIO pin for trigger
ECHO_PIN = 24  # GPIO pin for echo
MAX_DISTANCE = 400  # Maximum distance in cm
DISTANCE_TIMEOUT = 0.04  # 40ms timeout for echo

class SimpleCameraAI:
    def __init__(self):
        self.camera = None
        self.ai_client = None
        self.tts = None
        self.http_session = None  # HTTP session for TTS plugin
        self.room = None
        self.running = False
        
        # State
        self.ai_enabled = False
        self.tts_enabled = False
        self.last_ai_analysis = 0
        self.current_ai_result = None
        self.analysis_count = 0
        self.last_spoken_message = None
        self.last_tts_time = 0
        
        # Ultrasonic sensor state
        self.ultrasonic_enabled = False
        self.last_distance = None
        self.distance_readings = []  # For averaging
        
        # TTS Settings
        self.tts_cooldown = 2.0  # Minimum 2 seconds between TTS messages
        self.repeat_threshold = 5.0  # Repeat same message after 5 seconds
        
        print("📷 Simple Camera AI Service with TTS Starting...")
        
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(TRIG_PIN, GPIO.OUT)
            GPIO.setup(ECHO_PIN, GPIO.IN)
            self.ultrasonic_enabled = True
            print("📏 Ultrasonic sensor initialized")
        
    def setup_ai_client(self):
        """Initialize Gemini AI"""
        if not GEMINI_API_KEY:
            print("❌ GEMINI_API_KEY environment variable not set!")
            print("💡 Run: export GEMINI_API_KEY='your_api_key_here'")
            self.ai_enabled = False
            return
            
        try:
            self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
            print("🧠 Gemini AI: ✅ Connected")
            self.ai_enabled = True
        except Exception as e:
            print(f"🧠 Gemini AI: ❌ Failed - {e}")
            self.ai_enabled = False

    async def setup_tts(self):
        """Initialize LiveKit TTS with HTTP session fix"""
        if not LIVEKIT_AVAILABLE:
            print("🔊 TTS: ❌ LiveKit not available")
            self.tts_enabled = False
            return
            
        if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
            print("❌ LIVEKIT_API_KEY or LIVEKIT_API_SECRET not set!")
            print("💡 Set: export LIVEKIT_API_KEY='your_key' LIVEKIT_API_SECRET='your_secret'")
            self.tts_enabled = False
            return
            
        try:
            # Create HTTP session for TTS plugin (FIXES THE ERROR)
            self.http_session = aiohttp.ClientSession()
            
            # Initialize TTS with HTTP session
            self.tts = cartesia.TTS(model=TTS_MODEL, http_session=self.http_session)
            
            print("🔊 TTS: ✅ LiveKit TTS initialized with HTTP session (direct audio)")
            self.tts_enabled = True
            
        except Exception as e:
            print(f"🔊 TTS: ❌ Failed - {e}")
            self.tts_enabled = False
        
    def setup_camera(self):
        """Initialize camera"""
        if not CAMERA_AVAILABLE:
            print("📷 Camera: ❌ picamera2 not available")
            return
            
        try:
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(main={"size": (640, 480)})
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)  # Let camera warm up
            print("📷 Camera: ✅ Initialized")
        except Exception as e:
            print(f"📷 Camera: ❌ Failed - {e}")
            self.camera = None
        
    def read_distance(self) -> float:
        """Read distance from ultrasonic sensor (HC-SR04)"""
        if not self.ultrasonic_enabled or not GPIO_AVAILABLE:
            return None
            
        try:
            # Trigger pulse
            GPIO.output(TRIG_PIN, GPIO.HIGH)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(TRIG_PIN, GPIO.LOW)
            
            # Wait for echo start
            start_time = time.time()
            while GPIO.input(ECHO_PIN) == GPIO.LOW:
                if time.time() - start_time > DISTANCE_TIMEOUT:
                    return None
                    
            pulse_start = time.time()
            
            # Wait for echo end
            while GPIO.input(ECHO_PIN) == GPIO.HIGH:
                if time.time() - pulse_start > DISTANCE_TIMEOUT:
                    return None
                    
            pulse_end = time.time()
            
            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150  # Speed of sound / 2 (round trip)
            
            # Validate reading
            if 2 <= distance <= MAX_DISTANCE:
                return round(distance, 1)
            else:
                return None
                
        except Exception as e:
            print(f"📏 Ultrasonic sensor error: {e}")
            return None
    
    def get_averaged_distance(self) -> float:
        """Get averaged distance from multiple readings"""
        if not self.ultrasonic_enabled:
            return None
            
        # Take 3 quick readings for accuracy
        readings = []
        for _ in range(3):
            distance = self.read_distance()
            if distance is not None:
                readings.append(distance)
            time.sleep(0.1)  # Brief delay between readings
        
        if readings:
            avg_distance = sum(readings) / len(readings)
            self.last_distance = round(avg_distance, 1)
            
            # Keep last 5 readings for trend analysis
            self.distance_readings.append(self.last_distance)
            if len(self.distance_readings) > 5:
                self.distance_readings.pop(0)
                
            return self.last_distance
        
        return None
    
    def get_distance_description(self, distance: float) -> str:
        """Get human-readable distance description"""
        if distance is None:
            return "unknown distance"
        
        if distance < 30:
            return f"very close at {distance:.0f} centimeters"
        elif distance < 100:
            return f"close at {distance:.0f} centimeters"
        elif distance < 200:
            return f"{distance:.0f} centimeters away"
        else:
            return f"far at {distance:.0f} centimeters"

    def capture_image(self) -> bytes:
        """Capture image from camera"""
        if not self.camera:
            # Return a dummy response for testing without camera
            return b"no_camera_data"
            
        try:
            stream = io.BytesIO()
            self.camera.capture_file(stream, format='jpeg')
            stream.seek(0)
            return stream.getvalue()
        except Exception as e:
            print(f"⚠️ Camera capture error: {e}")
            return None
    
    async def analyze_with_ai(self, image_data: bytes) -> dict:
        """AI analysis with simplified structured output"""
        if not self.ai_enabled or not self.ai_client:
            return {
                "type": "neutral",
                "message": "AI analysis not available",
                "error": "AI not available"
            }
            
        if image_data == b"no_camera_data":
            return {
                "type": "neutral",
                "message": "Camera not available - test mode",
                "test_mode": True
            }
            
        try:
            # Get current distance reading
            current_distance = self.get_averaged_distance()
            distance_info = ""
            
            if current_distance is not None:
                distance_desc = self.get_distance_description(current_distance)
                distance_info = f"\n\nULTRASONIC SENSOR DATA:\n- Object detected {distance_desc}\n- This sensor measures the closest object directly ahead"
            else:
                distance_info = "\n\nULTRASONIC SENSOR DATA:\n- No reliable distance reading available"
            
            # Visual-first analysis prompt with distance as confirmation
            prompt = f"""You are an AI assistant helping a blind person navigate safely. 

Analyze this first-person camera view and identify any immediate hazards or navigation information.
{distance_info}

**ANALYSIS PRIORITY:**
1. **PRIMARY**: Analyze the camera image for visual hazards and describe in a concise way what you see(stairs, walls, people, etc...). dont mention the color as we are trying to help blind persons.
2. **SECONDARY**: Use distance sensor data ONLY to confirm or enhance visually detected hazards

**DISTANCE SENSOR RULES:**
- If visual path is CLEAR: Ignore distance readings >50cm (don't mention distant walls/objects)
- If visual HAZARD detected: Use distance to provide precise measurements
- If distance <30cm but no visual hazard: Mention potential obstacle (glass, thin pole, etc.)

Respond with JSON only in this EXACT format:
{{
  "type": "danger|warning|neutral",
  "message": "Clear, direct message for text-to-speech (under 20 words)"
}}

**RESPONSE GUIDELINES:**
- **danger**: Immediate visual threats OR distance <30cm with potential collision risk
- **warning**: Visual hazards ahead OR approaching people/stairs
- **neutral**: Clear visual path (ignore distant objects >50cm)

**EXAMPLES:**
- Visual wall + close distance: {{"type": "danger", "message": "Wall directly ahead at 25 centimeters, stop"}}
- Visual clear + distant object: {{"type": "neutral", "message": "Path clear ahead"}}
- Visual stairs: {{"type": "warning", "message": "Stairs going down ahead, slow down"}}
- No visual hazard + very close reading: {{"type": "warning", "message": "Possible obstacle very close at 15 centimeters"}}

Focus on what the CAMERA shows first,describe what you see , and use distance sensor only to enhance hazard details."""

            # Create image part
            image_part = types.Part.from_bytes(data=image_data, mime_type='image/jpeg')
            
            # Make AI request
            response = self.ai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0)  # Fast mode
                )
            )
            
            # Parse response
            json_str = response.text.strip()
            if json_str.startswith('```'):
                lines = json_str.split('\n')
                json_str = '\n'.join(lines[1:-1])
            
            ai_result = json.loads(json_str)
            
            # Validate and ensure required fields
            if "type" not in ai_result:
                ai_result["type"] = "neutral"
            if "message" not in ai_result:
                ai_result["message"] = "Analysis complete"
                
            # Ensure type is valid
            if ai_result["type"] not in ["danger", "warning", "neutral"]:
                ai_result["type"] = "neutral"
            
            # Add metadata
            ai_result['timestamp'] = time.time()
            ai_result['analysis_id'] = self.analysis_count
            ai_result['sensor_distance_cm'] = current_distance
            
            # Log with distance info
            distance_log = f" (📏 {current_distance}cm)" if current_distance else " (📏 no distance)"
            type_emoji = {"danger": "🚨", "warning": "⚠️", "neutral": "✅"}
            emoji = type_emoji.get(ai_result["type"], "ℹ️")
            print(f"🧠 AI Result #{self.analysis_count}: {emoji} {ai_result['type'].upper()} - {ai_result['message']}{distance_log}")
            
            return ai_result
            
        except Exception as e:
            print(f"❌ AI analysis error: {e}")
            return {
                "type": "neutral",
                "message": "Analysis failed",
                "error": str(e),
                "sensor_distance_cm": self.last_distance
            }

    async def speak_message(self, message: str, urgency: str = "medium"):
        """Convert text to speech using LiveKit TTS with direct audio playback"""
        if not self.tts_enabled or not self.tts:
            print(f"🔊 TTS disabled - would say: '{message}'")
            return
            
        current_time = time.time()
        
        # Check cooldown
        if (current_time - self.last_tts_time) < self.tts_cooldown:
            print(f"🔊 TTS cooldown active - skipping: '{message}'")
            return
            
        # Check if same message was recently spoken
        if (self.last_spoken_message == message and 
            (current_time - self.last_tts_time) < self.repeat_threshold):
            print(f"🔊 Same message recently spoken - skipping: '{message}'")
            return
            
        try:
            print(f"🔊 Speaking ({urgency}): '{message}'")
            
            # Create TTS stream
            tts_stream = self.tts.stream()
            tts_stream.push_text(message)
            tts_stream.end_input()
            
            # Collect audio data
            audio_data = []
            async for audio_chunk in tts_stream:
                audio_data.append(audio_chunk.frame.data)
            
            if audio_data:
                # Save to temporary WAV file and play
                await self._play_audio_direct(audio_data)
            
            # Update state
            self.last_spoken_message = message
            self.last_tts_time = current_time
            
            print(f"🔊 TTS completed: '{message}'")
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
    
    async def _play_audio_direct(self, audio_data: list):
        """Play audio data directly through system speakers"""
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
                
                # Write WAV file
                with wave.open(temp_filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(24000)  # 24kHz (Cartesia format)
                    
                    # Combine all audio data
                    combined_audio = b''.join(audio_data)
                    wav_file.writeframes(combined_audio)
            
            # Play through system speakers
            try:
                process = await asyncio.create_subprocess_exec(
                    "aplay", temp_filename,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await process.communicate()
            except Exception as e:
                print(f"⚠️ Audio playback error: {e}")
            
            # Cleanup temp file
            try:
                os.unlink(temp_filename)
            except:
                pass
                
        except Exception as e:
            print(f"❌ Direct audio error: {e}")
    
    def should_speak_result(self, ai_result: dict) -> bool:
        """Determine if AI result should trigger TTS"""
        if not ai_result:
            return False
            
        urgency = ai_result.get('type', 'neutral')
        
        # Only speak for hazards (danger/warning)
        if urgency in ['danger', 'warning']:
            return True
            
        # Don't speak for low urgency or clear paths
        return False
    
    async def run_continuous_analysis(self):
        """Continuously capture and analyze images"""
        print("🔄 Starting continuous AI analysis...")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for analysis
                if (current_time - self.last_ai_analysis) >= AI_ANALYSIS_INTERVAL:
                    
                    # Capture image
                    print(f"📸 Capturing image #{self.analysis_count}...")
                    image_data = self.capture_image()
                    
                    if image_data:
                        # Analyze with AI
                        print(f"🧠 Analyzing with AI...")
                        ai_result = await self.analyze_with_ai(image_data)
                        
                        # Store result
                        self.current_ai_result = ai_result
                        self.last_ai_analysis = current_time
                        self.analysis_count += 1
                        
                        # Log result
                        urgency_icon = {"danger": "🚨", "warning": "⚠️", "neutral": "✅"}.get(ai_result.get('type', 'neutral'), "⚪")
                        print(f"{urgency_icon} Analysis #{self.analysis_count}: {ai_result.get('type').upper()} - '{ai_result.get('message')}'")
                        
                        # Trigger TTS for hazards
                        if self.should_speak_result(ai_result):
                            message = ai_result.get('message', 'Hazard detected')
                            urgency = ai_result.get('type', 'medium')
                            await self.speak_message(message, urgency)
                    
                await asyncio.sleep(0.5)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                print(f"⚠️ Analysis loop error: {e}")
                await asyncio.sleep(1)
    
    async def cleanup(self):
        """Clean up resources with proper HTTP session closure"""
        print("🧹 Cleaning up...")
        self.running = False
        
        # Close camera
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                self.camera = None
                print("📷 Camera closed")
            except Exception as e:
                print(f"⚠️ Camera cleanup error: {e}")
        
        # Cleanup GPIO
        if self.ultrasonic_enabled and GPIO_AVAILABLE:
            try:
                GPIO.cleanup()
                print("📏 GPIO cleaned up")
            except Exception as e:
                print(f"⚠️ GPIO cleanup error: {e}")
        
        # Close HTTP session for TTS
        if self.http_session:
            try:
                await self.http_session.close()
                print("🌐 HTTP session closed")
            except Exception as e:
                print(f"⚠️ HTTP session cleanup error: {e}")
        
        print("✅ Cleanup complete")
        
# Initialize service
service = SimpleCameraAI()

# Modern FastAPI lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Simple Camera AI Service with TTS...")
    service.setup_ai_client()
    await service.setup_tts()
    service.setup_camera()
    service.running = True
    
    # Start background analysis
    asyncio.create_task(service.run_continuous_analysis())
    print("✅ Service started!")
    
    yield
    
    # Shutdown
    await service.cleanup()

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Simple SentientSight Camera AI with TTS (FIXED)",
    description="Basic camera AI with text-to-speech for hazard alerts - HTTP session error fixed",
    version="1.1.1",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    """API info"""
    return {
        "service": "Simple SentientSight Camera AI with TTS (FIXED)",
        "version": "1.1.1",
        "status": "running",
        "fixes_applied": [
            "HTTP session for Cartesia TTS plugin",
            "FastAPI lifespan context manager", 
            "Proper resource cleanup",
            "Direct audio playback",
            "Ultrasonic sensor integration"
        ],
        "ai_enabled": service.ai_enabled,
        "tts_enabled": service.tts_enabled,
        "camera_available": service.camera is not None,
        "ultrasonic_enabled": service.ultrasonic_enabled,
        "current_distance_cm": service.last_distance,
        "analysis_count": service.analysis_count,
        "endpoints": {
            "current_analysis": "/analysis",
            "latest_alert": "/alert", 
            "test_analysis": "/test",
            "test_tts": "/test-tts",
            "tts_settings": "/tts",
            "distance": "/distance",
            "health": "/health"
        }
    }

@app.get("/analysis")
async def get_current_analysis():
    """Get the latest AI analysis"""
    if not service.current_ai_result:
        return {
            "status": "no_analysis_yet",
            "message": "No analysis performed yet, wait a few seconds",
            "next_analysis_in": AI_ANALYSIS_INTERVAL - (time.time() - service.last_ai_analysis) if service.last_ai_analysis > 0 else AI_ANALYSIS_INTERVAL
        }
    
    return {
        "status": "analysis_available",
        "latest_analysis": service.current_ai_result,
        "analysis_count": service.analysis_count,
        "time_since_analysis": time.time() - service.last_ai_analysis,
        "next_analysis_in": AI_ANALYSIS_INTERVAL - (time.time() - service.last_ai_analysis)
    }

@app.get("/alert")
async def get_current_alert():
    """Get current alert if urgent"""
    ai_result = service.current_ai_result
    
    if ai_result and service.should_speak_result(ai_result):
        return {
            "alert_active": True,
            "hazard_type": ai_result.get('type'),
            "message": ai_result.get('message'),
            "urgency": ai_result.get('type'),
            "timestamp": ai_result.get('timestamp'),
            "spoken": service.should_speak_result(ai_result),
            "tts_enabled": service.tts_enabled
        }
    else:
        return {
            "alert_active": False,
            "status": "path_clear",
            "last_analysis": ai_result.get('message', 'No urgent alerts') if ai_result else 'No analysis yet'
        }

@app.get("/test")
async def test_analysis():
    """Force a new analysis right now (for testing)"""
    if not service.ai_enabled:
        return {"error": "AI not enabled", "check": "GEMINI_API_KEY environment variable"}
    
    try:
        print("🧪 Manual test analysis requested...")
        image_data = service.capture_image()
        
        if image_data:
            ai_result = await service.analyze_with_ai(image_data)
            service.current_ai_result = ai_result
            service.analysis_count += 1
            
            # Test TTS if hazard detected
            tts_triggered = False
            if service.should_speak_result(ai_result):
                message = ai_result.get('message', 'Test hazard')
                urgency = ai_result.get('type', 'medium')
                await service.speak_message(message, urgency)
                tts_triggered = True
            
            return {
                "test_complete": True,
                "ai_analysis": ai_result,
                "tts_triggered": tts_triggered,
                "message": "Manual analysis completed successfully"
            }
        else:
            return {"error": "Failed to capture image"}
            
    except Exception as e:
        return {"error": f"Test failed: {e}"}

@app.post("/test-tts")
async def test_tts(message: str = "Test message from navigation AI"):
    """Test TTS with a custom message"""
    if not service.tts_enabled:
        return {
            "error": "TTS not enabled", 
            "check": "LIVEKIT_API_KEY and LIVEKIT_API_SECRET environment variables"
        }
    
    try:
        await service.speak_message(message, "medium")
        return {
            "tts_test_complete": True,
            "message_spoken": message,
            "tts_enabled": service.tts_enabled,
            "http_session_active": service.http_session is not None
        }
    except Exception as e:
        return {"error": f"TTS test failed: {e}"}

@app.get("/tts")
async def get_tts_settings():
    """Get TTS configuration and status"""
    return {
        "tts_enabled": service.tts_enabled,
        "tts_cooldown": service.tts_cooldown,
        "repeat_threshold": service.repeat_threshold,
        "last_spoken_message": service.last_spoken_message,
        "last_tts_time": service.last_tts_time,
        "time_since_last_tts": time.time() - service.last_tts_time if service.last_tts_time > 0 else None,
        "model": TTS_MODEL,
        "livekit_available": LIVEKIT_AVAILABLE,
        "http_session_active": service.http_session is not None
    }

@app.get("/distance")
async def get_distance():
    """Get current distance reading from ultrasonic sensor"""
    if not service.ultrasonic_enabled:
        return {
            "error": "Ultrasonic sensor not available",
            "ultrasonic_enabled": False,
            "gpio_available": GPIO_AVAILABLE
        }
    
    # Get fresh distance reading
    current_distance = service.get_averaged_distance()
    
    return {
        "ultrasonic_enabled": service.ultrasonic_enabled,
        "current_distance_cm": current_distance,
        "distance_description": service.get_distance_description(current_distance),
        "last_distance_cm": service.last_distance,
        "recent_readings": service.distance_readings[-3:] if service.distance_readings else [],
        "sensor_config": {
            "trig_pin": TRIG_PIN,
            "echo_pin": ECHO_PIN,
            "max_distance_cm": MAX_DISTANCE,
            "timeout_seconds": DISTANCE_TIMEOUT
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "version": "1.1.1",
        "fixes_applied": True,
        "ai_enabled": service.ai_enabled,
        "tts_enabled": service.tts_enabled,
        "camera_available": service.camera is not None,
        "ultrasonic_enabled": service.ultrasonic_enabled,
        "current_distance_cm": service.last_distance,
        "running": service.running,
        "analysis_count": service.analysis_count,
        "last_analysis": service.last_ai_analysis,
        "uptime_seconds": time.time() - service.last_ai_analysis if service.last_ai_analysis > 0 else 0,
        "livekit_available": LIVEKIT_AVAILABLE,
        "gpio_available": GPIO_AVAILABLE,
        "http_session_active": service.http_session is not None
    }

if __name__ == "__main__":
    print("📷 Simple SentientSight Camera AI Service with TTS (FIXED VERSION)")
    print("🧠 Camera → Gemini AI → Navigation Alerts → TTS")
    print("🔊 TTS only speaks for hazards (danger/warning)")
    print("📏 Ultrasonic sensor integration for enhanced navigation")
    print("📊 Structured output: danger | warning | neutral")
    print("🚀 Starting server...")
    print(f"🌐 API will be available at: http://localhost:{API_PORT}")
    print("🔗 Use ngrok to expose: ngrok http 8000")
    print()
    print("✅ FIXES APPLIED:")
    print("   - HTTP session for Cartesia TTS plugin")
    print("   - FastAPI lifespan context manager")
    print("   - Proper resource cleanup")
    print()
    print("📝 Required environment variables:")
    print("   GEMINI_API_KEY=your_gemini_api_key")
    print("   LIVEKIT_API_KEY=your_livekit_api_key")
    print("   LIVEKIT_API_SECRET=your_livekit_api_secret")
    print("   LIVEKIT_URL=wss://your-livekit-server.com")
    print()
    print("💡 Make sure you have: pip install aiohttp")
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)
