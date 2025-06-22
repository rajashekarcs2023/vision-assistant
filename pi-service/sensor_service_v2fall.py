#!/usr/bin/env python3
"""
Simple SentientSight Camera AI Service with TTS and Convex Integration
Camera → AI inference → Convex storage → TTS alerts

NEW FEATURES:
- Convex integration for storing alerts and images
- Non-blocking database operations
- Frame upload to Convex storage
"""

import asyncio
import time
import io
import os
import json
import tempfile
import wave
import math
import subprocess
import base64
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

# HTTP client for TTS and Convex
import aiohttp

# VAPI client for emergency calls
try:
    from vapi import Vapi
    VAPI_AVAILABLE = True
    print("✅ VAPI client imported successfully")
except ImportError:
    VAPI_AVAILABLE = False
    print("⚠️ VAPI not available - install with: pip install vapi")

# ADXL345 accelerometer imports for fall detection
try:
    from smbus2 import SMBus
    ACCELEROMETER_AVAILABLE = True
    print("✅ SMBus (accelerometer) imported successfully")
except ImportError:
    ACCELEROMETER_AVAILABLE = False
    print("⚠️ SMBus not available - install with: pip install smbus2")

# Convex client
try:
    from convex import ConvexClient

    CONVEX_AVAILABLE = True
    print("✅ Convex client imported successfully")
except ImportError:
    CONVEX_AVAILABLE = False
    print("⚠️ Convex not available - install with: pip install convex")

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

# Convex Configuration
CONVEX_URL = os.getenv('CONVEX_URL')

# VAPI Configuration
VAPI_TOKEN = os.getenv('VAPI_TOKEN')
VAPI_ASSISTANT_ID = os.getenv('VAPI_ASSISTANT_ID', 'fd83ba03-904b-4554-8288-38644145b6fd')
VAPI_PHONE_NUMBER_ID = os.getenv('VAPI_PHONE_NUMBER_ID')
EMERGENCY_PHONE_NUMBER = os.getenv('EMERGENCY_PHONE_NUMBER', '+16695774085')

# ADXL345 Accelerometer Configuration
I2C_BUS = 1          # Default I2C bus on Raspberry Pi
ADXL345_ADDR = 0x53  # ADXL345 default I2C address
REG_POWER = 0x2D
REG_FORMAT = 0x31
REG_DATA = 0x32      # First of 6 data registers (X0, X1, Y0, Y1, Z0, Z1)
SCALE_G = 0.0039     # 4 mg/LSB in full-resolution ±2 g mode

# Fall Detection Configuration
FALL_G_FORCE_THRESHOLD = 1.5  # G-force threshold for fall detection
FALL_DEBOUNCE_SECONDS = 30    # Minimum time between fall detections

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
        self.convex_client = None  # Convex client
        self.vapi_client = None  # VAPI client for emergency calls
        self.accelerometer_bus = None  # I2C bus for accelerometer
        self.room = None
        self.running = False

        # State
        self.ai_enabled = False
        self.tts_enabled = False
        self.convex_enabled = False
        self.vapi_enabled = False
        self.fall_detection_enabled = False
        self.last_ai_analysis = 0
        self.current_ai_result = None
        self.analysis_count = 0
        self.last_spoken_message = None
        self.last_tts_time = 0

        # Ultrasonic sensor state
        self.ultrasonic_enabled = False
        self.last_distance = None
        self.distance_readings = []  # For averaging

        # Fall detection state
        self.last_fall_time = 0
        self.current_accelerometer_data = None
        self.g_force_readings = []  # For averaging and trend analysis

        # TTS Settings
        self.tts_cooldown = 2.0  # Minimum 2 seconds between TTS messages
        self.repeat_threshold = 5.0  # Repeat same message after 5 seconds

        print("📷 Simple Camera AI Service with TTS and Convex Starting...")

        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(TRIG_PIN, GPIO.OUT)
            GPIO.setup(ECHO_PIN, GPIO.IN)
            self.ultrasonic_enabled = True
            print("📏 Ultrasonic sensor initialized")

    def setup_convex_client(self):
        """Initialize Convex client"""
        if not CONVEX_AVAILABLE:
            print("📦 Convex: ❌ Client not available")
            self.convex_enabled = False
            return

        if not CONVEX_URL:
            print("❌ CONVEX_URL environment variable not set!")
            print("💡 Run: export CONVEX_URL='https://your-deployment.convex.cloud'")
            self.convex_enabled = False
            return

        try:
            self.convex_client = ConvexClient(CONVEX_URL)
            print("📦 Convex: ✅ Connected")
            self.convex_enabled = True
        except Exception as e:
            print(f"📦 Convex: ❌ Failed - {e}")
            self.convex_enabled = False

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

    def setup_vapi_client(self):
        """Initialize VAPI client for emergency calls"""
        if not VAPI_AVAILABLE or not VAPI_TOKEN:
            print("📞 VAPI: ❌ Not available or token missing")
            return

        try:
            self.vapi_client = Vapi(token=VAPI_TOKEN)
            print("📞 VAPI: ✅ Client initialized")
            self.vapi_enabled = True
        except Exception as e:
            print(f"📞 VAPI: ❌ Failed - {e}")
            self.vapi_enabled = False

    def setup_accelerometer(self):
        """Initialize ADXL345 accelerometer for fall detection"""
        if not ACCELEROMETER_AVAILABLE:
            print("🔄 Accelerometer: ❌ SMBus not available")
            return

        try:
            self.accelerometer_bus = SMBus(I2C_BUS)
            
            # Initialize ADXL345
            self.accelerometer_bus.write_byte_data(ADXL345_ADDR, REG_POWER, 0x08)   # Measure mode
            self.accelerometer_bus.write_byte_data(ADXL345_ADDR, REG_FORMAT, 0x08)  # Full-res, ±2 g
            
            print("🔄 Accelerometer: ✅ ADXL345 initialized")
            self.fall_detection_enabled = True
        except Exception as e:
            print(f"🔄 Accelerometer: ❌ Failed - {e}")
            self.fall_detection_enabled = False

    async def upload_image_to_convex(self, image_data: bytes) -> str:
        """Upload image to Convex storage and return storage ID"""
        if not self.convex_enabled or not self.convex_client:
            print("📦 Convex upload skipped - not available")
            return None

        try:
            # Get upload URL from Convex
            upload_url = self.convex_client.mutation("files:generateUploadUrl")

            if not upload_url:
                print("❌ Failed to get Convex upload URL")
                return None

            # Upload image data
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        upload_url,
                        data=image_data,
                        headers={'Content-Type': 'image/jpeg'}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        storage_id = result.get('storageId')
                        print(f"📦 Image uploaded to Convex: {storage_id}")
                        return storage_id
                    else:
                        print(f"❌ Convex upload failed: {response.status}")
                        return None

        except Exception as e:
            print(f"❌ Convex upload error: {e}")
            return None

    async def store_alert_in_convex(self, alert_type: str, message: str, image_id: str = None):
        """Store alert in Convex database (non-blocking)"""
        if not self.convex_enabled or not self.convex_client:
            print(f"📦 Convex storage skipped - would store: {alert_type}: {message}")
            return

        try:
            # Fire-and-forget: don't await the result
            asyncio.create_task(self._store_alert_async(alert_type, message, image_id))
            print(f"📦 Alert queued for Convex storage: {alert_type}")

        except Exception as e:
            print(f"❌ Convex alert storage error: {e}")

    async def _store_alert_async(self, alert_type: str, message: str, image_id: str = None):
        """Internal async method to store alert in Convex"""
        try:
            # Prepare alert data
            alert_data = {
                "type": alert_type,
                "message": message
            }

            # Add image_id if provided
            if image_id:
                alert_data["image_id"] = image_id

            # Store in Convex (this is the non-blocking part)
            result = self.convex_client.mutation("alerts:add_alerts", alert_data)
            print(f"📦 Alert stored in Convex: {result.get('id', 'unknown')} - {alert_type}: {message}")

        except Exception as e:
            print(f"❌ Async Convex storage error: {e}")

    def read_accelerometer(self) -> dict:
        """Read X, Y, Z acceleration data from ADXL345"""
        if not self.fall_detection_enabled or not self.accelerometer_bus:
            return None

        try:
            # Read 6 bytes starting from REG_DATA (X0, X1, Y0, Y1, Z0, Z1)
            data = self.accelerometer_bus.read_i2c_block_data(ADXL345_ADDR, REG_DATA, 6)
            
            # Convert little-endian 16-bit values to acceleration
            axes = []
            for i in (0, 2, 4):  # X, Y, Z
                val = data[i] | (data[i+1] << 8)
                if val & 0x8000:  # Two's complement for negative values
                    val = -((0xFFFF - val) + 1)
                axes.append(val * SCALE_G)  # Convert to g-force
            
            x, y, z = axes
            
            # Calculate total magnitude
            magnitude = math.sqrt(x**2 + y**2 + z**2)
            
            accel_data = {
                'x': round(x, 3),
                'y': round(y, 3), 
                'z': round(z, 3),
                'magnitude': round(magnitude, 3),
                'timestamp': time.time()
            }
            
            self.current_accelerometer_data = accel_data
            
            # Keep last 10 readings for trend analysis
            self.g_force_readings.append(magnitude)
            if len(self.g_force_readings) > 10:
                self.g_force_readings.pop(0)
            
            return accel_data
            
        except Exception as e:
            print(f"🔄 Accelerometer read error: {e}")
            return None

    def detect_fall(self) -> bool:
        """Detect if a fall occurred based on accelerometer data"""
        if not self.fall_detection_enabled:
            return False
            
        accel_data = self.read_accelerometer()
        if not accel_data:
            return False
            
        current_time = time.time()
        magnitude = accel_data['magnitude']
        
        # Check if magnitude exceeds fall threshold
        if magnitude > FALL_G_FORCE_THRESHOLD:
            # Check debounce period - don't trigger multiple falls too quickly
            if (current_time - self.last_fall_time) > FALL_DEBOUNCE_SECONDS:
                print(f"🚨 FALL DETECTED! G-force: {magnitude:.2f}g (threshold: {FALL_G_FORCE_THRESHOLD}g)")
                self.last_fall_time = current_time
                return True
            else:
                print(f"🔄 High G-force detected ({magnitude:.2f}g) but within debounce period")
                
        return False

    async def make_emergency_call(self, reason: str = "Fall detected"):
        """Make emergency call using VAPI"""
        if not self.vapi_enabled or not self.vapi_client:
            print(f"📞 Emergency call skipped - VAPI not available. Reason: {reason}")
            return False

        try:
            print(f"📞 Initiating emergency call: {reason}")
            
            # Prepare call data
            call_data = {
                "assistant_id": VAPI_ASSISTANT_ID,
                "phone_number_id": VAPI_PHONE_NUMBER_ID,
                "customer": {
                    "number": EMERGENCY_PHONE_NUMBER
                }
            }
            
            # Make the call
            call = self.vapi_client.calls.create(**call_data)
            
            print(f"📞 Emergency call initiated successfully: {call.id}")
            print(f"📞 Calling {EMERGENCY_PHONE_NUMBER} for: {reason}")
            
            # Store emergency call in Convex if available
            if self.convex_enabled:
                await self.store_alert_in_convex(
                    alert_type="danger",
                    message=f"Emergency call initiated: {reason}",
                    image_id=None
                )
            
            return True
            
        except Exception as e:
            print(f"❌ Emergency call failed: {e}")
            return False

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
        """AI analysis with TTS-first priority, then fire-and-forget Convex operations"""
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
            
            prompt = f"""You are an AI assistant guiding a blind person as they walk. Your goal is to provide **short, clear, human-style verbal directions** based on the first-person camera view and proximity sensor data.

Start by analyzing the **camera view** for any iwhat you see ahead, immediate onstacles,  changes in terrain, or nearby people. Then use the **distance sensor** ONLY to enhance or confirm risks.

{distance_info}

🧠 **ANALYSIS PRIORITY:**
1. Focus FIRST on visual input — describe any immediate obstacles, stairs, drop-offs, walls, people, etc. Dont use vague terms like 'obstacle ahead'. 
2. Then mention the approximate distance to what you see:
   - There's a visible hazard (give the distance)
   - Distance is <30cm but the camera sees nothing (potential invisible obstacle like glass or pole)

🧭 **INSTRUCTION STYLE:**
- Be **natural, friendly, and directive** — like a sighted guide helping someone walk.
- Avoid words like “identify,” “investigate,” or “analyze”
- Say things like: “Watch out,” “Slow down,” “Turn slightly left,” “Clear ahead,” etc.
- Keep responses short and under 20 words for text-to-speech
- put it under the danger category only if the obstacle is directly ahead, otherwise put it under the warning category like - I see a person walking directly towards you, or a person is standing at a close distance ahead of you.

📏 **DISTANCE RULES:**
- Ignore any distance >50cm unless it confirms a clear visual hazard
- If distance <30cm and camera sees nothing, mention a **possible** close obstacle
- Never mention raw distance unless it makes the message clearer or safer

🔁 **JSON RESPONSE FORMAT:**
Respond only in this exact format:

{{
  "type": "danger" | "warning" | "neutral",
  "message": "Concise voice message under 20 words"
}}

📌 **EXAMPLES:**
- Wall ahead in view:       {{"type": "danger", "message": "Wall ahead, stop now"}}
- chair ahead in view:       {{"type": "danger", "message": "chair ahead, stop now and turn left"}}
- Clear visually:           {{"type": "neutral", "message": "Path clear ahead"}}
- Stairs detected:          {{"type": "warning", "message": "Stairs going down, walk carefully"}}
- Nothing in view, but <15cm:  {{"type": "warning", "message": "Possible object close ahead, move slowly"}}

🔒 **DO NOT SAY**: “Analyzing,” “Object detected,” “Identify,” “Investigate,” “8 centimeters” unless it’s urgent
🎯 **DO SAY**: What a sighted friend would tell you while walking


Focus on what the CAMERA shows first,describe what you see , and use distance sensor only to enhance hazard details."""
            image_part = types.Part.from_bytes(data=image_data, mime_type='image/jpeg')
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
            if json_str.startswith('\`\`\`'):
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

            # Log result BEFORE Convex operations
            distance_log = f" (📏 {current_distance}cm)" if current_distance else " (📏 no distance)"
            type_emoji = {"danger": "🚨", "warning": "⚠️", "neutral": "✅"}
            emoji = type_emoji.get(ai_result["type"], "ℹ️")
            print(
                f"🧠 AI Result #{self.analysis_count}: {emoji} {ai_result['type'].upper()} - {ai_result['message']}{distance_log}")

            # FIRE-AND-FORGET: Upload image and store alert in background (AFTER TTS will happen)
            if ai_result["type"] in ["danger", "warning"] and self.convex_enabled:
                # Start background task for Convex operations - don't await!
                asyncio.create_task(self._handle_convex_storage_async(image_data, ai_result))
                print(f"📦 Convex storage queued for background processing")
            elif self.convex_enabled and image_data != b"no_camera_data":
                # Even for neutral results, still upload image for analysis history
                asyncio.create_task(self._upload_image_only_async(image_data))
                print(f"📦 Image upload queued for background processing")

            return ai_result

        except Exception as e:
            print(f"❌ AI analysis error: {e}")
            return {
                "type": "neutral",
                "message": "Analysis failed",
                "error": str(e),
                "sensor_distance_cm": self.last_distance
            }

    async def _handle_convex_storage_async(self, image_data: bytes, ai_result: dict):
        """Background task: Upload image and store alert (fire-and-forget)"""
        try:
            print("📦 Background: Starting Convex storage operations...")

            # Upload image first
            image_id = await self.upload_image_to_convex(image_data)

            if image_id:
                ai_result['image_id'] = image_id
                print(f"📦 Background: Image uploaded - {image_id}")

            # Store alert with image reference
            await self._store_alert_async(
                alert_type=ai_result["type"],
                message=ai_result["message"],
                image_id=image_id
            )

            print(f"📦 Background: Convex storage completed for {ai_result['type']} alert")

        except Exception as e:
            print(f"❌ Background Convex storage error: {e}")

    async def _upload_image_only_async(self, image_data: bytes):
        """Background task: Upload image only (for neutral results)"""
        try:
            print("📦 Background: Uploading image for analysis history...")
            image_id = await self.upload_image_to_convex(image_data)
            if image_id:
                print(f"📦 Background: Image uploaded - {image_id}")
        except Exception as e:
            print(f"❌ Background image upload error: {e}")

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
        print("🔄 Starting continuous AI analysis with Convex integration...")

        while self.running:
            try:
                current_time = time.time()

                # Check if it's time for analysis
                if (current_time - self.last_ai_analysis) >= AI_ANALYSIS_INTERVAL:

                    # Capture image
                    print(f"📸 Capturing image #{self.analysis_count}...")
                    image_data = self.capture_image()

                    if image_data:
                        # Analyze with AI (now returns immediately, Convex ops happen in background)
                        print(f"🧠 Analyzing with AI...")
                        ai_result = await self.analyze_with_ai(image_data)

                        # Store result
                        self.current_ai_result = ai_result
                        self.last_ai_analysis = current_time
                        self.analysis_count += 1

                        # PRIORITY 1: Trigger TTS for hazards IMMEDIATELY
                        if self.should_speak_result(ai_result):
                            message = ai_result.get('message', 'Hazard detected')
                            urgency = ai_result.get('type', 'medium')
                            print(f"🔊 PRIORITY: Speaking hazard alert immediately")
                            await self.speak_message(message, urgency)

                        # Log result (Convex operations already queued in background)
                        urgency_icon = {"danger": "🚨", "warning": "⚠️", "neutral": "✅"}.get(
                            ai_result.get('type', 'neutral'), "⚪")
                        print(
                            f"{urgency_icon} Analysis #{self.analysis_count} complete: {ai_result.get('type').upper()} - '{ai_result.get('message')}'")

                # CRITICAL: Check for falls continuously (every loop iteration)
                if self.fall_detection_enabled:
                    if self.detect_fall():
                        print("🚨 EMERGENCY: Fall detected - initiating emergency call")
                        
                        # Speak immediate fall alert
                        await self.speak_message("Emergency: Fall detected, calling for help", "danger")
                        
                        # Make emergency call
                        call_success = await self.make_emergency_call("Fall detected by accelerometer")
                        
                        if call_success:
                            print("✅ Emergency call initiated successfully")
                        else:
                            print("❌ Emergency call failed")
                            # Fallback: store in Convex if available
                            if self.convex_enabled:
                                await self.store_alert_in_convex(
                                    alert_type="danger",
                                    message="FALL DETECTED - Emergency call failed",
                                    image_id=None
                                )

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
    print("🚀 Starting Simple Camera AI Service with TTS and Convex...")
    service.setup_convex_client()
    service.setup_ai_client()
    await service.setup_tts()
    service.setup_camera()
    service.setup_vapi_client()
    service.setup_accelerometer()
    service.running = True

    # Start background analysis
    asyncio.create_task(service.run_continuous_analysis())
    print("✅ Service started!")

    yield

    # Shutdown
    await service.cleanup()


# Initialize FastAPI with lifespan
app = FastAPI(
    title="Simple SentientSight Camera AI with TTS and Convex",
    description="Camera AI with text-to-speech and Convex database integration",
    version="2.0.0",
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
        "service": "Simple SentientSight Camera AI with TTS and Convex",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Camera image capture",
            "Gemini AI analysis",
            "Convex image storage",
            "Convex alert database",
            "Non-blocking database operations",
            "LiveKit TTS integration",
            "Ultrasonic sensor support"
        ],
        "ai_enabled": service.ai_enabled,
        "tts_enabled": service.tts_enabled,
        "convex_enabled": service.convex_enabled,
        "camera_available": service.camera is not None,
        "ultrasonic_enabled": service.ultrasonic_enabled,
        "current_distance_cm": service.last_distance,
        "analysis_count": service.analysis_count,
        "endpoints": {
            "current_analysis": "/analysis",
            "latest_alert": "/alert",
            "test_analysis": "/test",
            "test_tts": "/test-tts",
            "test_convex": "/test-convex",
            "tts_settings": "/tts",
            "distance": "/distance",
            "health": "/health",
            "accelerometer": "/accelerometer",
            "test_fall": "/test-fall"
        }
    }


@app.get("/analysis")
async def get_current_analysis():
    """Get the latest AI analysis"""
    if not service.current_ai_result:
        return {
            "status": "no_analysis_yet",
            "message": "No analysis performed yet, wait a few seconds",
            "next_analysis_in": AI_ANALYSIS_INTERVAL - (
                        time.time() - service.last_ai_analysis) if service.last_ai_analysis > 0 else AI_ANALYSIS_INTERVAL
        }

    return {
        "status": "analysis_available",
        "latest_analysis": service.current_ai_result,
        "analysis_count": service.analysis_count,
        "time_since_analysis": time.time() - service.last_ai_analysis,
        "next_analysis_in": AI_ANALYSIS_INTERVAL - (time.time() - service.last_ai_analysis),
        "convex_enabled": service.convex_enabled
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
            "tts_enabled": service.tts_enabled,
            "stored_in_convex": service.convex_enabled,
            "image_id": ai_result.get('image_id')
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
                "convex_storage": service.convex_enabled,
                "image_uploaded": ai_result.get('image_id') is not None,
                "message": "Manual analysis completed successfully"
            }
        else:
            return {"error": "Failed to capture image"}

    except Exception as e:
        return {"error": f"Test failed: {e}"}


@app.get("/accelerometer")
async def get_accelerometer_data():
    """Get current accelerometer data and fall detection status"""
    if not service.fall_detection_enabled:
        return {
            "error": "Fall detection not available",
            "fall_detection_enabled": False,
            "accelerometer_available": ACCELEROMETER_AVAILABLE,
            "vapi_available": VAPI_AVAILABLE
        }

    # Get fresh accelerometer reading
    accel_data = service.read_accelerometer()

    return {
        "fall_detection_enabled": service.fall_detection_enabled,
        "vapi_enabled": service.vapi_enabled,
        "current_data": accel_data,
        "last_fall_time": service.last_fall_time,
        "time_since_last_fall": time.time() - service.last_fall_time if service.last_fall_time > 0 else None,
        "recent_g_force_readings": service.g_force_readings[-5:] if service.g_force_readings else [],
        "fall_threshold": FALL_G_FORCE_THRESHOLD,
        "debounce_seconds": FALL_DEBOUNCE_SECONDS,
        "config": {
            "i2c_bus": I2C_BUS,
            "adxl345_addr": hex(ADXL345_ADDR),
            "scale_g": SCALE_G
        }
    }


@app.post("/test-fall")
async def test_fall_detection():
    """Test fall detection and emergency calling system"""
    if not service.fall_detection_enabled:
        return {"error": "Fall detection not available"}

    try:
        print("🧪 Testing fall detection system...")
        
        # Test accelerometer reading
        accel_data = service.read_accelerometer()
        if not accel_data:
            return {"error": "Failed to read accelerometer data"}
        
        # Simulate fall detection (bypass threshold check)
        print("🧪 Simulating fall detection...")
        
        # Test emergency call
        call_success = await service.make_emergency_call("Test fall detection system")
        
        return {
            "test_complete": True,
            "accelerometer_data": accel_data,
            "emergency_call_initiated": call_success,
            "vapi_enabled": service.vapi_enabled,
            "emergency_number": EMERGENCY_PHONE_NUMBER,
            "assistant_id": VAPI_ASSISTANT_ID,
            "message": "Fall detection test completed"
        }
        
    except Exception as e:
        return {"error": f"Fall detection test failed: {e}"}


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


@app.post("/test-convex")
async def test_convex():
    """Test Convex integration"""
    if not service.convex_enabled:
        return {
            "error": "Convex not enabled",
            "check": "CONVEX_URL environment variable and convex package"
        }

    try:
        # Test storing a sample alert
        await service.store_alert_in_convex("warning", "Test alert from API", None)

        return {
            "convex_test_complete": True,
            "message": "Test alert stored in Convex",
            "convex_enabled": service.convex_enabled,
            "convex_url": CONVEX_URL
        }
    except Exception as e:
        return {"error": f"Convex test failed: {e}"}


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
        "version": "2.0.0",
        "integrations": {
            "ai_enabled": service.ai_enabled,
            "tts_enabled": service.tts_enabled,
            "convex_enabled": service.convex_enabled,
            "camera_available": service.camera is not None,
            "ultrasonic_enabled": service.ultrasonic_enabled
        },
        "current_distance_cm": service.last_distance,
        "running": service.running,
        "analysis_count": service.analysis_count,
        "last_analysis": service.last_ai_analysis,
        "uptime_seconds": time.time() - service.last_ai_analysis if service.last_ai_analysis > 0 else 0,
        "dependencies": {
            "livekit_available": LIVEKIT_AVAILABLE,
            "gpio_available": GPIO_AVAILABLE,
            "convex_available": CONVEX_AVAILABLE,
            "http_session_active": service.http_session is not None
        }
    }


if __name__ == "__main__":
    print("📷 Simple SentientSight Camera AI Service with TTS and Convex")
    print("🧠 Camera → Gemini AI → Convex Storage → TTS Alerts")
    print("🔊 TTS only speaks for hazards (danger/warning)")
    print("📦 Convex stores images and alerts automatically")
    print("📏 Ultrasonic sensor integration for enhanced navigation")
    print("🚀 Starting server...")
    print(f"🌐 API will be available at: http://localhost:{API_PORT}")
    print("🔗 Use ngrok to expose: ngrok http 8000")
    print()
    print("✅ NEW FEATURES:")
    print("   - Convex image upload and storage")
    print("   - Non-blocking alert database operations")
    print("   - Automatic hazard logging")
    print()
    print("📝 Required environment variables:")
    print("   GEMINI_API_KEY=your_gemini_api_key")
    print("   LIVEKIT_API_KEY=your_livekit_api_key")
    print("   LIVEKIT_API_SECRET=your_livekit_api_secret")
    print("   LIVEKIT_URL=wss://your-livekit-server.com")
    print("   CONVEX_URL=https://your-deployment.convex.cloud")
    print()
    print("💡 Make sure you have: pip install convex aiohttp")

    uvicorn.run(app, host=API_HOST, port=API_PORT)
