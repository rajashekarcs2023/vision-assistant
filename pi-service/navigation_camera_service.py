#!/usr/bin/env python3
"""
Dedicated Pi Camera Service for Indoor Navigation Demo
Lightweight script that only captures and uploads images to Convex for navigation
"""

import asyncio
import time
import base64
import io
import requests
import json
import logging
from typing import Optional
from datetime import datetime
import os
from pathlib import Path

# Camera imports (try Pi camera first, fallback to USB)
try:
    from picamera2 import Picamera2
    CAMERA_TYPE = "pi"
except ImportError:
    try:
        import cv2
        CAMERA_TYPE = "usb"
    except ImportError:
        CAMERA_TYPE = "none"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NavigationCameraService:
    def __init__(self, convex_url: str):
        """Initialize navigation camera service"""
        self.convex_url = convex_url.rstrip('/')
        self.camera = None
        self.camera_type = CAMERA_TYPE
        self.capture_interval = 2.0  # Capture every 2 seconds
        self.upload_queue = []
        self.is_running = False
        
        logger.info(f"Initialized with camera type: {self.camera_type}")
        
    def init_camera(self):
        """Initialize camera based on available hardware"""
        try:
            if self.camera_type == "pi":
                logger.info("Initializing Pi Camera...")
                self.camera = Picamera2()
                # Configure for faster capture
                config = self.camera.create_still_configuration(
                    main={"size": (640, 480)},  # Lower resolution for speed
                    lores={"size": (320, 240)},
                    display="lores"
                )
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)  # Let camera warm up
                logger.info("✅ Pi Camera initialized successfully")
                
            elif self.camera_type == "usb":
                logger.info("Initializing USB Camera...")
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 15)
                
                if not self.camera.isOpened():
                    raise Exception("Failed to open USB camera")
                logger.info("✅ USB Camera initialized successfully")
                
            else:
                raise Exception("No camera available")
                
        except Exception as e:
            logger.error(f"❌ Camera initialization failed: {e}")
            raise
    
    def capture_image(self) -> Optional[bytes]:
        """Capture image and return as JPEG bytes"""
        try:
            if self.camera_type == "pi":
                # Capture with Pi Camera
                stream = io.BytesIO()
                self.camera.capture_file(stream, format='jpeg')
                stream.seek(0)
                return stream.read()
                
            elif self.camera_type == "usb":
                # Capture with USB camera
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to capture frame from USB camera")
                    return None
                    
                # Convert to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return buffer.tobytes()
                
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return None
    
    async def upload_to_convex(self, image_data: bytes) -> Optional[str]:
        """Upload image to Convex and return image ID"""
        try:
            # Convert to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Convex upload payload
            upload_payload = {
                "image_data": image_b64,
                "timestamp": datetime.now().isoformat(),
                "source": "navigation_camera",
                "metadata": {
                    "purpose": "indoor_navigation",
                    "camera_type": self.camera_type,
                    "resolution": "640x480"
                }
            }
            
            # Upload to Convex
            upload_url = f"{self.convex_url}/upload_navigation_image"
            response = requests.post(
                upload_url,
                json=upload_payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                image_id = result.get('image_id')
                logger.info(f"📸 Image uploaded successfully: {image_id}")
                return image_id
            else:
                logger.error(f"Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading to Convex: {e}")
            return None
    
    async def capture_and_upload_loop(self):
        """Main loop: capture images and upload to Convex"""
        logger.info("🚀 Starting navigation camera service...")
        
        while self.is_running:
            try:
                # Capture image
                image_data = self.capture_image()
                
                if image_data:
                    logger.info(f"📷 Captured image ({len(image_data)} bytes)")
                    
                    # Upload to Convex (async)
                    image_id = await self.upload_to_convex(image_data)
                    
                    if image_id:
                        logger.info(f"✅ Navigation image ready: {image_id}")
                    else:
                        logger.warning("⚠️ Upload failed, will retry next cycle")
                else:
                    logger.warning("⚠️ Failed to capture image")
                
                # Wait before next capture
                await asyncio.sleep(self.capture_interval)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def start_service(self):
        """Start the navigation camera service"""
        try:
            self.init_camera()
            self.is_running = True
            
            logger.info("🎥 Navigation camera service started")
            logger.info(f"📡 Uploading to: {self.convex_url}")
            logger.info(f"⏱️ Capture interval: {self.capture_interval}s")
            
            await self.capture_and_upload_loop()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Service stopped by user")
        except Exception as e:
            logger.error(f"Service error: {e}")
        finally:
            await self.stop_service()
    
    async def stop_service(self):
        """Stop the service and cleanup"""
        self.is_running = False
        
        if self.camera:
            try:
                if self.camera_type == "pi":
                    self.camera.stop()
                elif self.camera_type == "usb":
                    self.camera.release()
                logger.info("📹 Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
    
    def test_capture(self):
        """Test camera capture (for debugging)"""
        logger.info("🧪 Testing camera capture...")
        
        try:
            self.init_camera()
            image_data = self.capture_image()
            
            if image_data:
                logger.info(f"✅ Test capture successful ({len(image_data)} bytes)")
                
                # Save test image
                test_file = f"test_capture_{int(time.time())}.jpg"
                with open(test_file, 'wb') as f:
                    f.write(image_data)
                logger.info(f"💾 Test image saved: {test_file}")
                
                return True
            else:
                logger.error("❌ Test capture failed")
                return False
                
        except Exception as e:
            logger.error(f"Test error: {e}")
            return False
        finally:
            if self.camera:
                if self.camera_type == "pi":
                    self.camera.stop()
                elif self.camera_type == "usb":
                    self.camera.release()


async def main():
    """Main function"""
    # Hardcoded Convex URL for navigation service
    convex_url = "https://neat-gopher-434.convex.cloud"
    
    logger.info(f"🎯 Using Convex URL: {convex_url}")
    
    # Create and start service
    service = NavigationCameraService(convex_url)
    
    # Test mode or service mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logger.info("🧪 Running in test mode...")
        success = service.test_capture()
        if success:
            logger.info("✅ Camera test passed - ready for navigation service")
        else:
            logger.error("❌ Camera test failed")
    else:
        logger.info("🚀 Starting navigation camera service...")
        await service.start_service()


if __name__ == "__main__":
    asyncio.run(main())
