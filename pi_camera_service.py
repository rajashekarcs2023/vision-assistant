#!/usr/bin/env python3
"""
Simple Pi Camera Service for Indoor Navigation
Provides camera images via HTTP API for the navigation agent
"""

import asyncio
import base64
import io
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global camera object
camera = None

def initialize_camera():
    """Initialize camera (Pi Camera or USB camera)"""
    global camera
    try:
        # Try Pi Camera first
        try:
            import picamera
            camera = picamera.PiCamera()
            camera.resolution = (640, 480)
            logger.info("Pi Camera initialized")
            return True
        except ImportError:
            logger.info("Pi Camera not available, trying USB camera")
        
        # Fallback to USB camera via OpenCV
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logger.info("USB Camera initialized")
            return True
        else:
            logger.error("No camera available")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return False

def capture_image_pi_camera():
    """Capture image using Pi Camera"""
    try:
        stream = io.BytesIO()
        camera.capture(stream, format='jpeg')
        stream.seek(0)
        image_data = stream.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error capturing Pi Camera image: {e}")
        return None

def capture_image_usb_camera():
    """Capture image using USB camera"""
    try:
        ret, frame = camera.read()
        if ret:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()
            return base64.b64encode(image_data).decode('utf-8')
        else:
            logger.error("Failed to capture frame from USB camera")
            return None
    except Exception as e:
        logger.error(f"Error capturing USB camera image: {e}")
        return None

@app.route('/capture', methods=['GET'])
def capture_image():
    """Capture and return current camera image as base64"""
    try:
        if camera is None:
            return jsonify({'error': 'Camera not initialized'}), 500
        
        # Determine camera type and capture accordingly
        if hasattr(camera, 'capture'):  # Pi Camera
            image_base64 = capture_image_pi_camera()
        else:  # USB Camera
            image_base64 = capture_image_usb_camera()
        
        if image_base64:
            return jsonify({
                'status': 'success',
                'image_data': image_base64,
                'timestamp': asyncio.get_event_loop().time()
            })
        else:
            return jsonify({'error': 'Failed to capture image'}), 500
            
    except Exception as e:
        logger.error(f"Error in capture endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def camera_status():
    """Get camera status"""
    return jsonify({
        'camera_available': camera is not None,
        'camera_type': 'pi_camera' if hasattr(camera, 'capture') else 'usb_camera'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'pi_camera_service'})

def cleanup_camera():
    """Cleanup camera resources"""
    global camera
    if camera:
        try:
            if hasattr(camera, 'close'):  # Pi Camera
                camera.close()
            else:  # USB Camera
                camera.release()
            logger.info("Camera cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up camera: {e}")

if __name__ == '__main__':
    try:
        logger.info("Starting Pi Camera Service for Indoor Navigation")
        
        if initialize_camera():
            logger.info("Camera service ready")
            # Run Flask app
            app.run(host='0.0.0.0', port=8000, debug=False)
        else:
            logger.error("Failed to initialize camera. Exiting.")
    
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        cleanup_camera()
        logger.info("Pi Camera Service shutdown complete")
