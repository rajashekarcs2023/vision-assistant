// types.ts - Corrected to match Pi sensor service

export enum DetectionStatus {
  IDLE = "IDLE",
  INITIALIZING = "INITIALIZING", 
  DETECTING = "DETECTING",
  PROCESSING = "PROCESSING",
  HAZARD_DETECTED = "HAZARD_DETECTED", // Renamed from EVENT_DETECTED
  PATH_CLEAR = "PATH_CLEAR",         // Renamed from NO_EVENT
  ERROR = "ERROR",
  API_KEY_MISSING = "API_KEY_MISSING",
  CAMERA_ERROR = "CAMERA_ERROR",
  // Pi integration statuses:
  WEBSOCKET_DISCONNECTED = "WEBSOCKET_DISCONNECTED",
  WEBSOCKET_CONNECTING = "WEBSOCKET_CONNECTING"
}

export interface WebcamFeedRef {
  captureFrame: () => string | null;
}

// Defines the expected structure of the JSON response from Gemini API
export interface HazardAnalysisResponse {
  hazard_type: string; // e.g., "WALL_AHEAD", "STAIRS_DOWN_AHEAD", "PATH_CLEAR"
  message: string; // User-friendly message for TTS and display
}

// CORRECTED: AccelerometerData to match Pi service output
export interface AccelerometerData {
  x: number;
  y: number;
  z: number;
  magnitude: number;           // FIXED: Pi sends "magnitude" not "total_acceleration"
  motion_magnitude: number;    // ADDED: Pi sends this
  pitch: number;
  roll: number;
  is_moving: boolean;
  motion_confidence: number;   // ADDED: Pi sends this instead of "motion_status"
}

// CORRECTED: CameraData to match Pi service output
export interface CameraData {
  image_data?: string;         // FIXED: Pi sends "image_data" not "data"
  format: string;              // e.g., "jpeg"
  width: number;               // ADDED: Pi sends width/height not "resolution"
  height: number;              // ADDED: Pi sends width/height not "resolution"
  timestamp: number;
  size_bytes: number;          // ADDED: Pi sends this
  status?: string;             // ADDED: For status messages when no image
  last_capture?: number;       // ADDED: Pi sends this
}

// CORRECTED: UltrasonicData to match Pi service output
export interface UltrasonicData {
  distance_cm: number;
  status: string;              // ADDED: Pi sends "close" or "clear"
  danger_level: string;        // ADDED: Pi sends "critical" or "safe"
}

// CORRECTED: SensorPackage to match Pi service output
export interface SensorPackage {
  timestamp: number;
  datetime: string;            // ADDED: Pi sends formatted time string
  accelerometer: AccelerometerData | null;
  ultrasonic: UltrasonicData | null;
  camera: CameraData | null;
  system: {                    // ADDED: Pi sends system info
    motion_state: string;      // "moving" or "stationary"
    sensors_active: string[];  // List of active sensors
    uptime_seconds: number;    // System uptime
    connected_clients: number; // Number of connected clients
  };
}

// Enhanced analysis response with sensor context
export interface EnhancedHazardAnalysisResponse extends HazardAnalysisResponse {
  sensor_context?: {
    motion_detected: boolean;
    head_orientation: string;
    ultrasonic_distance?: number;
  };
}