// KEEP all your existing constants exactly as they are:

export const WEBSOCKET_URL = 'ws://192.168.137.8:8765';
export const GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-04-17";
export const FRAME_CAPTURE_INTERVAL_MS = 2000; // Slightly reduced for faster feedback

// ADD these new motion-aware intervals:
export const FRAME_CAPTURE_INTERVAL_MOVING = 1500;    // Faster when walking
export const FRAME_CAPTURE_INTERVAL_STATIONARY = 5000; // Slower when still
export const MOTION_THRESHOLD = 0.1; // g-force threshold for movement

// ADD WebSocket Configuration:

// For Pi network access, change to: "ws://192.168.1.XXX:8080" where XXX is your Pi's IP

// KEEP all your existing hazard type constants:
export const HAZARD_TYPE_WALL = "WALL_AHEAD";
export const HAZARD_TYPE_OBSTACLE = "OBSTACLE_AHEAD";
export const HAZARD_TYPE_STAIRS_DOWN = "STAIRS_DOWN_AHEAD";
export const HAZARD_TYPE_STAIRS_UP = "STAIRS_UP_AHEAD";
export const HAZARD_TYPE_TRIP_LOW = "TRIP_HAZARD_LOW";
export const HAZARD_TYPE_HEAD_HIGH = "HEAD_HAZARD_HIGH";
export const HAZARD_TYPE_OPEN_DOORWAY = "OPEN_DOORWAY_AHEAD";
export const HAZARD_TYPE_CLOSED_DOOR = "CLOSED_DOOR_AHEAD";
export const HAZARD_TYPE_PATH_CLEAR = "PATH_CLEAR";

// ADD this new hazard type:
export const HAZARD_TYPE_IMMEDIATE_OBSTACLE = "IMMEDIATE_OBSTACLE";

// ADD Navigation Constants (for future Google Maps integration):
export const NAVIGATION_TURN_LEFT = "TURN_LEFT";
export const NAVIGATION_TURN_RIGHT = "TURN_RIGHT";
export const NAVIGATION_CONTINUE = "CONTINUE_STRAIGHT";
export const NAVIGATION_ARRIVED = "DESTINATION_REACHED";

// REPLACE your existing GEMINI_PROMPT_HAZARD_DETECTION with this enhanced version:
export const GEMINI_PROMPT_HAZARD_DETECTION = `You are an AI assistant for a visual aid application helping blind or visually impaired users navigate. Analyze the image provided, which represents a first-person view from a camera worn by the user. Your primary goal is to identify immediate potential hazards or important navigational cues directly in the user's path or immediate vicinity.

You will also receive real-time sensor data including:
- User motion status (walking vs stationary)
- Head orientation (pitch/roll angles)
- Ultrasonic distance measurements
- Movement intensity

Consider things like walls, obstacles, stairs (up or down), trip hazards, head-level obstructions, doorways, and general path clarity.

Respond ONLY with a JSON object. Do NOT use markdown like \`\`\`json. The JSON object must have the following structure:
{
  "hazard_type": "TYPE_STRING",
  "message": "A concise descriptive message for Text-To-Speech, explaining the hazard or path status. Be direct and clear."
}

Possible TYPE_STRING values and corresponding message guidance:
- "${HAZARD_TYPE_WALL}": "Wall ahead." or "Approaching a wall."
- "${HAZARD_TYPE_OBSTACLE}": "Obstacle ahead." or "Caution, object in your path." (mention if it's left/right/center if discernible)
- "${HAZARD_TYPE_STAIRS_DOWN}": "Stairs going down ahead."
- "${HAZARD_TYPE_STAIRS_UP}": "Stairs going up ahead."
- "${HAZARD_TYPE_TRIP_LOW}": "Trip hazard on the ground." or "Low obstacle, watch your step."
- "${HAZARD_TYPE_HEAD_HIGH}": "Head-level hazard." or "Caution, low overhang."
- "${HAZARD_TYPE_OPEN_DOORWAY}": "Open doorway ahead."
- "${HAZARD_TYPE_CLOSED_DOOR}": "Closed door ahead."
- "${HAZARD_TYPE_PATH_CLEAR}": "Path appears clear." or "Continue forward."
- "${HAZARD_TYPE_IMMEDIATE_OBSTACLE}": "Immediate obstacle detected." (use when ultrasonic sensor detects very close objects)

IMPORTANT SENSOR INTEGRATION RULES:
1. If ultrasonic sensor shows object <50cm but camera doesn't clearly show obstacle, prioritize ultrasonic data
2. If user is walking (motion detected), add urgency to hazard warnings
3. If user is looking up (positive pitch), prioritize overhead hazards  
4. If user is looking down (negative pitch), prioritize ground-level hazards
5. If user is stationary, provide more detailed spatial guidance
6. Always incorporate distance information from ultrasonic when relevant

If multiple hazards are present, prioritize the most immediate or dangerous one. If the image is unclear or too dark to make a reliable assessment, use:
{
  "hazard_type": "UNCLEAR_IMAGE",
  "message": "View is unclear, cannot reliably detect hazards."
}

Focus on what is directly in front or very near. Be concise but incorporate sensor context when relevant. Your response must be ONLY the JSON object.`;

// KEEP your existing default messages:
export const MESSAGE_PATH_CLEAR_DEFAULT = "Path appears clear.";
export const MESSAGE_UNCLEAR_IMAGE = "View is unclear, cannot reliably detect hazards.";
export const MESSAGE_ANALYSIS_ERROR = "Could not analyze surroundings.";

// ADD these new messages for WebSocket:
export const MESSAGE_WEBSOCKET_CONNECTING = "Connecting to Pi sensors...";
export const MESSAGE_WEBSOCKET_DISCONNECTED = "Disconnected from Pi sensors.";

// ADD Audio Priority Levels (for future enhancement):
export const AUDIO_PRIORITY = {
  IMMEDIATE_DANGER: 1,    // Stop everything
  HAZARD_WARNING: 2,      // Override navigation
  NAVIGATION_CRITICAL: 3, // Turn coming up
  NAVIGATION_INFO: 4,     // General guidance
  PATH_STATUS: 5          // Path clear, etc.
} as const;

// ADD Sensor Thresholds:
export const SENSOR_THRESHOLDS = {
  ULTRASONIC_IMMEDIATE: 30,   // cm - immediate warning
  ULTRASONIC_CLOSE: 100,      // cm - close warning
  PITCH_THRESHOLD: 15,        // degrees - looking up/down
  ROLL_THRESHOLD: 20,         // degrees - head tilt
  MOTION_THRESHOLD: 0.1       // g - motion detection
} as const;