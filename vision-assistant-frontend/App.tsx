import React, { useState, useEffect, useRef, useCallback } from 'react';
import WebcamFeed from './components/WebcamFeed';
import StatusIndicator from './components/StatusIndicator';
import SensorDisplay from './components/SensorDisplay'; // NEW: Add sensor display
import { analyzeImageForHazards, analyzeImageForHazardsWithSensorContext } from './services/geminiService'; // ADD: Enhanced function
import { DetectionStatus, WebcamFeedRef, HazardAnalysisResponse, SensorPackage } from './types'; // ADD: SensorPackage
import { 
  FRAME_CAPTURE_INTERVAL_MS, 
  HAZARD_TYPE_PATH_CLEAR,
  MESSAGE_PATH_CLEAR_DEFAULT,
  MESSAGE_UNCLEAR_IMAGE,
  MESSAGE_ANALYSIS_ERROR,
  WEBSOCKET_URL // NEW: Add WebSocket URL
} from './constants';

const App: React.FC = () => {
  // KEEP: All your existing state
  const [status, setStatus] = useState<DetectionStatus>(DetectionStatus.INITIALIZING);
  const [statusMessage, setStatusMessage] = useState<string>("Initializing application...");
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [apiKeyPresent, setApiKeyPresent] = useState<boolean>(false);
  
  // ADD: New state for Pi integration
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [sensorData, setSensorData] = useState<SensorPackage | null>(null);
  const [lastFrameData, setLastFrameData] = useState<string | null>(null);
  
  // KEEP: All your existing refs
  const webcamRef = useRef<WebcamFeedRef>(null);
  const processingFrameRef = useRef<boolean>(false);
  const lastSpokenMessageRef = useRef<string | null>(null);
  const lastHazardTypeRef = useRef<string | null>(null);
  
  // ADD: New refs for WebSocket
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const statusRef = useRef<DetectionStatus>(status);
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  // API key check
  useEffect(() => {
    const key = process.env.GEMINI_API_KEY;  // Keep as you had it
    if (key && key.trim() !== "") {
      setApiKeyPresent(true);
      setStatus(DetectionStatus.IDLE); 
      setStatusMessage("API Key found. Connecting to Pi...");
    } else {
      setApiKeyPresent(false);
      setStatus(DetectionStatus.API_KEY_MISSING);
      setStatusMessage("Error: GEMINI_API_KEY environment variable is not set.");
    }
  }, []);

  // ADD: WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    setStatus(DetectionStatus.WEBSOCKET_CONNECTING);
    setStatusMessage("Connecting to Pi sensors...");

    try {
      wsRef.current = new WebSocket(WEBSOCKET_URL);
      
      wsRef.current.onopen = () => {
        console.log('🔗 WebSocket connected to Pi');
        setWsConnected(true);
        setStatus(DetectionStatus.IDLE);
        setStatusMessage("Connected to Pi sensors. Ready to start detection.");
        
        // Clear any reconnection timeout
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };

      wsRef.current.onmessage = (event) => {
        try {
          const sensorPackage: SensorPackage = JSON.parse(event.data);
          setSensorData(sensorPackage);
          
          // FIXED: Store frame data if available
          if (sensorPackage.camera?.image_data) {
            setLastFrameData(sensorPackage.camera.image_data);
          }
          
          // FIXED: Process frame if detection is active and we have frame data
          if (isDetecting && sensorPackage.camera?.image_data && !processingFrameRef.current) {
            processFrameWithSensorContext(sensorPackage);
          }
        } catch (error) {
          console.error('Error parsing sensor data:', error);
        }
      };

      wsRef.current.onclose = () => {
        console.log('🔌 WebSocket disconnected from Pi');
        setWsConnected(false);
        setStatus(DetectionStatus.WEBSOCKET_DISCONNECTED);
        setStatusMessage("Disconnected from Pi. Attempting to reconnect...");
        
        // Attempt to reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 3000);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus(DetectionStatus.WEBSOCKET_DISCONNECTED);
        setStatusMessage("Connection error. Check if Pi sensor service is running.");
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setStatus(DetectionStatus.WEBSOCKET_DISCONNECTED);
      setStatusMessage("Failed to connect to Pi. Is the sensor service running?");
    }
  }, [isDetecting]);

  // ADD: Connect to WebSocket when component mounts and API key is available
  useEffect(() => {
    if (apiKeyPresent) {
      connectWebSocket();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [apiKeyPresent, connectWebSocket]);

  // UPDATE: Camera status handler for fallback mode
  const handleCameraStatusChange = useCallback((cameraStatus: DetectionStatus, message?: string) => {
    // Only use webcam fallback if not connected to Pi
    if (!wsConnected && statusRef.current !== DetectionStatus.API_KEY_MISSING) {
      setStatus(cameraStatus);
      if (message) setStatusMessage(message);
      
      if (cameraStatus === DetectionStatus.CAMERA_ERROR) {
        setIsDetecting(false); 
      } else if (cameraStatus === DetectionStatus.IDLE && apiKeyPresent) {
        setStatusMessage(message || "Camera ready. Click 'Start Detection'.");
      }
    }
  }, [wsConnected, apiKeyPresent]);

  // ENHANCE: Speech function with sensor context
  const speak = (text: string, hazardType: string) => {
    // Avoid re-speaking the exact same hazard message if it's the same type of hazard consecutively
    if (text === lastSpokenMessageRef.current && hazardType === lastHazardTypeRef.current) {
      return; 
    }
    
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      
      // ADD: Adjust speech rate based on motion
      if (sensorData?.accelerometer?.is_moving) {
        utterance.rate = 1.1; // Slightly faster when moving
        utterance.volume = 0.9; // Slightly louder when moving
      } else {
        utterance.rate = 1.0;
        utterance.volume = 0.8;
      }
      
      speechSynthesis.speak(utterance);
      lastSpokenMessageRef.current = text;
      lastHazardTypeRef.current = hazardType;
    } else {
      console.warn("Text-to-speech not supported in this browser.");
    }
  };

  // ADD: New function for processing frames with sensor context
  const processFrameWithSensorContext = async (sensorPackage: SensorPackage) => {
    // FIXED: Check for image_data instead of data
    if (processingFrameRef.current || !sensorPackage.camera?.image_data) return;

    processingFrameRef.current = true;
    
    // Update status to processing
    if (statusRef.current === DetectionStatus.DETECTING || 
        statusRef.current === DetectionStatus.PATH_CLEAR || 
        statusRef.current === DetectionStatus.HAZARD_DETECTED) {
      setStatus(DetectionStatus.PROCESSING);
      setStatusMessage("Processing frame with sensor data...");
    }

    try {
      // FIXED: Use image_data instead of data
      const analysisResult = await analyzeImageForHazardsWithSensorContext(
        sensorPackage.camera.image_data, 
        sensorPackage
      );
      
      if ('error' in analysisResult) {
        console.error("Hazard analysis error:", analysisResult.error);
        if (statusRef.current !== DetectionStatus.WEBSOCKET_DISCONNECTED && 
            statusRef.current !== DetectionStatus.API_KEY_MISSING) {
          setStatus(DetectionStatus.ERROR);
          setStatusMessage(analysisResult.error.substring(0, 100));
        }
      } else {
        const { hazard_type, message } = analysisResult as HazardAnalysisResponse;
        
        // Generate contextual message based on sensor data
        const contextualMessage = generateContextualMessage(message, sensorPackage);
        
        if (hazard_type === HAZARD_TYPE_PATH_CLEAR) {
          setStatus(DetectionStatus.PATH_CLEAR);
          setStatusMessage(contextualMessage);
          
          // Speak less frequently when path is clear
          if (lastHazardTypeRef.current !== HAZARD_TYPE_PATH_CLEAR) {
            speak(contextualMessage, hazard_type);
          }
        } else if (hazard_type === "UNCLEAR_IMAGE") {
          setStatus(DetectionStatus.DETECTING);
          setStatusMessage(contextualMessage);
          
          if (lastHazardTypeRef.current !== "UNCLEAR_IMAGE") {
            speak(contextualMessage, hazard_type);
          }
        } else {
          // Hazard detected
          setStatus(DetectionStatus.HAZARD_DETECTED);
          setStatusMessage(contextualMessage);
          speak(contextualMessage, hazard_type);
        }
      }
    } catch (error) {
      console.error("Error during frame analysis:", error);
      if (statusRef.current !== DetectionStatus.WEBSOCKET_DISCONNECTED && 
          statusRef.current !== DetectionStatus.API_KEY_MISSING) {
        setStatus(DetectionStatus.ERROR);
        setStatusMessage(error instanceof Error ? error.message : MESSAGE_ANALYSIS_ERROR);
      }
    }
    
    processingFrameRef.current = false;
  };

  // ADD: Generate contextual message based on sensor data
  const generateContextualMessage = (baseMessage: string, sensorPackage: SensorPackage): string => {
    const accel = sensorPackage.accelerometer;
    const ultrasonic = sensorPackage.ultrasonic;
    
    if (!accel) return baseMessage;
    
    let contextualMessage = baseMessage;
    
    // Add motion context
    if (accel.is_moving) {
      if (baseMessage.includes("Hazard") || baseMessage.includes("obstacle")) {
        contextualMessage += " Stop and assess before proceeding.";
      }
    } else {
      if (baseMessage === MESSAGE_PATH_CLEAR_DEFAULT) {
        contextualMessage = "Path clear. You may proceed.";
      }
    }
    
    // Add ultrasonic context for close obstacles
    if (ultrasonic && ultrasonic.distance_cm < 50) {
      contextualMessage += ` Object ${ultrasonic.distance_cm}cm ahead.`;
    }
    
    // Add head orientation context
    if (Math.abs(accel.pitch) > 15) {
      const direction = accel.pitch > 0 ? "up" : "down";
      contextualMessage += ` You're looking ${direction}.`;
    }
    
    return contextualMessage;
  };

  // KEEP: Your existing useEffect for webcam fallback (only runs when not connected to Pi)
  useEffect(() => {
    // Only run this if NOT connected to Pi WebSocket (fallback to webcam)
    if (!wsConnected && isDetecting && apiKeyPresent && 
        statusRef.current !== DetectionStatus.CAMERA_ERROR && 
        statusRef.current !== DetectionStatus.API_KEY_MISSING) {
      
      setStatus(DetectionStatus.DETECTING);
      setStatusMessage("Detection active: Analyzing surroundings...");
      lastSpokenMessageRef.current = null;
      lastHazardTypeRef.current = null;

      const intervalId = setInterval(async () => {
        if (processingFrameRef.current || !webcamRef.current) return;

        processingFrameRef.current = true;
        if (statusRef.current !== DetectionStatus.CAMERA_ERROR && 
            statusRef.current !== DetectionStatus.API_KEY_MISSING) {
          if(statusRef.current === DetectionStatus.DETECTING || 
             statusRef.current === DetectionStatus.PATH_CLEAR || 
             statusRef.current === DetectionStatus.HAZARD_DETECTED) {
            setStatus(DetectionStatus.PROCESSING);
            setStatusMessage("Processing frame...");
          }
        }

        const base64ImageData = webcamRef.current.captureFrame();

        if (base64ImageData) {
          try {
            const analysisResult = await analyzeImageForHazards(base64ImageData);
            
            if ('error' in analysisResult) {
              console.error("Hazard analysis error:", analysisResult.error);
              if (statusRef.current !== DetectionStatus.CAMERA_ERROR && 
                  statusRef.current !== DetectionStatus.API_KEY_MISSING) {
                setStatus(DetectionStatus.ERROR);
                setStatusMessage(analysisResult.error.substring(0,100));
              }
            } else {
              const { hazard_type, message } = analysisResult as HazardAnalysisResponse;
              
              if (hazard_type === HAZARD_TYPE_PATH_CLEAR) {
                setStatus(DetectionStatus.PATH_CLEAR);
                setStatusMessage(message || MESSAGE_PATH_CLEAR_DEFAULT);
                if (lastHazardTypeRef.current !== HAZARD_TYPE_PATH_CLEAR) {
                  speak(message || MESSAGE_PATH_CLEAR_DEFAULT, hazard_type);
                }
              } else if (hazard_type === "UNCLEAR_IMAGE") {
                setStatus(DetectionStatus.DETECTING);
                setStatusMessage(message || MESSAGE_UNCLEAR_IMAGE);
                if (lastHazardTypeRef.current !== "UNCLEAR_IMAGE") {
                  speak(message || MESSAGE_UNCLEAR_IMAGE, hazard_type);
                }
              } else {
                setStatus(DetectionStatus.HAZARD_DETECTED);
                setStatusMessage(message || "Hazard detected!");
                speak(message || "Caution, hazard detected!", hazard_type);
              }
            }
          } catch (error) {
            console.error("Error during frame analysis:", error);
            if (statusRef.current !== DetectionStatus.CAMERA_ERROR && 
                statusRef.current !== DetectionStatus.API_KEY_MISSING) {
              setStatus(DetectionStatus.ERROR);
              setStatusMessage(error instanceof Error ? error.message : MESSAGE_ANALYSIS_ERROR);
            }
          }
        } else {
          if (statusRef.current !== DetectionStatus.CAMERA_ERROR && 
              statusRef.current !== DetectionStatus.API_KEY_MISSING &&
              statusRef.current !== DetectionStatus.ERROR) {
            setStatus(DetectionStatus.DETECTING); 
            setStatusMessage("Could not capture frame, retrying...");
          }
        }
        processingFrameRef.current = false;
      }, FRAME_CAPTURE_INTERVAL_MS);

      return () => {
        clearInterval(intervalId);
        processingFrameRef.current = false;
        if (statusRef.current !== DetectionStatus.API_KEY_MISSING && 
            statusRef.current !== DetectionStatus.CAMERA_ERROR) {
          setStatus(DetectionStatus.IDLE);
          setStatusMessage("Detection stopped.");
        }
      };
    }
  }, [isDetecting, apiKeyPresent, wsConnected]); // ADD: wsConnected dependency

  // UPDATE: Toggle detection function
  const toggleDetection = () => {
    if (isDetecting) {
      setIsDetecting(false);
      setStatus(DetectionStatus.IDLE);
      setStatusMessage("Detection stopped.");
    } else {
      if (wsConnected && lastFrameData) {
        // Pi mode
        setIsDetecting(true);
        setStatus(DetectionStatus.DETECTING);
        setStatusMessage("Detection active: Analyzing surroundings with sensors...");
        lastSpokenMessageRef.current = null;
        lastHazardTypeRef.current = null;
      } else if (!wsConnected && webcamRef.current) {
        // Fallback webcam mode
        setIsDetecting(true);
      } else if (!wsConnected) {
        setStatusMessage("Cannot start - not connected to Pi sensors and no webcam available.");
      } else {
        setStatusMessage("Cannot start - no camera data available.");
      }
    }
  };
  
  // UPDATE: Can start detection conditions
  const canStartDetection = apiKeyPresent && 
    ((wsConnected && lastFrameData) || (!wsConnected && webcamRef.current)) &&
    (statusRef.current === DetectionStatus.IDLE || 
     statusRef.current === DetectionStatus.PATH_CLEAR || 
     statusRef.current === DetectionStatus.HAZARD_DETECTED ||
     statusRef.current === DetectionStatus.DETECTING ||
     statusRef.current === DetectionStatus.PROCESSING);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 space-y-6 bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100">
      <header className="text-center">
        <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 mb-2">
          SentientSight
        </h1>
        <p className="text-xl text-gray-300">AI-powered navigation assistance with sensor fusion.</p>
      </header>

      <main className="w-full max-w-4xl p-6 bg-gray-800 bg-opacity-50 rounded-xl shadow-2xl backdrop-blur-md border border-gray-700">
        {/* UPDATE: New grid layout for camera and sensor data */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Camera Feed Section */}
          <div>
            {wsConnected ? (
              <div className="relative w-full aspect-video bg-gray-800 rounded-lg shadow-xl overflow-hidden border-2 border-green-500">
                {lastFrameData ? (
                  <img 
                    src={`data:image/jpeg;base64,${lastFrameData}`} 
                    alt="Pi Camera Feed"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75">
                    <p className="text-lg">Waiting for camera data...</p>
                  </div>
                )}
                <div className="absolute top-2 left-2 bg-green-600 text-white px-2 py-1 rounded text-sm">
                  Pi Camera {wsConnected ? '🟢' : '🔴'}
                </div>
              </div>
            ) : (
              <WebcamFeed ref={webcamRef} onCameraStatusChange={handleCameraStatusChange} />
            )}
          </div>
          
          {/* NEW: Sensor Data Section */}
          <div>
            <SensorDisplay sensorData={sensorData} wsConnected={wsConnected} />
          </div>
        </div>
        
        <StatusIndicator status={status} message={statusMessage} />

        <button
          onClick={toggleDetection}
          disabled={!canStartDetection && !isDetecting}
          aria-live="polite"
          className={`w-full py-3 px-6 text-lg font-semibold rounded-lg transition-all duration-300 ease-in-out
            ${(!canStartDetection && !isDetecting)
              ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
              : isDetecting 
                ? 'bg-red-600 hover:bg-red-700 text-white shadow-lg hover:shadow-red-500/50' 
                : 'bg-green-500 hover:bg-green-600 text-white shadow-lg hover:shadow-green-500/50'
            } focus:outline-none focus:ring-4 ${isDetecting ? 'focus:ring-red-400' : 'focus:ring-green-400'}`}
        >
          {isDetecting ? 'Stop Detection' : 'Start Detection'}
        </button>
        
        {/* UPDATE: Status Messages */}
        {!apiKeyPresent && (
          <p className="text-center text-red-400 mt-3 text-sm" role="alert">
            API Key is missing. Please set the <code>VITE_API_KEY</code> environment variable.
          </p>
        )}
        
        {!wsConnected && apiKeyPresent && (
          <div className="text-center text-yellow-400 mt-3 text-sm" role="alert">
            <p>Not connected to Pi sensors. Using webcam fallback.</p>
            <p className="text-xs">Make sure the Pi sensor service is running: <code>python3 sensor_service.py</code></p>
          </div>
        )}
        
        {status === DetectionStatus.CAMERA_ERROR && !wsConnected && (
          <p className="text-center text-red-400 mt-3 text-sm" role="alert">
            {statusMessage || "Camera not available. Detection cannot start."}
          </p>
        )}
      </main>

      <footer className="mt-8 text-center text-gray-500">
        <p>&copy; {new Date().getFullYear()} SentientSight. Sensor-Enhanced Navigation.</p>
      </footer>
    </div>
  );
};

export default App;