import { GoogleGenAI, GenerateContentResponse, Part } from "@google/genai";
import { GEMINI_MODEL_NAME, GEMINI_PROMPT_HAZARD_DETECTION } from '../constants';
import { HazardAnalysisResponse, SensorPackage } from "../types"; // ADD: SensorPackage import

let ai: GoogleGenAI | null = null;

const initializeAiClient = (): GoogleGenAI | null => {
  if (ai) return ai;
  const apiKey = process.env.GEMINI_API_KEY;  // Use your current variable
  if (!apiKey) {
    console.error("GEMINI_API_KEY environment variable not found.");
    return null;
  }
  ai = new GoogleGenAI({ apiKey });
  return ai;
};

// ADD: New function to build sensor context for AI
const buildSensorContextPrompt = (sensorData: SensorPackage): string => {
  let sensorContext = "\n\nSensor Context:\n";
  
  if (sensorData.accelerometer) {
    const accel = sensorData.accelerometer;
    sensorContext += `- Motion Status: ${accel.is_moving ? 'USER IS WALKING' : 'USER IS STATIONARY'}\n`;
    sensorContext += `- Head Orientation: Pitch ${accel.pitch.toFixed(1)}° (${accel.pitch > 15 ? 'looking up' : accel.pitch < -15 ? 'looking down' : 'looking forward'}), `;
    sensorContext += `Roll ${accel.roll.toFixed(1)}° (${accel.roll > 20 ? 'head tilted right' : accel.roll < -20 ? 'head tilted left' : 'head level'})\n`;
    sensorContext += `- Movement Intensity: ${accel.total_acceleration.toFixed(2)}g\n`;
  }
  
  if (sensorData.ultrasonic) {
    const distance = sensorData.ultrasonic.distance_cm;
    sensorContext += `- Ultrasonic Distance: ${distance}cm ahead\n`;
    if (distance < 50) {
      sensorContext += `- CRITICAL: Object detected very close (${distance}cm) - prioritize immediate obstacle warning\n`;
    } else if (distance < 100) {
      sensorContext += `- CAUTION: Object detected nearby (${distance}cm) - include distance information\n`;
    }
  }
  
  sensorContext += `\nAnalysis Priority Instructions:
- If user is WALKING and hazard detected: Emphasize immediate stopping and assessment
- If user is STATIONARY: Provide detailed directional guidance
- If looking UP: Prioritize overhead hazards (low branches, signs, etc.)
- If looking DOWN: Prioritize ground-level hazards (steps, holes, obstacles)
- If ultrasonic shows close object but camera doesn't: Mention "object detected nearby" with distance
- Always consider sensor data to provide the most relevant and timely warnings\n`;
  
  return sensorContext;
};

// ADD: New enhanced function with sensor context
export const analyzeImageForHazardsWithSensorContext = async (
  base64ImageData: string, 
  sensorData: SensorPackage
): Promise<HazardAnalysisResponse | { error: string }> => {
  const client = initializeAiClient();
  if (!client) {
    return { error: "Gemini AI client not initialized. API_KEY might be missing." };
  }

  const imagePart: Part = {
    inlineData: {
      mimeType: 'image/jpeg',
      data: base64ImageData,
    },
  };

  // Build enhanced prompt with sensor context
  const sensorContextPrompt = buildSensorContextPrompt(sensorData);
  const enhancedPrompt = GEMINI_PROMPT_HAZARD_DETECTION + sensorContextPrompt;

  const textPart: Part = {
    text: enhancedPrompt,
  };

  try {
    const response: GenerateContentResponse = await client.models.generateContent({
      model: GEMINI_MODEL_NAME,
      contents: { parts: [imagePart, textPart] },
      config: {
        responseMimeType: "application/json",
        // thinkingConfig: { thinkingBudget: 0 } // Enable for potentially lower latency
      },
    });

    let jsonStr = response.text.trim();
    const fenceRegex = /^```(\w*)?\s*\n?(.*?)\n?\s*```$/s;
    const match = jsonStr.match(fenceRegex);
    if (match && match[2]) {
      jsonStr = match[2].trim();
    }

    try {
      const parsedData = JSON.parse(jsonStr) as HazardAnalysisResponse;
      if (!parsedData.hazard_type || !parsedData.message) {
        console.warn("Parsed JSON is missing required fields:", parsedData);
        return { error: "AI response missing required fields (hazard_type or message)." };
      }
      
      // Log sensor-enhanced analysis for debugging
      console.log(`🧠 AI Analysis with sensors: ${parsedData.hazard_type} - Motion: ${sensorData.accelerometer?.is_moving ? 'Moving' : 'Stationary'}`);
      
      return parsedData;
    } catch (e) {
      console.error("Failed to parse JSON response from Gemini:", e, "Raw response:", jsonStr);
      return { error: "Failed to parse AI response as JSON." };
    }

  } catch (error) {
    console.error("Error calling Gemini API:", error);
    if (error instanceof Error) {
      return { error: `Gemini API Error: ${error.message}` };
    }
    return { error: "Unknown error calling Gemini API." };
  }
};

// KEEP: Your original function for backward compatibility (webcam fallback)
export const analyzeImageForHazards = async (base64ImageData: string): Promise<HazardAnalysisResponse | { error: string }> => {
  const client = initializeAiClient();
  if (!client) {
    return { error: "Gemini AI client not initialized. API_KEY might be missing." };
  }

  const imagePart: Part = {
    inlineData: {
      mimeType: 'image/jpeg',
      data: base64ImageData,
    },
  };

  const textPart: Part = { // Although the prompt is complex, it's still a text part for the vision model
    text: GEMINI_PROMPT_HAZARD_DETECTION,
  };

  try {
    const response: GenerateContentResponse = await client.models.generateContent({
      model: GEMINI_MODEL_NAME,
      contents: { parts: [imagePart, textPart] }, // Sending image and the main prompt
      config: { 
        responseMimeType: "application/json",
        // thinkingConfig: { thinkingBudget: 0 } // Enable for potentially lower latency
      }, 
    });
    
    let jsonStr = response.text.trim();
    const fenceRegex = /^```(\w*)?\s*\n?(.*?)\n?\s*```$/s;
    const match = jsonStr.match(fenceRegex);
    if (match && match[2]) {
      jsonStr = match[2].trim();
    }

    try {
      const parsedData = JSON.parse(jsonStr) as HazardAnalysisResponse;
      if (!parsedData.hazard_type || !parsedData.message) {
        console.warn("Parsed JSON is missing required fields:", parsedData);
        return { error: "AI response missing required fields (hazard_type or message)." };
      }
      return parsedData;
    } catch (e) {
      console.error("Failed to parse JSON response from Gemini:", e, "Raw response:", jsonStr);
      return { error: "Failed to parse AI response as JSON." };
    }

  } catch (error) {
    console.error("Error calling Gemini API:", error);
    if (error instanceof Error) {
      return { error: `Gemini API Error: ${error.message}` };
    }
    return { error: "Unknown error calling Gemini API." };
  }
};