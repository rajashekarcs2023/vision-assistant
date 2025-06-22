import asyncio
import logging
from dotenv import load_dotenv
import os
import json
import base64
import requests
from typing import Optional, Dict, List
import datetime
from pathlib import Path
from photo_analyzer import PhotoAnalyzer

from livekit import agents
from livekit.agents import Agent, AgentSession
import livekit.plugins.openai as openai_plugin
from livekit.plugins import silero

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reference Images Directory
REFERENCE_IMAGES_DIR = Path(__file__).parent / "reference_images" / "tan_oak"

# Room Database - TAN OAK Room (4th Floor MLK)
ROOM_DATABASE = {
    "tan_oak_room": {
        "name": "TAN OAK Conference Room",
        "floor": "4th Floor",
        "building": "MLK Student Union",
        
        "landmarks": {
            "conference_table": {
                "description": "Large wooden conference table with chairs",
                "visual_cues": ["rectangular table", "multiple chairs", "dark wooden surface"],
                "position": "center"
            },
            "projector_screen": {
                "description": "Wall-mounted projector screen",
                "visual_cues": ["white screen", "mounted on wall", "rectangular"],
                "position": "front_wall"
            },
            "windows": {
                "description": "Large windows with view outside",
                "visual_cues": ["natural light", "glass panels", "window frames"],
                "position": "exterior_wall"
            },
            "whiteboards": {
                "description": "Wall-mounted whiteboards",
                "visual_cues": ["white surface", "marker trays", "rectangular"],
                "position": "side_wall"
            },
            "door_entrance": {
                "description": "Main room entrance door",
                "visual_cues": ["door handle", "door frame", "exit sign"],
                "position": "entrance_wall"
            }
        },
        
        "exits": {
            "main_door": {
                "description": "Main entrance/exit door",
                "location": "entrance_wall",
                "leads_to": "hallway_to_elevator",
                "visual_cues": ["door handle", "exit sign", "hallway visible"]
            }
        },
        
        "navigation_instructions": {
            "from_center_to_exit": {
                "if_facing_screen": "Turn around 180 degrees, walk straight to the door",
                "if_facing_windows": "Turn left 90 degrees, walk straight to the door", 
                "if_facing_door": "Walk straight ahead to the door",
                "if_back_to_door": "Turn around 180 degrees, walk straight to the door"
            },
            "from_chair_to_exit": {
                "default": "Stand up, face the door, and walk straight to the exit"
            }
        },
        
        "reference_positions": {
            # We'll map specific viewpoints from your photos
            # Format: "photo_name": {"position": "description", "landmarks_visible": [...]}
        }
    }
}

class ConvexClient:
    def __init__(self, deployment_url: str):
        """Initialize Convex client for getting navigation images"""
        self.base_url = deployment_url.rstrip('/')
        
    def get_latest_navigation_image(self) -> Optional[str]:
        """Get the latest navigation image as base64 string"""
        url = f"{self.base_url}/get_latest_navigation_image"
        
        try:
            response = requests.post(
                url,
                json={},
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get('success') and result.get('image_data'):
                logger.info(f"📸 Got latest image: {result.get('image_id')}")
                return result['image_data']  # Already base64 encoded
            else:
                logger.warning(f"No image available: {result.get('message', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting latest navigation image: {e}")
            return None
    
    def get_navigation_service_status(self) -> Dict:
        """Get status of the navigation camera service"""
        url = f"{self.base_url}/get_navigation_service_status"
        
        try:
            response = requests.post(
                url,
                json={},
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting service status: {e}")
            return {"service_active": False, "error": str(e)}

def load_reference_images() -> Dict:
    """Load reference images from the photos directory"""
    reference_images = {}
    
    if REFERENCE_IMAGES_DIR.exists():
        for image_file in REFERENCE_IMAGES_DIR.glob("*.jpg"):
            try:
                with open(image_file, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    reference_images[image_file.stem] = image_data
                    logger.info(f"Loaded reference image: {image_file.name}")
            except Exception as e:
                logger.error(f"Error loading {image_file.name}: {e}")
    
    return reference_images

class IndoorNavigationAgent(Agent):
    """Indoor Navigation Assistant for TAN OAK Room"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an indoor navigation assistant helping blind users navigate the TAN OAK conference room. "
                "You can identify their location in the room and provide clear exit directions. "
                "Speak naturally and give precise, helpful directions. "
                "Your main functions are: 1) Tell users where they are in the room, 2) Guide them to the exit."
            )
        )
        self.convex_client = ConvexClient(os.getenv('CONVEX_URL'))
        self.reference_images = load_reference_images()
        
        # Initialize photo analyzer
        reference_dir = Path(__file__).parent / "reference_images" / "tan_oak"
        embeddings_file = Path(__file__).parent / "tan_oak_embeddings.json"
        self.photo_analyzer = PhotoAnalyzer(str(reference_dir))
        
        # Load or create embeddings
        if embeddings_file.exists():
            self.photo_analyzer.load_embeddings(str(embeddings_file))
            logger.info("Loaded existing photo embeddings")
        else:
            logger.info("No embeddings found - will process reference images on first use")
        
    async def get_camera_image(self) -> Optional[str]:
        """Get current camera image from Convex as base64"""
        try:
            image_data = self.convex_client.get_latest_navigation_image()
            if image_data:
                return image_data
            else:
                logger.error("Failed to get image from Convex")
                return None
        except Exception as e:
            logger.error(f"Error getting camera image: {e}")
            return None
    
    async def analyze_room_position(self, image_base64: str) -> Dict:
        """Analyze image to determine position in TAN OAK room using photo similarity"""
        
        # First, try photo similarity analysis
        try:
            position_result = self.photo_analyzer.analyze_position(image_base64)
            
            if position_result and position_result.get("confidence") != "low":
                # Good match found via photo similarity
                return {
                    "method": "photo_similarity",
                    "estimated_position": position_result["position"],
                    "confidence": position_result["confidence"],
                    "similarity_score": position_result["similarity_score"],
                    "best_match": position_result["best_match"],
                    "landmarks_visible": [],  # Could extract from metadata
                    "facing_direction": "unknown",  # Could infer from position
                    "exit_visible": False  # Could determine from position
                }
        except Exception as e:
            logger.error(f"Photo similarity analysis failed: {e}")
        
        # Fallback to AI vision analysis
        logger.info("Using AI vision analysis as fallback")
        
        room_data = ROOM_DATABASE["tan_oak_room"]
        landmarks_info = "\n".join([
            f"- {name}: {data['description']} (look for: {', '.join(data['visual_cues'])})"
            for name, data in room_data["landmarks"].items()
        ])
        
        prompt = f"""Analyze this image from TAN OAK conference room (4th floor MLK building) to determine user's position.

ROOM LAYOUT REFERENCE:
{landmarks_info}

ANALYSIS TASKS:
1. IDENTIFY visible landmarks in the image
2. DETERMINE user's approximate position and facing direction  
3. ASSESS visibility of the main door/exit

RESPONSE FORMAT (JSON):
{{
    "landmarks_visible": ["list of landmarks you can see"],
    "estimated_position": "where user appears to be",
    "facing_direction": "what direction user is facing",
    "exit_visible": true/false,
    "confidence": "high/medium/low"
}}"""

        try:
            import openai
            client = openai.AsyncOpenAI()
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            analysis_text = response.choices[0].message.content
            try:
                analysis_data = json.loads(analysis_text)
                analysis_data["method"] = "ai_vision"
                return analysis_data
            except json.JSONDecodeError:
                return {
                    "method": "ai_vision",
                    "landmarks_visible": [],
                    "estimated_position": "unable to determine",
                    "facing_direction": "unknown",
                    "exit_visible": False,
                    "confidence": "low",
                    "raw_analysis": analysis_text
                }
                
        except Exception as e:
            logger.error(f"Error analyzing room position: {e}")
            return {
                "method": "error",
                "landmarks_visible": [],
                "estimated_position": "analysis error",
                "facing_direction": "unknown", 
                "exit_visible": False,
                "confidence": "low"
            }
    
    async def get_exit_directions(self, position_analysis: Dict) -> str:
        """Generate exit directions based on position analysis"""
        
        room_data = ROOM_DATABASE["tan_oak_room"]
        navigation_rules = room_data["navigation_instructions"]["from_center_to_exit"]
        
        facing = position_analysis.get("facing_direction", "").lower()
        exit_visible = position_analysis.get("exit_visible", False)
        
        if exit_visible:
            return "I can see the exit door in your view. Walk straight ahead toward the door."
        
        # Determine direction based on what user is facing
        if "screen" in facing or "projector" in facing:
            return navigation_rules["if_facing_screen"]
        elif "window" in facing:
            return navigation_rules["if_facing_windows"] 
        elif "door" in facing:
            return navigation_rules["if_facing_door"]
        else:
            # Default instruction
            return "Turn around slowly until you can see the door, then walk straight toward it. The exit is the main door you entered through."

async def entrypoint(ctx: agents.JobContext):
    """Indoor Navigation Agent Entry Point"""
    logger.info("🧭 Starting Indoor Navigation Agent for TAN OAK Room")
    
    # Create agent session
    session = AgentSession(
        stt=openai_plugin.STT(
            detect_language=False,
        ),
        llm=openai_plugin.LLM(
            model="gpt-4o",
        ),
        tts=openai_plugin.TTS(
            voice="alloy",
        ),
        vad=silero.VAD.load(),
    )
    
    # Start session
    await session.start(
        room=ctx.room,
        agent=IndoorNavigationAgent()
    )
    
    # Connect to room
    await ctx.connect()
    
    logger.info("✅ Navigation agent ready")
    
    # Initial greeting
    await session.say("Hello! I'm your indoor navigation assistant for the TAN OAK room. I can tell you where you are and guide you to the exit. How can I help?")
    
    # Set up event handlers for user speech
    @session.on("user_speech")
    def on_user_speech(text: str):
        logger.info(f"👤 User said: '{text}'")
        # Use asyncio.create_task for async processing
        asyncio.create_task(handle_user_speech(session, text))
    
    async def handle_user_speech(session, text: str):
        """Handle user speech asynchronously"""
        user_message = text.lower()
        
        # Handle "Where am I?" requests
        if any(phrase in user_message for phrase in ["where am i", "where i am", "my location", "my position"]):
            logger.info("🔍 Processing location request...")
            await session.say("Let me look around and see where you are. One moment...")
            
            # Get agent instance and analyze position
            agent_instance = session.agent
            image_data = await agent_instance.get_camera_image()
            
            if image_data:
                position_analysis = await agent_instance.analyze_room_position(image_data)
                position = position_analysis.get("estimated_position", "unknown location")
                landmarks = position_analysis.get("landmarks_visible", [])
                confidence = position_analysis.get("confidence", "low")
                
                logger.info(f"📍 Position: {position}, Confidence: {confidence}")
                
                if confidence == "high" and landmarks:
                    response = f"You are {position}. I can see {', '.join(landmarks)} from your viewpoint."
                else:
                    response = f"You appear to be {position}, though I'm having some difficulty getting a clear view of the room landmarks."
                
                await session.say(response)
            else:
                await session.say("I'm having trouble accessing the latest image right now. Please check the camera connection.")
        
        # Handle exit/navigation requests  
        elif any(phrase in user_message for phrase in ["exit", "how to get out", "way out", "leave", "door"]):
            logger.info("🚪 Processing exit request...")
            await session.say("Let me see where you are so I can guide you to the exit...")
            
            agent_instance = session.agent
            image_data = await agent_instance.get_camera_image()
            
            if image_data:
                position_analysis = await agent_instance.analyze_room_position(image_data)
                exit_directions = await agent_instance.get_exit_directions(position_analysis)
                
                logger.info(f"🧭 Exit directions: {exit_directions}")
                await session.say(f"To exit the room: {exit_directions}")
            else:
                await session.say("I can't access the latest image right now, but the exit is the main door you entered through. Turn around slowly until you find the door, then walk straight toward it.")
        
        # Handle other conversation
        else:
            if any(word in user_message for word in ["navigate", "direction", "help", "find", "room"]):
                await session.say("I can help you with two main things: telling you where you are in the TAN OAK room, and guiding you to the exit. What would you like to know?")
            else:
                logger.info("💬 Generating general response...")
                # Use LLM for general conversation
                await session.say(f"I heard you say: {text}. I'm here to help with navigation in the TAN OAK room. You can ask me where you are or how to exit.")
    
    try:
        # Keep session running
        await asyncio.sleep(float('inf'))
        
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Navigation agent interrupted")
    finally:
        await session.aclose()
        logger.info("Navigation agent shutting down")

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
