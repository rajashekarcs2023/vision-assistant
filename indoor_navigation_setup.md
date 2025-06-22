# Indoor Navigation Demo Setup

## Overview
This demo provides voice-based indoor navigation for the **TAN OAK conference room** (4th floor MLK building). Users can ask:
- "Where am I?" - System identifies current position using photo similarity
- "How do I exit?" - System provides step-by-step directions to the door

## Architecture

```
Pi Camera → Convex Storage → Navigation Agent (Mac) → LiveKit Voice
                ↓
Your Reference Photos → CLIP Embeddings → Position Detection
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements_navigation.txt
```

### 2. Configure Environment Variables
```bash
# Copy and edit the environment file
cp .env.example .env

# Edit .env with your actual values:
# - CONVEX_DEPLOYMENT_URL: Your Convex deployment URL
# - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET: Your LiveKit credentials  
# - OPENAI_API_KEY: Your OpenAI API key
```

### 3. Prepare Reference Photos
```bash
# Create directory for your TAN OAK room photos
mkdir -p reference_images/tan_oak

# Copy your 20-25 room photos to this directory
# Name them descriptively (see photo_setup_guide.md for details)
```

### 4. Process Reference Photos (One-time Setup)
```bash
# This creates CLIP embeddings from your photos
python photo_analyzer.py
```

### 5. Run the Demo

#### Option A: LiveKit Agents CLI (Recommended)
```bash
python -m livekit.agents.cli start indoor_navigation_agent.py
```

#### Option B: Direct Python
```bash
python indoor_navigation_agent.py
```

## How It Works

### Image Acquisition
- Pi captures images and stores them in **Convex**
- Navigation agent gets the **latest image** from Convex (not direct Pi connection)
- Much more reliable than direct Pi camera API

### Position Detection  
1. **Photo Similarity**: Current view → CLIP embedding → Find most similar reference photo
2. **AI Fallback**: If similarity is low, use GPT-4V to analyze landmarks
3. **Confidence Scoring**: High similarity = precise position, low similarity = general guidance

### Voice Interaction
- **Speech-to-Text**: User voice commands
- **LLM Processing**: Natural language understanding
- **Text-to-Speech**: Spoken responses with position and directions

## Demo Script

### Starting the Demo
1. **Start LiveKit agent**: `python -m livekit.agents.cli start indoor_navigation_agent.py`
2. **Join room**: Use your LiveKit room URL
3. **Test voice commands**

### Voice Commands to Try

**Position Detection:**
- "Where am I?"
- "What's my location?"
- "Where am I in the room?"

**Exit Directions:**
- "How do I exit?"
- "How do I get out?"
- "Where's the door?"
- "How do I leave?"

## Expected Responses

### High Confidence Position (Photo Match)
```
User: "Where am I?"
Agent: "You're sitting at the front left chair facing the projector screen. 
        Confidence: High (94% similarity match)"
```

### Medium Confidence (AI Analysis)
```
User: "Where am I?"  
Agent: "I can see the conference table and projector screen. 
        You appear to be near the center of the room facing north."
```

### Exit Directions
```
User: "How do I exit?"
Agent: "From your current position at the front left chair, 
        turn right 90 degrees and walk straight toward the door. 
        It should be about 8 steps ahead of you."
```

## Troubleshooting

### "Can't access latest image"
- **Check Convex URL**: Make sure CONVEX_DEPLOYMENT_URL is correct
- **Check Pi connectivity**: Ensure Pi is capturing and uploading images
- **Check Convex function**: Verify `get_recent_images` function exists

### "Low confidence position"
- **Add more reference photos** from that location
- **Check photo naming** - use descriptive filenames
- **Verify embeddings**: Re-run `python photo_analyzer.py`

### "No voice response"
- **Check LiveKit credentials** in .env file
- **Check microphone permissions**
- **Verify OpenAI API key**

### "Navigation directions are generic"
- **Update room database** with actual TAN OAK layout
- **Add more position-specific exit routes**
- **Improve landmark descriptions**

## Architecture Benefits

✅ **Reliable Image Access**: Uses Convex storage instead of direct Pi connection
✅ **Photo-Based Positioning**: Your actual room photos provide ground truth
✅ **Voice-First Interface**: Natural speech interaction
✅ **Fallback Systems**: Multiple methods for position detection
✅ **Scalable**: Easy to add more rooms or reference positions

## Next Steps

1. **Test with actual room photos** in TAN OAK room
2. **Refine navigation instructions** based on real room layout  
3. **Add more reference positions** for better coverage
4. **Calibrate exit directions** with actual step counts and turns
