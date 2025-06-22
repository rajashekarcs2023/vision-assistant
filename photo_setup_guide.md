# Photo-Based Indoor Navigation Setup

## How It Works

**Vector Similarity Search**: Your 20-25 photos become the "ground truth" for room positions.

```
Your Photos → CLIP Embeddings → Vector Database
Current Camera View → CLIP Embedding → Find Most Similar Photo → Position!
```

## Setup Steps

### 1. Prepare Your Photos
```bash
# Create directory structure
mkdir -p /Users/radhikadanda/vision-assistant/reference_images/tan_oak

# Copy your 20-25 TAN OAK room photos to this directory
# Name them descriptively:
# - chair_1_facing_screen.jpg
# - chair_2_facing_windows.jpg  
# - standing_center_facing_door.jpg
# - near_door_entrance.jpg
# - corner_table_view.jpg
```

### 2. Install Dependencies
```bash
pip install sentence-transformers pillow numpy
```

### 3. Process Your Photos
```bash
# This will create embeddings from your photos
python photo_analyzer.py
```

### 4. Test the System
```bash
# Run the navigation agent
python indoor_navigation_agent.py
```

## Photo Naming Convention

**Good naming helps with position detection:**

```
Position-based names:
- chair_front_left.jpg
- chair_back_right.jpg
- standing_center.jpg
- near_door.jpg
- corner_window_side.jpg

Orientation-based names:
- facing_screen.jpg
- facing_windows.jpg
- facing_door.jpg
- back_to_door.jpg
```

## How Position Detection Works

### Step 1: Reference Processing
```python
# Your photos are processed once
for photo in your_photos:
    embedding = CLIP_model.encode(photo)
    store_embedding_with_position(embedding, position_from_filename)
```

### Step 2: Live Position Detection
```python
# During demo
current_view = get_camera_image()
current_embedding = CLIP_model.encode(current_view)

# Find most similar reference photo
best_match = find_most_similar(current_embedding, reference_embeddings)
position = get_position_from_match(best_match)

# Result: "You're sitting at chair 3 facing the projector screen"
```

## Advantages of This Approach

✅ **Accurate**: Uses your actual room photos as reference
✅ **Fast**: Vector similarity search is very quick  
✅ **Robust**: Works even with lighting/angle changes
✅ **Scalable**: Easy to add more reference positions
✅ **No Manual Mapping**: Your photos define the positions automatically

## Demo Flow

```
User: "Where am I?"
→ Camera captures current view
→ System finds most similar reference photo (0.87 similarity)
→ "You're sitting at the front left chair facing the projector screen"

User: "How do I exit?"
→ Based on identified position: "Turn right 90 degrees, walk straight to the door"
```

## Troubleshooting

### Low Similarity Scores
- **Add more reference photos** from that position
- **Check lighting conditions** - take photos in similar lighting
- **Improve photo quality** - avoid blurry images

### Wrong Position Detection
- **Better photo naming** - be more specific about positions
- **More reference angles** - take photos from multiple angles at each position
- **Remove duplicate positions** - don't have too many similar photos

### Performance Issues
- **Reduce photo resolution** to 640x480 for faster processing
- **Cache embeddings** (already implemented)
- **Limit reference photos** to 15-20 best ones

This approach will give you **much better accuracy** than trying to guess room landmarks, because it uses your actual room photos as the reference!
