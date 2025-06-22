#!/usr/bin/env python3
"""
Photo Analysis Script for TAN OAK Room Navigation
Uses CLIP embeddings to find similar reference images for position detection
"""

import os
import json
import base64
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhotoAnalyzer:
    def __init__(self, reference_images_dir: str):
        self.reference_dir = Path(reference_images_dir)
        # Use CLIP model for image embeddings
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.reference_embeddings = {}
        self.reference_metadata = {}
        
    def process_reference_images(self):
        """Process all reference images and create embeddings"""
        logger.info("Processing reference images...")
        
        # Support both .jpg and .jpeg extensions
        image_files = list(self.reference_dir.glob("*.jpg")) + list(self.reference_dir.glob("*.jpeg"))
        
        for image_file in image_files:
            try:
                # Load image
                with open(image_file, "rb") as f:
                    image_data = f.read()
                
                # Create embedding
                image = Image.open(io.BytesIO(image_data))
                embedding = self.model.encode(image)
                
                # Store embedding and metadata
                self.reference_embeddings[image_file.stem] = embedding
                self.reference_metadata[image_file.stem] = {
                    "filename": image_file.name,
                    "position": self._extract_position_from_filename(image_file.stem),
                    "landmarks": []  # To be filled manually or via AI
                }
                
                logger.info(f"Processed: {image_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
    
    def _extract_position_from_filename(self, filename: str) -> str:
        """Extract position info from filename if available"""
        # Example: "chair_1_facing_screen.jpg" → "sitting at chair 1 facing screen"
        # Example: "door_entrance.jpg" → "near door entrance"
        
        filename_lower = filename.lower()
        
        if "chair" in filename_lower:
            return f"sitting at {filename_lower.replace('_', ' ')}"
        elif "door" in filename_lower:
            return f"near {filename_lower.replace('_', ' ')}"
        elif "center" in filename_lower:
            return f"at {filename_lower.replace('_', ' ')}"
        else:
            return filename_lower.replace('_', ' ')
    
    def find_similar_position(self, current_image_base64: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Find most similar reference images to current view"""
        try:
            # Decode current image
            image_data = base64.b64decode(current_image_base64)
            current_image = Image.open(io.BytesIO(image_data))
            
            # Get embedding for current image
            current_embedding = self.model.encode(current_image)
            
            # Calculate similarities
            similarities = []
            for ref_name, ref_embedding in self.reference_embeddings.items():
                similarity = np.dot(current_embedding, ref_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(ref_embedding)
                )
                position = self.reference_metadata[ref_name]["position"]
                similarities.append((ref_name, similarity, position))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar position: {e}")
            return []
    
    def analyze_position(self, current_image_base64: str) -> Dict:
        """Analyze current position using similarity search"""
        similar_positions = self.find_similar_position(current_image_base64)
        
        if not similar_positions:
            return {"position": "unknown", "confidence": "low", "matches": []}
        
        # Get best match
        best_match = similar_positions[0]
        best_name, best_similarity, best_position = best_match
        
        # Determine confidence
        if best_similarity > 0.8:
            confidence = "high"
        elif best_similarity > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "position": best_position,
            "confidence": confidence,
            "similarity_score": float(best_similarity),
            "best_match": best_name,
            "top_matches": [
                {"name": name, "similarity": float(sim), "position": pos}
                for name, sim, pos in similar_positions
            ]
        }
    
    def save_embeddings(self, filepath: str):
        """Save embeddings and metadata to file"""
        data = {
            "embeddings": {k: v.tolist() for k, v in self.reference_embeddings.items()},
            "metadata": self.reference_metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Load embeddings and metadata from file"""
        if not os.path.exists(filepath):
            logger.warning(f"Embeddings file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        self.reference_embeddings = {
            k: np.array(v) for k, v in data["embeddings"].items()
        }
        self.reference_metadata = data["metadata"]
        
        logger.info(f"Loaded embeddings from {filepath}")


def main():
    """Test the photo analyzer"""
    # Setup
    reference_dir = "/Users/radhikadanda/vision-assistant/reference_images/tan_oak"
    embeddings_file = "/Users/radhikadanda/vision-assistant/tan_oak_embeddings.json"
    
    analyzer = PhotoAnalyzer(reference_dir)
    
    # Process reference images if embeddings don't exist
    if not os.path.exists(embeddings_file):
        print("Processing reference images...")
        analyzer.process_reference_images()
        analyzer.save_embeddings(embeddings_file)
    else:
        print("Loading existing embeddings...")
        analyzer.load_embeddings(embeddings_file)
    
    print(f"Loaded {len(analyzer.reference_embeddings)} reference positions")
    
    # Test with a sample image (you can replace this with Pi camera API call)
    # test_image_path = "test_current_view.jpg"
    # if os.path.exists(test_image_path):
    #     with open(test_image_path, "rb") as f:
    #         test_image_b64 = base64.b64encode(f.read()).decode('utf-8')
    #     
    #     result = analyzer.analyze_position(test_image_b64)
    #     print(f"Position Analysis: {result}")


if __name__ == "__main__":
    main()
