// Convex functions for Indoor Navigation Demo
// Add these to your existing Convex functions

import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

// Upload navigation image (called by Pi)
export const upload_navigation_image = mutation({
  args: {
    image_data: v.string(), // base64 encoded image
    timestamp: v.string(),
    source: v.string(),
    metadata: v.optional(v.object({
      purpose: v.string(),
      camera_type: v.string(),
      resolution: v.string(),
    })),
  },
  handler: async (ctx, args) => {
    try {
      // Store the navigation image
      const imageId = await ctx.db.insert("navigation_images", {
        image_data: args.image_data,
        timestamp: args.timestamp,
        source: args.source,
        metadata: args.metadata || {},
        created_at: Date.now(),
      });

      console.log(`📸 Navigation image stored: ${imageId}`);
      
      return { 
        success: true, 
        image_id: imageId,
        message: "Navigation image uploaded successfully"
      };
      
    } catch (error) {
      console.error("Error storing navigation image:", error);
      return { 
        success: false, 
        error: error.message 
      };
    }
  },
});

// Get recent navigation images (called by Mac navigation agent)
export const get_recent_navigation_images = query({
  args: {
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const limit = args.limit || 1;
    
    try {
      const recentImages = await ctx.db
        .query("navigation_images")
        .order("desc")
        .take(limit);

      return recentImages.map(img => ({
        id: img._id,
        image_data: img.image_data,
        timestamp: img.timestamp,
        source: img.source,
        metadata: img.metadata,
        created_at: img.created_at,
      }));
      
    } catch (error) {
      console.error("Error fetching navigation images:", error);
      return [];
    }
  },
});

// Get latest navigation image URL (for Mac agent)
export const get_latest_navigation_image = query({
  args: {},
  handler: async (ctx, args) => {
    try {
      const latestImage = await ctx.db
        .query("navigation_images")
        .order("desc")
        .first();

      if (!latestImage) {
        return { success: false, message: "No images found" };
      }

      return {
        success: true,
        image_id: latestImage._id,
        image_data: latestImage.image_data,
        timestamp: latestImage.timestamp,
        metadata: latestImage.metadata,
      };
      
    } catch (error) {
      console.error("Error fetching latest image:", error);
      return { success: false, error: error.message };
    }
  },
});

// Clean up old navigation images (optional maintenance)
export const cleanup_old_navigation_images = mutation({
  args: {
    keep_count: v.optional(v.number()), // Number of recent images to keep
  },
  handler: async (ctx, args) => {
    const keepCount = args.keep_count || 50;
    
    try {
      // Get all images sorted by creation time
      const allImages = await ctx.db
        .query("navigation_images")
        .order("desc")
        .collect();

      if (allImages.length <= keepCount) {
        return { 
          success: true, 
          message: `Only ${allImages.length} images found, no cleanup needed`
        };
      }

      // Delete oldest images
      const imagesToDelete = allImages.slice(keepCount);
      let deletedCount = 0;

      for (const img of imagesToDelete) {
        await ctx.db.delete(img._id);
        deletedCount++;
      }

      return {
        success: true,
        message: `Cleaned up ${deletedCount} old navigation images`,
        remaining_count: keepCount,
      };
      
    } catch (error) {
      console.error("Error cleaning up images:", error);
      return { success: false, error: error.message };
    }
  },
});

// Get navigation service status
export const get_navigation_service_status = query({
  args: {},
  handler: async (ctx, args) => {
    try {
      const latestImage = await ctx.db
        .query("navigation_images")
        .order("desc")
        .first();

      const totalImages = await ctx.db
        .query("navigation_images")
        .collect()
        .then(imgs => imgs.length);

      return {
        service_active: !!latestImage,
        last_image_time: latestImage?.timestamp || null,
        last_image_id: latestImage?._id || null,
        total_images_stored: totalImages,
        camera_type: latestImage?.metadata?.camera_type || "unknown",
      };
      
    } catch (error) {
      console.error("Error checking service status:", error);
      return {
        service_active: false,
        error: error.message,
      };
    }
  },
});
