
import React, { useRef, useEffect, useImperativeHandle, forwardRef, useState } from 'react';
import { WebcamFeedRef, DetectionStatus } from '../types';

interface WebcamFeedProps {
  onCameraStatusChange: (status: DetectionStatus, message?: string) => void;
}

const WebcamFeed = forwardRef<WebcamFeedRef, WebcamFeedProps>(({ onCameraStatusChange }, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [streamActive, setStreamActive] = useState(false);

  useEffect(() => {
    const setupWebcam = async () => {
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            onCameraStatusChange(DetectionStatus.IDLE, "Camera ready.");
            setStreamActive(true);
          }
        } else {
          onCameraStatusChange(DetectionStatus.CAMERA_ERROR, "getUserMedia not supported.");
          setStreamActive(false);
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
        let message = "Error accessing webcam. Please ensure permission is granted.";
        if (err instanceof Error) {
            if (err.name === "NotAllowedError") {
                message = "Camera permission denied. Please enable camera access in your browser settings.";
            } else if (err.name === "NotFoundError") {
                message = "No camera found. Please ensure a camera is connected.";
            }
        }
        onCameraStatusChange(DetectionStatus.CAMERA_ERROR, message);
        setStreamActive(false);
      }
    };

    setupWebcam();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Runs once on mount

  useImperativeHandle(ref, () => ({
    captureFrame: () => {
      if (videoRef.current && canvasRef.current && videoRef.current.readyState === 4 && videoRef.current.videoWidth > 0) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        if (context) {
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          // Return base64 image data (JPEG format for smaller size)
          return canvas.toDataURL('image/jpeg', 0.8).split(',')[1]; // Remove "data:image/jpeg;base64," prefix
        }
      }
      return null;
    },
  }));

  return (
    <div className="relative w-full max-w-2xl mx-auto aspect-video bg-gray-800 rounded-lg shadow-xl overflow-hidden border-2 border-blue-500">
      <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      {!streamActive && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75">
          <p className="text-lg">Initializing Camera...</p>
        </div>
      )}
    </div>
  );
});

export default WebcamFeed;
