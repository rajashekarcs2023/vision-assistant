import React from 'react';
import { DetectionStatus } from '../types';

interface StatusIndicatorProps {
  status: DetectionStatus;
  message: string;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status, message }) => {
  let bgColor = 'bg-gray-700';
  let textColor = 'text-gray-200';
  let icon = 'ℹ️';

  switch (status) {
    case DetectionStatus.IDLE:
      bgColor = 'bg-blue-600';
      textColor = 'text-blue-100';
      icon = '🔵';
      break;
    case DetectionStatus.INITIALIZING:
      bgColor = 'bg-yellow-500';
      textColor = 'text-yellow-900';
      icon = '⏳';
      break;
    case DetectionStatus.DETECTING:
    case DetectionStatus.PROCESSING:
      bgColor = 'bg-indigo-600';
      textColor = 'text-indigo-100';
      icon = '🔎';
      break;
    case DetectionStatus.HAZARD_DETECTED: // Updated from EVENT_DETECTED
      bgColor = 'bg-orange-500'; // Hazards are warnings, so orange might be more appropriate
      textColor = 'text-orange-100';
      icon = '⚠️';
      break;
    case DetectionStatus.PATH_CLEAR: // Updated from NO_EVENT
      bgColor = 'bg-green-600'; 
      textColor = 'text-green-100';
      icon = '✅';
      break;
    case DetectionStatus.ERROR:
    case DetectionStatus.CAMERA_ERROR:
    case DetectionStatus.API_KEY_MISSING:
      bgColor = 'bg-red-600';
      textColor = 'text-red-100';
      icon = '❗';
      break;
  }

  return (
    <div className={`p-4 my-4 rounded-lg shadow-md ${bgColor} ${textColor} transition-all duration-300 ease-in-out`}>
      <p className="text-lg font-semibold flex items-center">
        <span className="mr-2 text-xl">{icon}</span>
        {message}
      </p>
    </div>
  );
};

export default StatusIndicator;
