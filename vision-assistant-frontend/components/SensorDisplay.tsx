import React from 'react';
import { SensorPackage } from '../types';

interface SensorDisplayProps {
  sensorData?: SensorPackage | null;
  wsConnected?: boolean;
  className?: string;
}

const SensorDisplay: React.FC<SensorDisplayProps> = ({ sensorData, wsConnected, className = '' }) => {
  if (!sensorData) {
    return (
      <div className={`sensor-display ${className}`}>
        <h3>Sensor Data</h3>
        <p>WebSocket: {wsConnected ? 'Connected' : 'Disconnected'}</p>
        <p>No sensor data available</p>
      </div>
    );
  }

  return (
    <div className={`sensor-display ${className}`}>
      <h3>Sensor Data</h3>
      <p>WebSocket: {wsConnected ? 'Connected' : 'Disconnected'}</p>
      <div className="sensor-grid">
        {sensorData.accelerometer && (
          <div className="sensor-item">
            <h4>Accelerometer</h4>
            <p>X: {sensorData.accelerometer.x?.toFixed(2)}</p>
            <p>Y: {sensorData.accelerometer.y?.toFixed(2)}</p>
            <p>Z: {sensorData.accelerometer.z?.toFixed(2)}</p>
            <p>Pitch: {sensorData.accelerometer.pitch?.toFixed(2)}°</p>
            <p>Roll: {sensorData.accelerometer.roll?.toFixed(2)}°</p>
            <p>Total Accel: {sensorData.accelerometer.total_acceleration?.toFixed(2)}</p>
            <p>Moving: {sensorData.accelerometer.is_moving ? 'Yes' : 'No'}</p>
            <p>Status: {sensorData.accelerometer.motion_status}</p>
          </div>
        )}
        
        {sensorData.ultrasonic && (
          <div className="sensor-item">
            <h4>Ultrasonic</h4>
            <p>Distance: {sensorData.ultrasonic.distance_cm?.toFixed(1)} cm</p>
            <p>Distance: {sensorData.ultrasonic.distance_m?.toFixed(2)} m</p>
          </div>
        )}
        
        {sensorData.camera && (
          <div className="sensor-item">
            <h4>Camera</h4>
            <p>Resolution: {sensorData.camera.resolution}</p>
            <p>Format: {sensorData.camera.format}</p>
            <p>Timestamp: {new Date(sensorData.camera.timestamp).toLocaleTimeString()}</p>
          </div>
        )}
        
        <div className="sensor-item">
          <h4>General</h4>
          <p>Last Update: {new Date(sensorData.timestamp).toLocaleTimeString()}</p>
        </div>
      </div>
    </div>
  );
};

export default SensorDisplay;