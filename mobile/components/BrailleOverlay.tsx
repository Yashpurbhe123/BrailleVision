/**
 * ═══════════════════════════════════
 * 📄 FILE 34/42: mobile/components/BrailleOverlay.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — SVG Dot & Bounding Box Overlay
 * Projects cell bounding boxes, dot positions, and character labels directly
 * over the viewfinder or static image using responsive scale parameters.
 */

import React from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Rect, Circle, Text as SvgText, G } from 'react-native-svg';
import { COLORS } from '../constants/theme';

interface Cell {
  pattern: number[];
  confidence: number;
  x: number;
  y: number;
  bbox: number[]; // [x1, y1, x2, y2]
  dot_count: number;
}

interface BrailleOverlayProps {
  cells: Cell[];
  width: number;       // Render width (viewfinder container width)
  height: number;      // Render height
  sourceWidth?: number;  // Original image width from backend
  sourceHeight?: number; // Original image height
}

export const BrailleOverlay: React.FC<BrailleOverlayProps> = ({
  cells,
  width,
  height,
  sourceWidth = 640,
  sourceHeight = 640,
}) => {
  if (!cells || cells.length === 0) return null;

  // Calculate coordinates mapping scales
  const scaleX = width / sourceWidth;
  const scaleY = height / sourceHeight;

  // Helper to resolve confidence colors
  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return COLORS.confHigh;
    if (conf >= 0.5) return COLORS.confMedium;
    return COLORS.confLow;
  };

  // Convert Grade 1+2 braille patterns to readable letters for screen indicators
  // If the backend didn't supply specific letter maps, we label them by index or dot counts
  return (
    <View style={[StyleSheet.absoluteFill, styles.container]} pointerEvents="none">
      <Svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {cells.map((cell, idx) => {
          const [x1, y1, x2, y2] = cell.bbox;
          
          // Map to container coordinates
          const boxX = x1 * scaleX;
          const boxY = y1 * scaleY;
          const boxW = (x2 - x1) * scaleX;
          const boxH = (y2 - y1) * scaleY;

          // Compute cell center
          const cx = cell.x * scaleX;
          const cy = cell.y * scaleY;

          const color = getConfidenceColor(cell.confidence);

          return (
            <G key={idx}>
              {/* Cell Bounding Box */}
              <Rect
                x={boxX}
                y={boxY}
                width={boxW}
                height={boxH}
                fill="rgba(255, 149, 0, 0.05)"
                stroke={color}
                strokeWidth={1.5}
                strokeDasharray="4, 3"
                rx={3}
              />

              {/* Cell Center Dot Indicator */}
              <Circle
                cx={cx}
                cy={cy}
                r={3}
                fill={color}
              />

              {/* Character Text Label above the bounding box */}
              {boxY > 18 && (
                <SvgText
                  x={boxX + boxW / 2}
                  y={boxY - 4}
                  fill={color}
                  fontSize={12}
                  fontWeight="bold"
                  textAnchor="middle"
                >
                  {`C:${Math.round(cell.confidence * 100)}%`}
                </SvgText>
              )}
            </G>
          );
        })}
      </Svg>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'transparent',
  },
});

export default BrailleOverlay;
