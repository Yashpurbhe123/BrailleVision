/**
 * ═══════════════════════════════════
 * 📄 FILE 36/42: mobile/components/ConfidenceDisplay.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Confidence Indicator & Letter Highlighter
 * Renders decoded Braille text, highlighting individual letters in colors
 * (Green = High confidence, Yellow = Medium, Red = Low) for verification.
 */

import React from 'react';
import { StyleSheet, View, Text } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../constants/theme';

interface ConfidenceDisplayProps {
  text: string;
  confidences: number[];
  avgConfidence: number;
}

export const ConfidenceDisplay: React.FC<ConfidenceDisplayProps> = ({
  text,
  confidences,
  avgConfidence,
}) => {
  if (!text) return null;

  // Helper to color code confidence levels
  const getLetterColor = (conf: number) => {
    if (conf >= 0.8) return COLORS.confHigh;
    if (conf >= 0.5) return COLORS.confMedium;
    return COLORS.confLow;
  };

  const getLetterBg = (conf: number) => {
    if (conf >= 0.8) return 'rgba(52, 199, 89, 0.1)';
    if (conf >= 0.5) return 'rgba(255, 204, 0, 0.1)';
    return 'rgba(255, 59, 48, 0.1)';
  };

  // Split text into characters and pair with confidence
  const charArray = text.split('');
  
  return (
    <View style={styles.container}>
      {/* Header Info */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Detection Reliability</Text>
        <View
          style={[
            styles.badge,
            {
              backgroundColor: getLetterBg(avgConfidence),
              borderColor: getLetterColor(avgConfidence),
            },
          ]}
          accessibilityLabel={`Average detection confidence is ${Math.round(avgConfidence * 100)} percent`}
        >
          <Text style={[styles.badgeText, { color: getLetterColor(avgConfidence) }]}>
            {Math.round(avgConfidence * 100)}% Match
          </Text>
        </View>
      </View>

      {/* Colour-coded character grid */}
      <View style={styles.textGrid}>
        {charArray.map((char, index) => {
          // Align character index to confidence index
          const conf = confidences[index] !== undefined ? confidences[index] : 0.9;
          const isSpace = char === ' ' || char === '\n';

          if (isSpace) {
            return (
              <View key={index} style={styles.spaceBlock}>
                <Text style={styles.spaceText}> </Text>
              </View>
            );
          }

          return (
            <View
              key={index}
              style={[
                styles.charBlock,
                {
                  backgroundColor: getLetterBg(conf),
                  borderColor: getLetterColor(conf),
                },
              ]}
              accessibilityLabel={`Letter ${char}, confidence ${Math.round(conf * 100)} percent`}
            >
              <Text style={[styles.charText, { color: getLetterColor(conf) }]}>
                {char}
              </Text>
            </View>
          );
        })}
      </View>

      {/* Legend */}
      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: COLORS.confHigh }]} />
          <Text style={styles.legendLabel}>Reliable (≥80%)</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: COLORS.confMedium }]} />
          <Text style={styles.legendLabel}>Review (50-79%)</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: COLORS.confLow }]} />
          <Text style={styles.legendLabel}>Uncertain (&lt;50%)</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'rgba(18, 12, 32, 0.65)',
    padding: SPACING.md,
    borderRadius: 24,
    width: '100%',
    marginVertical: SPACING.sm,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.20,
    shadowRadius: 16,
    elevation: 5,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACING.md,
  },
  headerTitle: {
    fontSize: 11,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.textSecondary,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
  },
  badge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    borderWidth: 1.5,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 6,
    elevation: 3,
  },
  badgeText: {
    fontSize: 10,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 1.0,
    textTransform: 'uppercase',
  },
  textGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: SPACING.md,
    backgroundColor: 'rgba(5, 3, 10, 0.85)',
    padding: 14,
    borderRadius: 20,
    borderWidth: 1.5,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  charBlock: {
    paddingHorizontal: 8,
    borderRadius: 12,
    borderWidth: 2,
    marginRight: 10,
    marginBottom: 10,
    minWidth: 36,
    height: 42,
    alignItems: 'center',
    justifyContent: 'center',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.45,
    shadowRadius: 8,
    elevation: 6,
  },
  charText: {
    fontSize: 18,
    fontWeight: TYPOGRAPHY.weight.heavy,
  },
  spaceBlock: {
    width: 22,
    height: 42,
    justifyContent: 'center',
  },
  spaceText: {
    fontSize: TYPOGRAPHY.size.lg,
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: 14,
    borderTopWidth: 1.5,
    borderTopColor: 'rgba(255, 255, 255, 0.05)',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 6,
  },
  legendLabel: {
    fontSize: 9.5,
    color: COLORS.textSecondary,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
});

export default ConfidenceDisplay;
