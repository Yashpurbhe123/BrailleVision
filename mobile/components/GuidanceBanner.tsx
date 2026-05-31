/**
 * ═══════════════════════════════════
 * 📄 FILE 35/42: mobile/components/GuidanceBanner.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Camera Guidance Banner
 * Renders real-time text instructions (e.g. "Move right", "Hold steady")
 * with visual border alerts color-coded by detection quality.
 */

import React, { useEffect, useRef } from 'react';
import { StyleSheet, View, Text, Animated } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from '../constants/theme';

interface GuidanceBannerProps {
  message: string;
  quality?: string; // 'good' | 'low' | 'poor' | 'unknown'
}

export const GuidanceBanner: React.FC<GuidanceBannerProps> = ({ message, quality = 'unknown' }) => {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(-20)).current;

  useEffect(() => {
    // Reset and trigger slide/fade animations when a new instruction arrives
    fadeAnim.setValue(0);
    slideAnim.setValue(-15);

    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 350,
        useNativeDriver: true,
      }),
    ]).start();
  }, [message]);

  if (!message) return null;

  // Resolve shadow glow color based on detection quality
  const getQualityShadow = () => {
    switch (quality) {
      case 'good':
        return COLORS.success;
      case 'low':
        return COLORS.warning;
      case 'poor':
        return COLORS.error;
      default:
        return COLORS.primary;
    }
  };

  // Resolve border and status dot color based on detection quality
  const getQualityColor = () => {
    switch (quality) {
      case 'good':
        return COLORS.success;
      case 'low':
        return COLORS.warning;
      case 'poor':
        return COLORS.error;
      default:
        return COLORS.primary;
    }
  };

  return (
    <Animated.View
      style={[
        styles.banner,
        {
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }],
          borderColor: getQualityColor(),
          shadowColor: getQualityShadow(),
        },
      ]}
      accessibilityRole="alert"
      accessibilityLabel={`Guidance: ${message}`}
    >
      <View style={styles.indicatorRow}>
        <View style={[styles.statusDot, { backgroundColor: getQualityColor() }]} />
        <Text style={styles.qualityText}>
          {quality.toUpperCase()} ACCURACY
        </Text>
      </View>
      <Text style={styles.messageText}>{message}</Text>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  banner: {
    backgroundColor: 'rgba(12, 8, 24, 0.94)',
    borderWidth: 2,
    borderRadius: 24,
    paddingVertical: 14,
    paddingHorizontal: 20,
    marginHorizontal: SPACING.lg,
    marginVertical: SPACING.xs,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.45,
    shadowRadius: 12,
    elevation: 8,
  },
  indicatorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  qualityText: {
    fontSize: 10.5,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.textSecondary,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
  },
  messageText: {
    fontSize: 14.5,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.text,
    lineHeight: 22,
  },
});

export default GuidanceBanner;
