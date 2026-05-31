/**
 * ═══════════════════════════════════
 * 📄 FILE 37/42: mobile/components/ScanAnimation.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Scanning Laser Animation
 * Loop animates a futuristic laser line sweeping vertically across the viewport
 * to indicate real-world camera alignment and processing.
 */

import React, { useEffect, useRef } from 'react';
import { StyleSheet, View, Animated, Dimensions } from 'react-native';
import { COLORS, BORDER_RADIUS } from '../constants/theme';

interface ScanAnimationProps {
  active: boolean;
}

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

export const ScanAnimation: React.FC<ScanAnimationProps> = ({ active }) => {
  const sweepAnim = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(0.4)).current;

  useEffect(() => {
    let sweepLoop: Animated.CompositeAnimation | null = null;
    let pulseLoop: Animated.CompositeAnimation | null = null;

    if (active) {
      // Loop vertical sweeping
      sweepLoop = Animated.loop(
        Animated.sequence([
          Animated.timing(sweepAnim, {
            toValue: 1,
            duration: 2500,
            useNativeDriver: true,
          }),
          Animated.timing(sweepAnim, {
            toValue: 0,
            duration: 2500,
            useNativeDriver: true,
          }),
        ])
      );
      sweepLoop.start();

      // Loop breathing background pulse
      pulseLoop = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 0.8,
            duration: 1200,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 0.4,
            duration: 1200,
            useNativeDriver: true,
          }),
        ])
      );
      pulseLoop.start();
    } else {
      sweepAnim.setValue(0);
      pulseAnim.setValue(0.4);
    }

    return () => {
      if (sweepLoop) sweepLoop.stop();
      if (pulseLoop) pulseLoop.stop();
    };
  }, [active, sweepAnim, pulseAnim]);

  if (!active) return null;

  // Map progress to height offset
  const translateY = sweepAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [40, SCREEN_HEIGHT - 280], // scan area bounds
  });

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* Corner targeting brackets (Jarvis HUD style) */}
      <View style={styles.reticleCornerTopLeft} />
      <View style={styles.reticleCornerTopRight} />
      <View style={styles.reticleCornerBottomLeft} />
      <View style={styles.reticleCornerBottomRight} />

      {/* Pulsing center radar glow */}
      <Animated.View style={[styles.scannerGlow, { opacity: pulseAnim }]} />

      {/* Sweeping laser line with secondary neon glow layer underneath */}
      <Animated.View style={[styles.laserGlowLayer, { transform: [{ translateY }] }]} />
      <Animated.View style={[styles.laserLine, { transform: [{ translateY }] }]} />
    </View>
  );
};

const styles = StyleSheet.create({
  laserLine: {
    position: 'absolute',
    left: 24,
    right: 24,
    height: 3,
    backgroundColor: '#FFFFFF',
    borderRadius: BORDER_RADIUS.round,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.98,
    shadowRadius: 10,
    elevation: 10,
  },
  laserGlowLayer: {
    position: 'absolute',
    left: 18,
    right: 18,
    height: 12,
    marginTop: -4, // center relative to thin white core
    backgroundColor: 'rgba(139, 92, 246, 0.4)',
    borderRadius: BORDER_RADIUS.round,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.9,
    shadowRadius: 16,
  },
  scannerGlow: {
    position: 'absolute',
    top: 60,
    bottom: 200,
    left: 24,
    right: 24,
    borderRadius: BORDER_RADIUS.lg,
    borderWidth: 1.5,
    borderColor: 'rgba(139, 92, 246, 0.22)',
    backgroundColor: 'rgba(139, 92, 246, 0.02)',
  },
  
  // HUD corners (Professional cyber reticle)
  reticleCornerTopLeft: {
    position: 'absolute',
    top: 60,
    left: 24,
    width: 28,
    height: 28,
    borderTopWidth: 3.5,
    borderLeftWidth: 3.5,
    borderColor: COLORS.primary,
    borderTopLeftRadius: BORDER_RADIUS.md,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 6,
  },
  reticleCornerTopRight: {
    position: 'absolute',
    top: 60,
    right: 24,
    width: 28,
    height: 28,
    borderTopWidth: 3.5,
    borderRightWidth: 3.5,
    borderColor: COLORS.primary,
    borderTopRightRadius: BORDER_RADIUS.md,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 6,
  },
  reticleCornerBottomLeft: {
    position: 'absolute',
    bottom: 200,
    left: 24,
    width: 28,
    height: 28,
    borderBottomWidth: 3.5,
    borderLeftWidth: 3.5,
    borderColor: COLORS.primary,
    borderBottomLeftRadius: BORDER_RADIUS.md,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 6,
  },
  reticleCornerBottomRight: {
    position: 'absolute',
    bottom: 200,
    right: 24,
    width: 28,
    height: 28,
    borderBottomWidth: 3.5,
    borderRightWidth: 3.5,
    borderColor: COLORS.primary,
    borderBottomRightRadius: BORDER_RADIUS.md,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 6,
  },
});

export default ScanAnimation;
