/**
 * ═══════════════════════════════════
 * 📄 FILE 30/42: mobile/app/(tabs)/scanner.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Live Camera Viewfinder & Detector
 * Mounts Expo CameraView, manages real-time detection loops, draws SVG HUD overlays,
 * and handles tactile snapshot actions with spoken guidance cues.
 */

import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator, Dimensions } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useFocusEffect } from 'expo-router';
import { Zap, ZapOff, BrainCircuit, Sparkles, Camera, AlertTriangle, ShieldAlert, Info } from 'lucide-react-native';

import { useAppStore } from '../../store/useAppStore';
import useBrailleScanner from '../../hooks/useBrailleScanner';
import { BrailleOverlay } from '../../components/BrailleOverlay';
import { GuidanceBanner } from '../../components/GuidanceBanner';
import { ScanAnimation } from '../../components/ScanAnimation';
import { TTSPlayer } from '../../components/TTSPlayer';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from '../../constants/theme';
import VoiceService from '../../services/voice';
import { useScannerVoiceCommands } from '../../hooks/useVoiceCommands';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function ScannerScreen() {
  const store = useAppStore();
  const cameraRef = useRef<any>(null);
  
  // Camera permission hooks
  const [permission, requestPermission] = useCameraPermissions();

  // Local settings
  const [flashlight, setFlashlight] = useState<boolean>(false);
  const [useCorrection, setUseCorrection] = useState<boolean>(true);
  const [isFocused, setIsFocused] = useState<boolean>(false);

  // Hook integrating the scanner logic
  const {
    isScanning,
    isLoading,
    currentResult,
    startLiveScanner,
    stopLiveScanner,
    captureFullImage,
  } = useBrailleScanner(cameraRef);

  // ── Voice commands for blind users ──
  useScannerVoiceCommands({
    onCapture: () => { if (!isLoading) captureFullImage({ correct: useCorrection }); },
    onFlashlight: (on) => {
      setFlashlight(on);
    },
    onReadResult: () => {
      const text = currentResult?.translatedText || currentResult?.correctedText;
      if (text) VoiceService.speak(text, 'live-scanner');
    },
    onStopAudio: () => VoiceService.stop(),
  });

  // Stop scanner when tab is blurred, start when focused
  useFocusEffect(
    React.useCallback(() => {
      setIsFocused(true);
      // Speak on screen focus for assistive guidance
      VoiceService.speakGuidance("Scanner active. Say scan to capture, or say help for all commands.");
      
      // Auto start live scanner
      if (permission && permission.granted) {
        startLiveScanner();
      }

      return () => {
        setIsFocused(false);
        stopLiveScanner();
      };
    }, [permission, startLiveScanner, stopLiveScanner])
  );

  if (!permission) {
    // Camera permissions are still loading
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    );
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.permissionContainer}>
        <View style={styles.permissionIconWrapper}>
          <ShieldAlert color={COLORS.error} size={54} strokeWidth={1.5} />
        </View>
        <Text style={styles.permissionText}>
          BrailleVision AI requires camera access to detect, segment, and decode embossed Braille dots.
        </Text>
        <TouchableOpacity style={styles.permissionBtn} onPress={requestPermission}>
          <Text style={styles.permissionBtnText}>GRANT CAMERA ACCESS</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Viewfinder Viewport */}
      <View style={styles.cameraContainer}>
        {isFocused && (
          <CameraView
            style={StyleSheet.absoluteFill}
            ref={cameraRef}
            facing="back"
            enableTorch={flashlight}
          />
        )}

        {/* Scan Sweeping Laser HUD Overlay */}
        <ScanAnimation active={isScanning && !isLoading} />

        {/* Visual Cell Bounding Boxes */}
        {currentResult && currentResult.cells && (
          <BrailleOverlay
            cells={currentResult.cells}
            width={SCREEN_WIDTH}
            height={SCREEN_WIDTH * 1.33} // standard 4:3 camera scale ratio
            sourceWidth={640}
            sourceHeight={640}
          />
        )}

        {/* Left Telemetry HUD Widget */}
        <View style={styles.leftTelemetry} pointerEvents="none">
          <View style={styles.telemetryItem}>
            <Text style={styles.telemetryLabel}>SYS MODE</Text>
            <Text style={styles.telemetryValue}>REALTIME</Text>
          </View>
          <View style={styles.telemetryItem}>
            <Text style={styles.telemetryLabel}>FPS</Text>
            <Text style={styles.telemetryValue}>30.0</Text>
          </View>
          <View style={styles.telemetryItem}>
            <Text style={styles.telemetryLabel}>TILT</Text>
            <Text style={styles.telemetryValue}>0.0°</Text>
          </View>
          <View style={styles.telemetryItem}>
            <Text style={styles.telemetryLabel}>FOCUS</Text>
            <Text style={styles.telemetryValue}>AUTO</Text>
          </View>
        </View>

        {/* Right Telemetry HUD Widget */}
        <View style={styles.rightTelemetry} pointerEvents="none">
          <View style={styles.telemetryItem}>
            <Text style={styles.telemetryLabel}>LIGHT</Text>
            <Text style={styles.telemetryValue}>DYNAMIC</Text>
          </View>
          <View style={styles.telemetryItem}>
            <Text style={styles.telemetryLabel}>SIDE</Text>
            <Text style={[styles.telemetryValue, { color: COLORS.accent }]}>
              {currentResult?.sideDetected?.toUpperCase() || 'AUTO'}
            </Text>
          </View>
          <View style={styles.telemetryItem}>
            <Text style={styles.telemetryLabel}>ENGINE</Text>
            <Text style={styles.telemetryValue}>NEURAL</Text>
          </View>
          <View style={styles.telemetryItem}>
            <Text style={styles.telemetryLabel}>ACCURACY</Text>
            <Text style={[styles.telemetryValue, { color: currentResult?.detectionQuality === 'good' ? COLORS.success : COLORS.warning }]}>
              {currentResult?.detectionQuality?.toUpperCase() || 'HIGH'}
            </Text>
          </View>
        </View>

        {/* Central Cockpit Targeting Reticle */}
        <View style={styles.centerReticle} pointerEvents="none">
          <View style={styles.centerReticleRing} />
          <View style={styles.centerReticleDot} />
          <Text style={styles.reticleLabel}>TARGET FIELD</Text>
        </View>

        {/* Guidance Directions Overlay Banner */}
        {currentResult && (
          <View style={styles.guidanceOverlay}>
            <GuidanceBanner
              message={currentResult.guidance}
              quality={currentResult.detectionQuality}
            />
          </View>
        )}

        {/* Top Control HUD Toggles */}
        <View style={styles.hudContainer}>
          {/* Flashlight toggle */}
          <TouchableOpacity
            style={[styles.hudButton, flashlight && styles.hudButtonActive]}
            onPress={() => {
              setFlashlight(!flashlight);
              VoiceService.speakGuidance(`Flashlight ${!flashlight ? 'on' : 'off'}`);
            }}
            accessibilityRole="button"
            accessibilityLabel="Toggle camera flashlight"
          >
            <Zap color={flashlight ? COLORS.accent : COLORS.textSecondary} size={18} strokeWidth={2.5} />
          </TouchableOpacity>

          {/* AI Correction toggle */}
          <TouchableOpacity
            style={[styles.hudButton, useCorrection && styles.hudButtonActive]}
            onPress={() => {
              setUseCorrection(!useCorrection);
              VoiceService.speakGuidance(`Grammar correction ${!useCorrection ? 'enabled' : 'disabled'}`);
            }}
            accessibilityRole="button"
            accessibilityLabel="Toggle AI text grammar correction"
          >
            <BrainCircuit color={useCorrection ? COLORS.primary : COLORS.textSecondary} size={18} strokeWidth={2.5} />
          </TouchableOpacity>
        </View>

        {/* Loader overlay during full capture process */}
        {isLoading && (
          <View style={[StyleSheet.absoluteFill, styles.loadingOverlay]}>
            <ActivityIndicator size="large" color={COLORS.primary} />
            <Text style={styles.loadingText}>Analyzing Braille dots...</Text>
          </View>
        )}
      </View>

      {/* Scanned result text card & speech audio control */}
      <View style={styles.resultContainer}>
        {currentResult && (currentResult.translatedText || currentResult.correctedText) ? (
          <View style={styles.resultContent}>
            <View style={styles.resultHeader}>
              <View style={styles.resultHeaderLeft}>
                <Sparkles color={COLORS.primary} size={14} strokeWidth={2.5} style={{ marginRight: 6 }} />
                <Text style={styles.resultTitle}>DECODED COCKPIT OUTPUT</Text>
              </View>
              {currentResult.avgConfidence > 0 && (
                <Text style={styles.confidenceText}>
                  Match {Math.round(currentResult.avgConfidence * 100)}%
                </Text>
              )}
            </View>
            <Text style={styles.resultText} numberOfLines={2}>
              {currentResult.translatedText || currentResult.correctedText}
            </Text>

            {/* Real-time Processing Stats Row */}
            <View style={styles.statsRow}>
              <View style={styles.statBadge}>
                <Text style={styles.statLabel}>DOTS</Text>
                <Text style={styles.statValue}>
                  {currentResult.dotCount ?? 0}
                </Text>
              </View>
              <View style={styles.statBadge}>
                <Text style={styles.statLabel}>CELLS</Text>
                <Text style={styles.statValue}>
                  {currentResult.cellCount ?? 0}
                </Text>
              </View>
              <View style={styles.statBadge}>
                <Text style={styles.statLabel}>CONFIDENCE</Text>
                <Text style={styles.statValue}>
                  {Math.round((currentResult.avgConfidence ?? 0) * 100)}%
                </Text>
              </View>
              {currentResult.processingTimeMs !== undefined && (
                <View style={styles.statBadge}>
                  <Text style={styles.statLabel}>LATENCY</Text>
                  <Text style={styles.statValue}>
                    {Math.round(currentResult.processingTimeMs)}ms
                  </Text>
                </View>
              )}
            </View>
            
            {/* Audio player bar */}
            <TTSPlayer
              text={currentResult.translatedText || currentResult.correctedText}
              playbackKey="live-scanner"
            />
          </View>
        ) : (
          <View style={styles.idleResultContent}>
            <Info color={COLORS.primary} size={18} strokeWidth={2.5} style={{ marginBottom: 4 }} />
            <Text style={styles.idleText}>
              Point camera at Braille text & double-tap to read.
            </Text>
          </View>
        )}

        {/* Master Capture Button (Hyper Cyber-Reactor Core) */}
        <View style={styles.captureBtnWrapper}>
          <TouchableOpacity
            style={[styles.captureBtn, store.highContrast && styles.captureBtnHighContrast]}
            onPress={() => captureFullImage({ correct: useCorrection })}
            accessibilityRole="button"
            accessibilityLabel="Double tap to scan high resolution image and read output text"
            accessibilityHint="Fires the hybrid AI decoding and spelling corrector pipeline"
            disabled={isLoading}
          >
            <View style={styles.captureBtnInner}>
              <Camera color="#FFFFFF" size={24} strokeWidth={2.5} />
            </View>
          </TouchableOpacity>
          <View style={styles.captureBtnPulseRing} />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  centerContainer: {
    flex: 1,
    backgroundColor: COLORS.background,
    justifyContent: 'center',
    alignItems: 'center',
  },
  permissionContainer: {
    flex: 1,
    backgroundColor: COLORS.background,
    justifyContent: 'center',
    alignItems: 'center',
    padding: SPACING.xl,
  },
  permissionIconWrapper: {
    width: 110,
    height: 110,
    borderRadius: 55,
    backgroundColor: 'rgba(240, 86, 200, 0.05)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.lg,
    borderWidth: 2,
    borderColor: 'rgba(240, 86, 200, 0.2)',
  },
  permissionText: {
    fontSize: 14.5,
    color: COLORS.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: SPACING.xl,
    paddingHorizontal: SPACING.md,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  permissionBtn: {
    backgroundColor: COLORS.primary,
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 20,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 6,
  },
  permissionBtnText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 1.0,
  },
  cameraContainer: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    overflow: 'hidden',
  },
  hudContainer: {
    position: 'absolute',
    top: 64,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    zIndex: 10,
  },
  hudButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(12, 8, 24, 0.88)',
    borderColor: 'rgba(240, 86, 200, 0.35)',
    borderWidth: 1.8,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  hudButtonActive: {
    borderColor: COLORS.accent,
    backgroundColor: 'rgba(6, 182, 212, 0.25)',
    shadowColor: COLORS.accent,
    shadowOpacity: 0.6,
    shadowRadius: 12,
  },
  guidanceOverlay: {
    position: 'absolute',
    top: 130,
    width: '100%',
  },
  loadingOverlay: {
    backgroundColor: 'rgba(4, 2, 9, 0.90)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: COLORS.text,
    marginTop: SPACING.md,
    fontSize: 15,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
  },
  
  // God-Elite HUD Telemetry Styling (Asymmetric curved brackets)
  leftTelemetry: {
    position: 'absolute',
    left: 15,
    top: 150,
    backgroundColor: 'rgba(12, 8, 24, 0.76)',
    borderColor: COLORS.border,
    borderWidth: 1.5,
    padding: 10,
    borderRadius: 16,
    zIndex: 10,
    width: 86,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
  },
  rightTelemetry: {
    position: 'absolute',
    right: 15,
    top: 150,
    backgroundColor: 'rgba(12, 8, 24, 0.76)',
    borderColor: COLORS.borderCyan,
    borderWidth: 1.5,
    padding: 10,
    borderRadius: 16,
    zIndex: 10,
    width: 86,
    shadowColor: COLORS.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
  },
  telemetryItem: {
    marginBottom: 10,
  },
  telemetryLabel: {
    fontSize: 8,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.textSecondary,
    letterSpacing: 1.0,
    textTransform: 'uppercase',
  },
  telemetryValue: {
    fontSize: 11,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.accent,
  },
  centerReticle: {
    position: 'absolute',
    top: '32%',
    left: '50%',
    marginLeft: -60,
    width: 120,
    height: 120,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 3,
  },
  centerReticleRing: {
    width: 110,
    height: 110,
    borderRadius: 55,
    borderWidth: 2,
    borderColor: 'rgba(6, 182, 212, 0.45)',
    borderStyle: 'dashed',
  },
  centerReticleDot: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.primary,
    shadowColor: COLORS.primary,
    shadowOpacity: 0.9,
    shadowRadius: 6,
  },
  reticleLabel: {
    position: 'absolute',
    bottom: -18,
    fontSize: 8,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.accent,
    letterSpacing: 2.0,
    textTransform: 'uppercase',
  },

  // Frosted Neon Card Styling (Glass cockpit)
  resultContainer: {
    position: 'absolute',
    bottom: 110,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(18, 12, 32, 0.88)', // Deep glass obsidian
    paddingTop: 18,
    paddingBottom: 18,
    paddingHorizontal: 22,
    borderRadius: 28,
    borderWidth: 2,
    borderColor: COLORS.border,
    alignItems: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.50,
    shadowRadius: 24,
    elevation: 12,
  },
  resultContent: {
    width: '100%',
    marginBottom: 12,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  resultHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  resultTitle: {
    fontSize: 10,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.primary,
    letterSpacing: 1.8,
  },
  confidenceText: {
    fontSize: 11,
    color: COLORS.success,
    fontWeight: TYPOGRAPHY.weight.heavy,
    backgroundColor: 'rgba(16, 185, 129, 0.12)',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: 'rgba(16, 185, 129, 0.3)',
  },
  resultText: {
    fontSize: 19,
    fontWeight: TYPOGRAPHY.weight.bold,
    color: COLORS.text,
    lineHeight: 26,
    marginBottom: 10,
    letterSpacing: -0.2,
  },
  idleResultContent: {
    width: '100%',
    height: 84,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
    backgroundColor: 'rgba(5, 3, 10, 0.65)',
    borderRadius: 20,
    paddingHorizontal: 20,
    borderWidth: 1.5,
    borderColor: 'rgba(255, 255, 255, 0.04)',
  },
  idleText: {
    fontSize: 13,
    color: COLORS.textSecondary,
    textAlign: 'center',
    lineHeight: 20,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  captureBtnWrapper: {
    position: 'relative',
    alignItems: 'center',
    justifyContent: 'center',
    width: 96,
    height: 96,
    marginTop: 6,
  },
  captureBtn: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: COLORS.accent, // Neon Cyan
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2,
    shadowColor: COLORS.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.85,
    shadowRadius: 16,
    elevation: 8,
  },
  captureBtnHighContrast: {
    backgroundColor: '#FFFF00',
    shadowColor: '#FFFF00',
  },
  captureBtnInner: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: 'transparent',
    alignItems: 'center',
    justifyContent: 'center',
  },
  captureBtnPulseRing: {
    position: 'absolute',
    width: 92,
    height: 92,
    borderRadius: 46,
    borderWidth: 2,
    borderColor: 'rgba(6, 182, 212, 0.55)',
    borderStyle: 'dashed',
    zIndex: 1,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
    marginBottom: 10,
    backgroundColor: 'rgba(240, 86, 200, 0.06)',
    borderRadius: 18,
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderWidth: 1.5,
    borderColor: COLORS.border,
  },
  statBadge: {
    alignItems: 'center',
    flex: 1,
  },
  statLabel: {
    fontSize: 9,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.textSecondary,
    letterSpacing: 0.8,
    marginBottom: 4,
    textTransform: 'uppercase',
  },
  statValue: {
    fontSize: 14,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.accent, // Holographic cyan metrics
  },
});

