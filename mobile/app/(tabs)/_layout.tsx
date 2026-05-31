/**
 * ═══════════════════════════════════
 * 📄 FILE 39c/42: mobile/app/(tabs)/_layout.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Tab Navigation Layout
 * Implements accessible, high-contrast visual navigation for the four main screens
 * (Scanner, Upload, History, and Settings) using lucide-react-native icons.
 * Also mounts the global Voice Command Engine and floating mic indicator.
 */

import React, { useEffect, useState, useRef } from 'react';
import { Tabs, useRouter } from 'expo-router';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  Animated,
  Platform,
} from 'react-native';
import { Scan, Upload, History, Settings, Mic, MicOff } from 'lucide-react-native';
import { useAppStore } from '../../store/useAppStore';
import { COLORS, SPACING, BORDER_RADIUS } from '../../constants/theme';
import VoiceCommandEngine from '../../services/voiceCommands';
import VoiceService from '../../services/voice';

export default function TabLayout() {
  const highContrast = useAppStore((state) => state.highContrast);
  const router = useRouter();

  // ── Voice UI State ──
  const [isListening, setIsListening] = useState(false);
  const [lastCommand, setLastCommand] = useState<string | undefined>();
  const [showBadge, setShowBadge] = useState(false);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const badgeTimer = useRef<any>(null);

  // Accessible colors depending on contrast settings
  const activeColor = highContrast ? '#FFFF00' : COLORS.primary;
  const inactiveColor = highContrast ? '#FFFFFF' : COLORS.textSecondary;
  const tabBgColor = highContrast ? '#000000' : COLORS.surface;
  const borderTopColor = highContrast ? '#FFFFFF' : COLORS.border;

  // ── Init Voice Command Engine ──
  useEffect(() => {
    if (Platform.OS !== 'web') return;

    // Wire navigation callback
    VoiceCommandEngine.setNavigateCallback((tab) => {
      router.push(`/(tabs)/${tab}` as any);
    });

    // Wire stop-audio callback (global)
    VoiceCommandEngine.setStopAudioCallback(() => {
      VoiceService.stop();
    });

    // Subscribe to state changes for UI indicator
    const onStateChange = (listening: boolean, cmd?: string) => {
      setIsListening(listening);
      if (cmd) {
        setLastCommand(cmd);
        setShowBadge(true);

        // Auto-hide badge after 2.5s
        if (badgeTimer.current) clearTimeout(badgeTimer.current);
        badgeTimer.current = setTimeout(() => {
          setShowBadge(false);
        }, 2500);
      }
    };

    VoiceCommandEngine.addStateListener(onStateChange);

    // Auto-start voice commands
    if (VoiceCommandEngine.supported) {
      VoiceCommandEngine.start();
    }

    return () => {
      VoiceCommandEngine.removeStateListener(onStateChange);
      VoiceCommandEngine.stop();
    };
  }, []);

  // ── Pulse animation while listening ──
  useEffect(() => {
    let loop: Animated.CompositeAnimation | null = null;
    if (isListening) {
      loop = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.25, duration: 800, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1.0, duration: 800, useNativeDriver: true }),
        ])
      );
      loop.start();
    } else {
      pulseAnim.setValue(1);
    }
    return () => { loop?.stop(); };
  }, [isListening]);

  // ── Badge fade in/out ──
  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: showBadge ? 1 : 0,
      duration: 250,
      useNativeDriver: true,
    }).start();
  }, [showBadge]);

  return (
    <View style={{ flex: 1 }}>
      <Tabs
        screenOptions={{
          headerShown: false,
          tabBarActiveTintColor: activeColor,
          tabBarInactiveTintColor: inactiveColor,
          tabBarStyle: {
            position: 'absolute',
            bottom: 20,
            left: 20,
            right: 20,
            backgroundColor: 'rgba(12, 8, 24, 0.82)', // High-fidelity deep glass obsidian
            borderColor: borderTopColor,
            borderWidth: highContrast ? 3.0 : 1.5,
            borderRadius: 28,
            height: 76,
            paddingBottom: Platform.OS === 'ios' ? 15 : 12,
            paddingTop: 10,
            elevation: 16,
            shadowColor: highContrast ? '#FFFFFF' : COLORS.primary,
            shadowOffset: { width: 0, height: 10 },
            shadowOpacity: highContrast ? 0.45 : 0.35,
            shadowRadius: 20,
            overflow: 'hidden',
          },
          tabBarLabelStyle: {
            fontSize: 9.5,
            fontWeight: '900',
            marginTop: 4,
            letterSpacing: 0.8,
            textTransform: 'uppercase',
          },
        }}
      >
        <Tabs.Screen
          name="scanner"
          options={{
            title: 'Scanner',
            tabBarLabel: 'Scanner',
            tabBarIcon: ({ color, size }) => (
              <Scan color={color} size={size + 1} strokeWidth={2.4} />
            ),
            tabBarAccessibilityLabel: 'Braille Scanner, capture dots using camera',
          }}
        />
        <Tabs.Screen
          name="upload"
          options={{
            title: 'Upload',
            tabBarLabel: 'Upload',
            tabBarIcon: ({ color, size }) => (
              <Upload color={color} size={size + 1} strokeWidth={2.4} />
            ),
            tabBarAccessibilityLabel: 'Upload file, pick image to decode',
          }}
        />
        <Tabs.Screen
          name="history"
          options={{
            title: 'History',
            tabBarLabel: 'History',
            tabBarIcon: ({ color, size }) => (
              <History color={color} size={size + 1} strokeWidth={2.4} />
            ),
            tabBarAccessibilityLabel: 'Scan History, view previous translations and stats',
          }}
        />
        <Tabs.Screen
          name="settings"
          options={{
            title: 'Settings',
            tabBarLabel: 'Settings',
            tabBarIcon: ({ color, size }) => (
              <Settings color={color} size={size + 1} strokeWidth={2.4} />
            ),
            tabBarAccessibilityLabel: 'Settings, configure speech rates and contrast toggles',
          }}
        />
      </Tabs>

      {/* ── Floating Mic Button + Last Command Badge ── */}
      {Platform.OS === 'web' && VoiceCommandEngine.supported && (
        <View style={styles.micWrapper} pointerEvents="box-none">
          {/* Last command badge */}
          <Animated.View style={[styles.commandBadge, { opacity: fadeAnim }]}>
            <Text style={styles.commandBadgeText} numberOfLines={1}>
              🎙️ "{lastCommand}"
            </Text>
          </Animated.View>

          {/* Mic button */}
          <TouchableOpacity
            style={styles.micBtnContainer}
            onPress={() => VoiceCommandEngine.toggle()}
            accessibilityRole="button"
            accessibilityLabel={isListening ? 'Tap to stop voice commands' : 'Tap to start voice commands'}
            accessibilityHint="Say help to hear all available commands"
            id="voice-command-mic-btn"
          >
            <Animated.View
              style={[
                styles.micPulseRing,
                {
                  transform: [{ scale: pulseAnim }],
                  opacity: isListening ? 0.35 : 0,
                },
              ]}
            />
            <View style={[styles.micBtn, isListening && styles.micBtnActive]}>
              {isListening ? (
                <Mic color="#030408" size={20} strokeWidth={2.8} />
              ) : (
                <MicOff color={COLORS.textSecondary} size={20} strokeWidth={2.4} />
              )}
            </View>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  micWrapper: {
    position: 'absolute',
    bottom: 110, // floats elegantly above the modernized tab bar
    right: 24,
    alignItems: 'flex-end',
    zIndex: 999,
  },
  commandBadge: {
    backgroundColor: 'rgba(12, 8, 24, 0.96)',
    borderColor: COLORS.border,
    borderWidth: 1.5,
    borderRadius: 18,
    paddingHorizontal: 16,
    paddingVertical: 10,
    marginBottom: 10,
    maxWidth: 240,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 10,
    elevation: 6,
  },
  commandBadgeText: {
    color: COLORS.text,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 0.5,
  },
  micBtnContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    width: 60,
    height: 60,
  },
  micPulseRing: {
    position: 'absolute',
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: COLORS.primary,
  },
  micBtn: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(28, 16, 48, 0.85)',
    borderWidth: 2,
    borderColor: 'rgba(240, 86, 200, 0.35)',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.45,
    shadowRadius: 12,
    elevation: 8,
  },
  micBtnActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.9,
    shadowRadius: 16,
  },
});
