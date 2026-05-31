/**
 * ═══════════════════════════════════
 * 📄 FILE 38/42: mobile/components/TTSPlayer.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Accessibility Audio Player
 * Displays floating playback controller bar with pause/stop actions,
 * read-along text snippets, and speech rate toggles.
 */

import React from 'react';
import { StyleSheet, View, Text, TouchableOpacity } from 'react-native';
import { Play, Pause, Square, Volume2, VolumeX } from 'lucide-react-native';
import { useAppStore } from '../store/useAppStore';
import VoiceService from '../services/voice';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from '../constants/theme';

interface TTSPlayerProps {
  text: string;
  playbackKey: string;
}

export const TTSPlayer: React.FC<TTSPlayerProps> = ({ text, playbackKey }) => {
  const store = useAppStore();
  const isPlaying = store.isPlayingAudio && store.activeAudioKey === playbackKey;

  if (!text || !store.ttsEnabled) return null;

  const handlePlayPause = () => {
    if (isPlaying) {
      VoiceService.stop();
    } else {
      VoiceService.speak(text, playbackKey);
    }
  };

  const handleStop = () => {
    VoiceService.stop();
  };

  // Toggle speed rate: +0% -> +25% -> -15% -> +0%
  const cycleSpeechRate = () => {
    let nextRate = '+0%';
    let rateLabel = 'Normal (1.0x)';
    
    if (store.speechRate === '+0%') {
      nextRate = '+25%';
      rateLabel = 'Fast (1.25x)';
    } else if (store.speechRate === '+25%') {
      nextRate = '-15%';
      rateLabel = 'Slow (0.85x)';
    }

    store.setSpeechRate(nextRate);
    VoiceService.speakGuidance(`Speed: ${rateLabel}`);
    
    // If currently playing the text, restart with new speed
    if (isPlaying) {
      setTimeout(() => {
        VoiceService.speak(text, playbackKey);
      }, 800);
    }
  };

  const getRateLabel = () => {
    if (store.speechRate === '+25%') return '1.25x';
    if (store.speechRate === '-15%') return '0.85x';
    return '1.0x';
  };

  return (
    <View style={styles.playerContainer}>
      {/* Read along snippet */}
      <View style={styles.textContainer}>
        <View style={styles.statusRow}>
          {isPlaying ? (
            <Volume2 color={COLORS.primary} size={13} strokeWidth={2.5} style={styles.statusIcon} />
          ) : (
            <VolumeX color={COLORS.textMuted} size={13} strokeWidth={2.5} style={styles.statusIcon} />
          )}
          <Text style={[styles.playingLabel, isPlaying && styles.playingLabelActive]}>
            {isPlaying ? 'ACTIVE AUDIO' : 'SPEECH READY'}
          </Text>
        </View>
        <Text style={styles.snippetText} numberOfLines={1}>
          {text}
        </Text>
      </View>

      {/* Media buttons */}
      <View style={styles.controlsRow}>
        {/* Speed button */}
        <TouchableOpacity
          style={styles.speedButton}
          onPress={cycleSpeechRate}
          accessibilityRole="button"
          accessibilityLabel={`Change speaking rate. Current is ${getRateLabel()}`}
        >
          <Text style={styles.speedText}>{getRateLabel()}</Text>
        </TouchableOpacity>

        {/* Play/Pause Button */}
        <TouchableOpacity
          style={[styles.playButton, isPlaying && styles.pauseButton]}
          onPress={handlePlayPause}
          accessibilityRole="button"
          accessibilityLabel={isPlaying ? 'Pause speech output' : 'Play speech output'}
        >
          {isPlaying ? (
            <Pause color="#FFFFFF" size={14} strokeWidth={3} fill="#FFFFFF" />
          ) : (
            <Play color="#FFFFFF" size={14} strokeWidth={3} fill="#FFFFFF" style={{ marginLeft: 1 }} />
          )}
        </TouchableOpacity>

        {/* Stop Button */}
        {isPlaying && (
          <TouchableOpacity
            style={styles.stopButton}
            onPress={handleStop}
            accessibilityRole="button"
            accessibilityLabel="Stop speech playback"
          >
            <Square color="#FFFFFF" size={10} strokeWidth={3} fill="#FFFFFF" />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  playerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(12, 8, 24, 0.65)',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 20,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    width: '100%',
    marginVertical: SPACING.xs,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.15,
    shadowRadius: 10,
    elevation: 4,
  },
  textContainer: {
    flex: 1,
    marginRight: SPACING.md,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  statusIcon: {
    marginRight: 6,
  },
  playingLabel: {
    fontSize: 9.5,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.textMuted,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
  },
  playingLabelActive: {
    color: COLORS.primary,
  },
  snippetText: {
    fontSize: 13,
    color: COLORS.textSecondary,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  controlsRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  speedButton: {
    backgroundColor: 'rgba(240, 86, 200, 0.08)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 10,
    borderWidth: 1.2,
    borderColor: 'rgba(240, 86, 200, 0.25)',
    marginRight: SPACING.sm,
    minWidth: 50,
    alignItems: 'center',
  },
  speedText: {
    color: COLORS.primary,
    fontSize: 10,
    fontWeight: TYPOGRAPHY.weight.heavy,
  },
  playButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 6,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.40,
    shadowRadius: 6,
    elevation: 4,
  },
  pauseButton: {
    backgroundColor: '#475569',
    shadowColor: '#475569',
  },
  stopButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: COLORS.error,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.error,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.40,
    shadowRadius: 6,
    elevation: 4,
  },
});

export default TTSPlayer;
