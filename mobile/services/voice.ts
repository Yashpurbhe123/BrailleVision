/**
 * ═══════════════════════════════════
 * 📄 FILE 27/42: mobile/services/voice.ts
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Audio Voice Service
 * Integrates Expo Speech (offline local engine) and server-side neural
 * TTS. Saves base64 streams locally and plays them via expo-av Sound.
 */

import { Platform } from 'react-native';
import * as Speech from 'expo-speech';
import * as FileSystem from 'expo-file-system';
import { Audio } from 'expo-av';

import ApiClient from './api';
import { useAppStore } from '../store/useAppStore';

let currentSound: Audio.Sound | null = null;
let currentWebSound: any = null;

// ─────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────

/** Max ms to wait for neural TTS before falling back to local speech */
const NEURAL_TTS_TIMEOUT_MS = 3000;

/**
 * Race a promise against a timeout. Rejects with 'TTS timeout'
 * if the promise does not resolve within the given milliseconds.
 */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error('TTS timeout')), ms)
  );
  return Promise.race([promise, timeout]);
}

export const VoiceService = {
  // ------------------------------------------------------------------
  // SPEAK NEURAL OR LOCAL FALLBACK
  // ------------------------------------------------------------------

  async speak(text: string, playbackKey: string = 'global'): Promise<void> {
    const store = useAppStore.getState();
    if (!store.ttsEnabled) return;

    // Suppress voice commands while TTS is active
    try {
      const VoiceCommandEngine = require('./voiceCommands').default;
      VoiceCommandEngine.setSuppressed(true);
    } catch (_) {}

    // Stop any currently playing audio
    await this.stop();

    const targetLang = store.targetLanguage;
    const rate = store.speechRate;

    if (Platform.OS === 'web') {
      try {
        store.setPlayingAudio(true, playbackKey);
        // Race neural TTS against a 3-second timeout
        const base64Audio = await withTimeout(
          ApiClient.generateSpeechBase64(text, targetLang, rate),
          NEURAL_TTS_TIMEOUT_MS
        );
        if (currentWebSound) {
          currentWebSound.pause();
        }
        currentWebSound = new (window as any).Audio(`data:audio/mp3;base64,${base64Audio}`);
        currentWebSound.play();
        currentWebSound.onended = () => {
          store.setPlayingAudio(false, null);
          currentWebSound = null;
          try {
            const VoiceCommandEngine = require('./voiceCommands').default;
            VoiceCommandEngine.setSuppressed(false);
          } catch (_) {}
        };
      } catch (error) {
        console.warn('VoiceService: Neural TTS timed out or failed, using browser speech:', error);
        store.setPlayingAudio(true, playbackKey);
        const speechLocale = targetLang === 'zh-CN' ? 'zh-CN' : targetLang;
        const utterance = new (window as any).SpeechSynthesisUtterance(text);
        utterance.lang = speechLocale;
        utterance.onend = () => {
          store.setPlayingAudio(false, null);
          try {
            const VoiceCommandEngine = require('./voiceCommands').default;
            VoiceCommandEngine.setSuppressed(false);
          } catch (_) {}
        };
        utterance.onerror = () => {
          store.setPlayingAudio(false, null);
          try {
            const VoiceCommandEngine = require('./voiceCommands').default;
            VoiceCommandEngine.setSuppressed(false);
          } catch (_) {}
        };
        (window as any).speechSynthesis.speak(utterance);
      }
      return;
    }

    try {
      // 1. Try server-side neural TTS with a strict timeout
      store.setPlayingAudio(true, playbackKey);
      
      const base64Audio = await withTimeout(
        ApiClient.generateSpeechBase64(text, targetLang, rate),
        NEURAL_TTS_TIMEOUT_MS
      );
      
      // Write base64 audio to local temporary file
      const fileUri = `${(FileSystem as any).cacheDirectory}speech_${playbackKey}_${Date.now()}.mp3`;
      await FileSystem.writeAsStringAsync(fileUri, base64Audio, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Play local audio file
      const { sound } = await Audio.Sound.createAsync(
        { uri: fileUri },
        { shouldPlay: true }
      );

      currentSound = sound;

      // Listen for playback completion
      sound.setOnPlaybackStatusUpdate((status) => {
        if (status.isLoaded && status.didJustFinish) {
          store.setPlayingAudio(false, null);
          // Cleanup temp file
          FileSystem.deleteAsync(fileUri, { idempotent: true }).catch(() => {});
          try {
            const VoiceCommandEngine = require('./voiceCommands').default;
            VoiceCommandEngine.setSuppressed(false);
          } catch (_) {}
        }
      });

    } catch (error) {
      console.warn('VoiceService: Neural TTS timed out or failed, using local speech:', error);
      
      // 2. Instant offline local Speech synthesis fallback
      store.setPlayingAudio(true, playbackKey);
      const speechLocale = targetLang === 'zh-CN' ? 'zh-CN' : targetLang;
      Speech.speak(text, {
        language: speechLocale,
        rate: 0.95,
        onDone: () => {
          store.setPlayingAudio(false, null);
          try {
            const VoiceCommandEngine = require('./voiceCommands').default;
            VoiceCommandEngine.setSuppressed(false);
          } catch (_) {}
        },
        onError: () => {
          store.setPlayingAudio(false, null);
          try {
            const VoiceCommandEngine = require('./voiceCommands').default;
            VoiceCommandEngine.setSuppressed(false);
          } catch (_) {}
        },
      });
    }
  },

  // ------------------------------------------------------------------
  // SCANNER NEURAL GUIDANCE FEEDBACK
  // ------------------------------------------------------------------

  async speakGuidance(message: string): Promise<void> {
    const store = useAppStore.getState();
    if (!store.ttsEnabled) return;

    // Suppress voice commands during guidance announcements
    try {
      const VoiceCommandEngine = require('./voiceCommands').default;
      VoiceCommandEngine.setSuppressed(true);
    } catch (_) {}

    // Guidance messages ("Hold steady", "Point camera") need to fire INSTANTLY.
    // We speak locally first so the user hears feedback with zero network delay.
    // Neural TTS is then attempted in the background as a quality enhancement,
    // but we never block on it for guidance.
    await this.stop();

    const targetLang = store.targetLanguage;

    if (Platform.OS === 'web') {
      // Fire browser SpeechSynthesis immediately — no network round-trip
      const utterance = new (window as any).SpeechSynthesisUtterance(message);
      utterance.lang = targetLang;
      utterance.rate = 1.1;
      (window as any).speechSynthesis.speak(utterance);

      // Silently try to upgrade to neural audio in background (non-blocking)
      ApiClient.generateGuidanceBase64(message, targetLang)
        .then((base64Audio) => {
          // Only play if browser speech hasn't already finished
          if (!(window as any).speechSynthesis.speaking) {
            if (currentWebSound) currentWebSound.pause();
            currentWebSound = new (window as any).Audio(`data:audio/mp3;base64,${base64Audio}`);
            currentWebSound.play();
          }
        })
        .catch(() => { /* silent — local speech is already playing */ });
      return;
    }

    // Native (Android/iOS): fire local speech immediately, zero wait
    Speech.speak(message, {
      language: targetLang,
      rate: 1.1, // slightly faster for snappy guidance feedback
      onDone: () => {
        try {
          const VoiceCommandEngine = require('./voiceCommands').default;
          VoiceCommandEngine.setSuppressed(false);
        } catch (_) {}
      },
      onError: () => {
        try {
          const VoiceCommandEngine = require('./voiceCommands').default;
          VoiceCommandEngine.setSuppressed(false);
        } catch (_) {}
      },
    });

    // Fire-and-forget neural audio upgrade in the background
    withTimeout(
      ApiClient.generateGuidanceBase64(message, targetLang),
      NEURAL_TTS_TIMEOUT_MS
    )
      .then(async (base64Audio) => {
        // Only play if local speech has already finished (don't overlap)
        const isSpeaking = await Speech.isSpeakingAsync();
        if (!isSpeaking) {
          const fileUri = `${(FileSystem as any).cacheDirectory}guidance_${Date.now()}.mp3`;
          await FileSystem.writeAsStringAsync(fileUri, base64Audio, {
            encoding: FileSystem.EncodingType.Base64,
          });
          const { sound } = await Audio.Sound.createAsync(
            { uri: fileUri },
            { shouldPlay: true }
          );
          currentSound = sound;
          sound.setOnPlaybackStatusUpdate((status) => {
            if (status.isLoaded && status.didJustFinish) {
              FileSystem.deleteAsync(fileUri, { idempotent: true }).catch(() => {});
              try {
                const VoiceCommandEngine = require('./voiceCommands').default;
                VoiceCommandEngine.setSuppressed(false);
              } catch (_) {}
            }
          });
        } else {
          // If already speaking, make sure we unsuppress when that speech ends
          try {
            const VoiceCommandEngine = require('./voiceCommands').default;
            VoiceCommandEngine.setSuppressed(false);
          } catch (_) {}
        }
      })
      .catch(() => {
        try {
          const VoiceCommandEngine = require('./voiceCommands').default;
          VoiceCommandEngine.setSuppressed(false);
        } catch (_) {}
      });
  },

  // ------------------------------------------------------------------
  // CONTROLS
  // ------------------------------------------------------------------

  async stop(): Promise<void> {
    const store = useAppStore.getState();
    
    // Stop local speech engine
    await Speech.stop();

    // Reset voice commands suppression
    try {
      const VoiceCommandEngine = require('./voiceCommands').default;
      VoiceCommandEngine.setSuppressed(false);
    } catch (_) {}

    if (Platform.OS === 'web') {
      if ((window as any).speechSynthesis) {
        (window as any).speechSynthesis.cancel();
      }
      if (currentWebSound) {
        currentWebSound.pause();
        currentWebSound = null;
      }
      store.setPlayingAudio(false, null);
      return;
    }

    // Stop audio file player
    if (currentSound) {
      try {
        const status = await currentSound.getStatusAsync();
        if (status.isLoaded) {
          await currentSound.stopAsync();
          await currentSound.unloadAsync();
        }
      } catch (e) {
        // Ignore unloading errors
      }
      currentSound = null;
    }

    store.setPlayingAudio(false, null);
  },
};
export default VoiceService;
