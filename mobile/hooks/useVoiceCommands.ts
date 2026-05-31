/**
 * ═══════════════════════════════════
 * mobile/hooks/useVoiceCommands.ts
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Voice Commands React Hook
 * Each screen uses this to register its own callbacks with the
 * VoiceCommandEngine. Automatically cleans up on unmount/blur.
 */

import { useEffect, useRef } from 'react';
import { useFocusEffect } from 'expo-router';
import VoiceCommandEngine from '../services/voiceCommands';

interface ScannerCallbacks {
  onCapture?: () => void;
  onReadResult?: () => void;
  onFlashlight?: (on: boolean) => void;
  onStopAudio?: () => void;
}

interface UploadCallbacks {
  onPickImage?: () => void;
  onReadResult?: () => void;
  onStopAudio?: () => void;
}

interface GenericCallbacks {
  onStopAudio?: () => void;
}

// ─────────────────────────────────────────────────────────────
// SCANNER SCREEN HOOK
// ─────────────────────────────────────────────────────────────
export function useScannerVoiceCommands(callbacks: ScannerCallbacks) {
  const cbRef = useRef(callbacks);
  cbRef.current = callbacks;

  useFocusEffect(() => {
    // Register callbacks when scanner is focused
    if (cbRef.current.onCapture) {
      VoiceCommandEngine.setCaptureCallback(() => cbRef.current.onCapture?.());
    }
    if (cbRef.current.onReadResult) {
      VoiceCommandEngine.setReadResultCallback(() => cbRef.current.onReadResult?.());
    }
    if (cbRef.current.onFlashlight) {
      VoiceCommandEngine.setFlashlightCallback((on) => cbRef.current.onFlashlight?.(on));
    }
    if (cbRef.current.onStopAudio) {
      VoiceCommandEngine.setStopAudioCallback(() => cbRef.current.onStopAudio?.());
    }

    return () => {
      // Clear when leaving scanner
      VoiceCommandEngine.clearCaptureCallback();
      VoiceCommandEngine.clearReadResultCallback();
      VoiceCommandEngine.clearFlashlightCallback();
    };
  });
}

// ─────────────────────────────────────────────────────────────
// UPLOAD SCREEN HOOK
// ─────────────────────────────────────────────────────────────
export function useUploadVoiceCommands(callbacks: UploadCallbacks) {
  const cbRef = useRef(callbacks);
  cbRef.current = callbacks;

  useFocusEffect(() => {
    if (cbRef.current.onPickImage) {
      VoiceCommandEngine.setPickImageCallback(() => cbRef.current.onPickImage?.());
    }
    if (cbRef.current.onReadResult) {
      VoiceCommandEngine.setReadResultCallback(() => cbRef.current.onReadResult?.());
    }
    if (cbRef.current.onStopAudio) {
      VoiceCommandEngine.setStopAudioCallback(() => cbRef.current.onStopAudio?.());
    }

    return () => {
      VoiceCommandEngine.clearPickImageCallback();
      VoiceCommandEngine.clearReadResultCallback();
    };
  });
}

// ─────────────────────────────────────────────────────────────
// GENERIC HOOK (history, settings)
// ─────────────────────────────────────────────────────────────
export function useGenericVoiceCommands(callbacks: GenericCallbacks = {}) {
  const cbRef = useRef(callbacks);
  cbRef.current = callbacks;

  useFocusEffect(() => {
    if (cbRef.current.onStopAudio) {
      VoiceCommandEngine.setStopAudioCallback(() => cbRef.current.onStopAudio?.());
    }

    return () => {
      // No screen-specific callbacks to clear here
    };
  });
}
