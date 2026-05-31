/**
 * ═══════════════════════════════════
 * 📄 FILE 33/42: mobile/app/(tabs)/settings.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Application Preferences & Accessibility Settings
 * Exposes server connectivity checks, speech speed regulators, high-contrast,
 * voice output switches, and language selection.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Switch,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { Server, SlidersHorizontal, Globe, Compass, Check, RefreshCw, Sparkles, Volume2, ShieldAlert } from 'lucide-react-native';

import { useAppStore } from '../../store/useAppStore';
import ApiClient from '../../services/api';
import VoiceService from '../../services/voice';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from '../../constants/theme';
import { useGenericVoiceCommands } from '../../hooks/useVoiceCommands';
import { useFocusEffect } from 'expo-router';

export default function SettingsScreen() {
  const store = useAppStore();

  const [inputUrl, setInputUrl] = useState(store.apiUrl);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'success' | 'failed' | null>(null);
  
  // Available translation languages
  const [languages, setLanguages] = useState<Record<string, string>>({
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Spanish": "es",
    "French": "fr",
  });

  useEffect(() => {
    // Attempt to load supported languages from backend dynamically on mount
    const fetchLangs = async () => {
      try {
        const backendLangs = await ApiClient.fetchLanguages();
        if (backendLangs) {
          setLanguages(backendLangs);
        }
      } catch (e) {
        // Fallback to static defaults if server is unreachable
      }
    };
    fetchLangs();
  }, []);

  // Announce settings screen on focus
  useFocusEffect(
    React.useCallback(() => {
      VoiceService.speakGuidance("Settings screen. Say go to scanner to start scanning, or say stop to stop audio.");
    }, [])
  );

  // ── Voice commands for blind users ──
  useGenericVoiceCommands({
    onStopAudio: () => VoiceService.stop(),
  });

  const handleTestConnection = async () => {
    setTestingConnection(true);
    setConnectionStatus(null);
    VoiceService.speakGuidance("Testing server connection.");

    try {
      // Temporarily write value to verify
      const originalUrl = store.apiUrl;
      store.setApiUrl(inputUrl);

      // Hit health check endpoint
      const response = await fetch(`${inputUrl}/health`, { method: 'GET' });
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'healthy') {
          setConnectionStatus('success');
          VoiceService.speakGuidance("Connection successful.");
          return;
        }
      }
      throw new Error();
    } catch (e) {
      setConnectionStatus('failed');
      VoiceService.speakGuidance("Connection failed. Check server address.");
    } finally {
      setTestingConnection(false);
    }
  };

  const handleSaveUrl = () => {
    store.setApiUrl(inputUrl);
    VoiceService.speakGuidance("Server URL updated.");
    Alert.alert("Success", "Server connection URL saved.");
  };

  const toggleHighContrast = (val: boolean) => {
    store.setHighContrast(val);
    VoiceService.speakGuidance(`High contrast mode ${val ? 'enabled' : 'disabled'}`);
  };

  const toggleTts = (val: boolean) => {
    store.setTtsEnabled(val);
    VoiceService.speakGuidance(`Neural voice output ${val ? 'enabled' : 'muted'}`);
  };

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
  };

  const getRateText = () => {
    if (store.speechRate === '+25%') return 'Fast (1.25x)';
    if (store.speechRate === '-15%') return 'Slow (0.85x)';
    return 'Normal (1.0x)';
  };

  const handleLanguageChange = (langCode: string, langName: string) => {
    store.setTargetLanguage(langCode);
    VoiceService.speakGuidance(`Selected translation language: ${langName}`);
  };

  const handleResetOnboarding = () => {
    Alert.alert(
      "Reset App",
      "Are you sure you want to return to the onboarding tutorial?",
      [
        { text: "Cancel", style: "cancel" },
        { 
          text: "Reset", 
          style: "destructive",
          onPress: () => {
            store.resetOnboarding();
            VoiceService.speakGuidance("App reset. Onboarding opened.");
          }
        }
      ]
    );
  };

  return (
    <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
      <Text style={styles.title} accessibilityRole="header">Settings</Text>
      <Text style={styles.subtitle}>Configure translation preferences, speech parameters, and server details.</Text>

      {/* SECTION 1: SERVER IP */}
      <View style={styles.sectionCard}>
        <View style={styles.sectionHeaderRow}>
          <Server color={COLORS.primary} size={15} strokeWidth={2.5} style={{ marginRight: 6 }} />
          <Text style={styles.sectionTitle}>SERVER CONFIGURATION</Text>
        </View>
        
        <TextInput
          style={styles.textInput}
          value={inputUrl}
          onChangeText={setInputUrl}
          placeholder="http://192.168.1.100:8000"
          placeholderTextColor="#475569"
          autoCapitalize="none"
          autoCorrect={false}
          accessibilityLabel="Server connection endpoint address"
        />

        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={styles.actionBtn}
            onPress={handleTestConnection}
            disabled={testingConnection}
            accessibilityRole="button"
            accessibilityLabel="Test server connection"
          >
            {testingConnection ? (
              <ActivityIndicator size="small" color={COLORS.primary} />
            ) : (
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <RefreshCw color={COLORS.primary} size={12} strokeWidth={2.5} style={{ marginRight: 5 }} />
                <Text style={styles.actionBtnText}>TEST CONNECT</Text>
              </View>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionBtn, styles.actionBtnSave]}
            onPress={handleSaveUrl}
            accessibilityRole="button"
            accessibilityLabel="Save server connection URL"
          >
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Check color="#0B0E14" size={13} strokeWidth={3} style={{ marginRight: 4 }} />
              <Text style={[styles.actionBtnText, styles.actionBtnTextSave]}>SAVE URL</Text>
            </View>
          </TouchableOpacity>
        </View>

        {connectionStatus === 'success' && (
          <View style={styles.successWrapper}>
            <Check color={COLORS.success} size={13} strokeWidth={2.5} style={{ marginRight: 4 }} />
            <Text style={styles.successMessage}>Connected: Scanning server is online.</Text>
          </View>
        )}
        {connectionStatus === 'failed' && (
          <View style={styles.errorWrapper}>
            <ShieldAlert color={COLORS.error} size={13} strokeWidth={2.5} style={{ marginRight: 4 }} />
            <Text style={styles.errorMessage}>Failed: Verify endpoint IP address.</Text>
          </View>
        )}
      </View>

      {/* SECTION 2: ACCESSIBILITY PREFERENCES */}
      <View style={styles.sectionCard}>
        <View style={styles.sectionHeaderRow}>
          <SlidersHorizontal color={COLORS.primary} size={15} strokeWidth={2.5} style={{ marginRight: 6 }} />
          <Text style={styles.sectionTitle}>ACCESSIBILITY & AUDIO</Text>
        </View>

        {/* High contrast */}
        <View style={styles.settingRow}>
          <View style={styles.settingRowText}>
            <Text style={styles.settingLabel}>High Contrast Mode</Text>
            <Text style={styles.settingSubLabel}>Enhance readability with high-visibility yellows</Text>
          </View>
          <Switch
            value={store.highContrast}
            onValueChange={toggleHighContrast}
            trackColor={{ false: 'rgba(255,255,255,0.06)', true: COLORS.primary }}
            thumbColor={store.highContrast ? '#FFFFFF' : '#64748B'}
            accessibilityLabel="Toggle High Contrast mode colors"
          />
        </View>

        {/* TTS Toggle */}
        <View style={styles.settingRow}>
          <View style={styles.settingRowText}>
            <Text style={styles.settingLabel}>Neural Speech Output</Text>
            <Text style={styles.settingSubLabel}>Aloud voice readings for visually impaired users</Text>
          </View>
          <Switch
            value={store.ttsEnabled}
            onValueChange={toggleTts}
            trackColor={{ false: 'rgba(255,255,255,0.06)', true: COLORS.primary }}
            thumbColor={store.ttsEnabled ? '#FFFFFF' : '#64748B'}
            accessibilityLabel="Toggle automatic voice speech reading"
          />
        </View>

        {/* Speech speed */}
        <TouchableOpacity
          style={styles.settingRowClickable}
          onPress={cycleSpeechRate}
          accessibilityRole="button"
          accessibilityLabel={`Voice speaking speed, current speed is ${getRateText()}. Double tap to change.`}
        >
          <View style={{ flex: 1 }}>
            <Text style={styles.settingLabel}>Voice Speaking Speed</Text>
            <Text style={styles.settingSubLabel}>Adjust playback speech rate on the fly</Text>
          </View>
          <View style={styles.speedPill}>
            <Volume2 color={COLORS.primary} size={13} strokeWidth={2.5} style={{ marginRight: 4 }} />
            <Text style={styles.clickableValue}>{getRateText()}</Text>
          </View>
        </TouchableOpacity>
      </View>

      {/* SECTION 3: TRANSLATION LANGUAGE */}
      <View style={styles.sectionCard}>
        <View style={styles.sectionHeaderRow}>
          <Globe color={COLORS.primary} size={15} strokeWidth={2.5} style={{ marginRight: 6 }} />
          <Text style={styles.sectionTitle}>TRANSLATION LANGUAGE</Text>
        </View>
        <Text style={styles.sectionDesc}>Select the default target language for translating Braille output.</Text>

        <View style={styles.langGrid}>
          {Object.entries(languages).map(([langName, langCode]) => {
            const isActive = store.targetLanguage === langCode;
            return (
              <TouchableOpacity
                key={langCode}
                style={[styles.langChip, isActive && styles.langChipActive]}
                onPress={() => handleLanguageChange(langCode, langName)}
                accessibilityRole="radio"
                accessibilityState={{ checked: isActive }}
                accessibilityLabel={`Select default translation to ${langName}`}
              >
                <Text style={[styles.langChipText, isActive && styles.langChipTextActive]}>
                  {langName}
                </Text>
                {isActive && <Check color={COLORS.primary} size={12} strokeWidth={2.5} style={{ marginLeft: 4 }} />}
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {/* SECTION 4: RESET */}
      <View style={styles.sectionCard}>
        <View style={styles.sectionHeaderRow}>
          <Compass color={COLORS.primary} size={15} strokeWidth={2.5} style={{ marginRight: 6 }} />
          <Text style={styles.sectionTitle}>SYSTEM ACTIONS</Text>
        </View>
        
        <TouchableOpacity
          style={styles.resetBtn}
          onPress={handleResetOnboarding}
          accessibilityRole="button"
          accessibilityLabel="Open application tutorial onboarding wizard"
        >
          <Sparkles color={COLORS.textSecondary} size={13} strokeWidth={2.5} style={{ marginRight: 5 }} />
          <Text style={styles.resetBtnText}>LAUNCH ONBOARDING TUTORIAL</Text>
        </TouchableOpacity>

        <Text style={styles.versionText}>BrailleVision AI — Version 1.0.0 (Build 51)</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: COLORS.background,
    paddingTop: 54,
    paddingBottom: 116,
    paddingHorizontal: SPACING.lg,
  },
  title: {
    fontSize: 28,
    fontWeight: TYPOGRAPHY.weight.bold,
    color: COLORS.text,
    marginBottom: 6,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 13,
    color: COLORS.textSecondary,
    lineHeight: 20,
    marginBottom: SPACING.xl,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  sectionCard: {
    backgroundColor: 'rgba(18, 12, 32, 0.76)',
    borderRadius: 28,
    borderWidth: 1.8,
    borderColor: COLORS.border,
    padding: 18,
    marginBottom: SPACING.md,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 4,
  },
  sectionHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 14,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.primary,
    letterSpacing: 1.5,
  },
  sectionDesc: {
    fontSize: 12,
    color: COLORS.textSecondary,
    marginBottom: 14,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  textInput: {
    backgroundColor: 'rgba(5, 3, 10, 0.75)',
    color: COLORS.text,
    paddingHorizontal: SPACING.md,
    paddingVertical: 14,
    fontSize: 14,
    borderRadius: 20,
    borderWidth: 1.8,
    borderColor: 'rgba(240, 86, 200, 0.3)',
    marginBottom: SPACING.md,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  actionBtn: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
    borderWidth: 1.8,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    paddingVertical: 14,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 6,
  },
  actionBtnSave: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
    marginRight: 0,
    marginLeft: 6,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 6,
    elevation: 4,
  },
  actionBtnText: {
    fontSize: 10.5,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.primary,
    letterSpacing: 0.8,
  },
  actionBtnTextSave: {
    color: '#FFFFFF',
  },
  successWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 14,
  },
  successMessage: {
    color: COLORS.success,
    fontSize: 12,
    fontWeight: 'bold',
  },
  errorWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 14,
  },
  errorMessage: {
    color: COLORS.error,
    fontSize: 12,
    fontWeight: 'bold',
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: 1.5,
    borderBottomColor: 'rgba(255, 255, 255, 0.03)',
  },
  settingRowText: {
    flex: 1,
    marginRight: SPACING.md,
  },
  settingRowClickable: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 16,
  },
  settingLabel: {
    fontSize: 14.5,
    color: COLORS.text,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  settingSubLabel: {
    fontSize: 10,
    color: COLORS.textSecondary,
    marginTop: 2,
    fontWeight: TYPOGRAPHY.weight.heavy,
    textTransform: 'uppercase',
    letterSpacing: 0.2,
  },
  speedPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(240, 86, 200, 0.08)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: 'rgba(240, 86, 200, 0.25)',
  },
  clickableValue: {
    fontSize: 11,
    color: COLORS.primary,
    fontWeight: TYPOGRAPHY.weight.heavy,
  },
  langGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 6,
  },
  langChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
    borderColor: 'rgba(255, 255, 255, 0.05)',
    borderWidth: 1.5,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 14,
    marginRight: 10,
    marginBottom: 10,
  },
  langChipActive: {
    borderColor: 'rgba(240, 86, 200, 0.35)',
    backgroundColor: 'rgba(240, 86, 200, 0.12)',
  },
  langChipText: {
    color: COLORS.textSecondary,
    fontSize: 12.5,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  langChipTextActive: {
    color: COLORS.primary,
  },
  resetBtn: {
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
    borderWidth: 1.8,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    paddingVertical: 16,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    marginBottom: 14,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 2,
  },
  resetBtnText: {
    color: COLORS.text,
    fontSize: 11,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 0.5,
  },
  versionText: {
    fontSize: 10,
    color: COLORS.textMuted,
    textAlign: 'center',
    fontWeight: TYPOGRAPHY.weight.bold,
    letterSpacing: 0.5,
  },
});
