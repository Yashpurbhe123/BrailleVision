/**
 * ═══════════════════════════════════
 * 📄 FILE: mobile/app/onboarding.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Onboarding Wizard
 * Beautiful, responsive, world-class 4-stage onboarding slider introducing the core
 * AI system features and guiding target language and accessibility configurations.
 */

import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Dimensions,
  AccessibilityInfo,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Eye, Volume2, Languages, Accessibility, Check, ChevronRight, ChevronLeft } from 'lucide-react-native';
import { useAppStore } from '../store/useAppStore';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from '../constants/theme';
import VoiceService from '../services/voice';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

const ONBOARDING_STEPS = [
  {
    title: 'BrailleVision AI',
    desc: 'Welcome to the future of tactile reading. Decode physical embossed Braille dots instantly into clear speech using advanced machine learning.',
  },
  {
    title: 'Set Target Language',
    desc: 'Translate English Braille instantly into one of many global languages and listen to the translated audio output spoken aloud.',
  },
  {
    title: 'Accessibility Options',
    desc: 'Customize your high contrast profile and voice speech parameters to make the device experience fully tailored to your needs.',
  },
];

export default function OnboardingScreen() {
  const router = useRouter();
  const store = useAppStore();
  const [currentStep, setCurrentStep] = useState(0);
  
  // Settings edit states
  const [selectedLang, setSelectedLang] = useState(store.targetLanguage);
  const [highContrastActive, setHighContrastActive] = useState(store.highContrast);
  const [ttsActive, setTtsActive] = useState(store.ttsEnabled);

  const announceToAccessibility = (text: string) => {
    AccessibilityInfo.announceForAccessibility(text);
  };

  const handleNext = () => {
    const nextStep = currentStep + 1;
    if (nextStep < ONBOARDING_STEPS.length) {
      setCurrentStep(nextStep);
      let speechText = `${ONBOARDING_STEPS[nextStep].title}. ${ONBOARDING_STEPS[nextStep].desc}`;
      if (nextStep === 0) {
        speechText += " Also featuring Live Guidance Cues: Receive real-time high-fidelity voice guidelines to center and align your device camera correctly.";
      }
      VoiceService.speakGuidance(speechText);
      announceToAccessibility(speechText);
    } else {
      // Save all states to store
      store.setTargetLanguage(selectedLang);
      store.setHighContrast(highContrastActive);
      store.setTtsEnabled(ttsActive);
      store.completeOnboarding();
      
      VoiceService.speakGuidance("Onboarding complete. Opening scanner.");
      router.replace('/(tabs)/scanner');
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      const prevStep = currentStep - 1;
      setCurrentStep(prevStep);
      let speechText = `${ONBOARDING_STEPS[prevStep].title}. ${ONBOARDING_STEPS[prevStep].desc}`;
      if (prevStep === 0) {
        speechText += " Also featuring Live Guidance Cues: Receive real-time high-fidelity voice guidelines to center and align your device camera correctly.";
      }
      VoiceService.speakGuidance(speechText);
      announceToAccessibility(speechText);
    }
  };

  const renderIcon = () => {
    const iconSize = 72;
    const iconColor = COLORS.primary;
    
    switch (currentStep) {
      case 0:
        return <Eye color={iconColor} size={iconSize} strokeWidth={1.2} />;
      case 1:
        return <Languages color={iconColor} size={iconSize} strokeWidth={1.2} />;
      case 2:
        return <Accessibility color={iconColor} size={iconSize} strokeWidth={1.2} />;
      default:
        return null;
    }
  };

  const renderConfigControls = () => {
    if (currentStep === 0) {
      // Live Guidance Cues card merged on main page
      return (
        <View style={[styles.guidanceCard, highContrastActive && styles.guidanceCardHighContrast]}>
          <View style={styles.guidanceHeader}>
            <View style={[styles.guidanceIconWrapper, highContrastActive && styles.guidanceIconWrapperHighContrast]}>
              <Volume2 color={highContrastActive ? '#FFFF00' : COLORS.primary} size={22} strokeWidth={2} />
            </View>
            <Text style={[styles.guidanceTitle, highContrastActive && styles.guidanceTitleHighContrast]}>
              LIVE GUIDANCE CUES
            </Text>
          </View>
          <Text style={[styles.guidanceDesc, highContrastActive && styles.guidanceDescHighContrast]}>
            Receive real-time high-fidelity voice guidelines to center and align your device camera correctly (e.g. "Move closer", "Hold steady").
          </Text>
        </View>
      );
    }

    if (currentStep === 1) {
      // Language config - attractive 3-column Box Grid layout
      const languages = [
        { label: 'English', code: 'en', badge: 'EN' },
        { label: 'Hindi', code: 'hi', badge: 'HI' },
        { label: 'Spanish', code: 'es', badge: 'ES' },
        { label: 'French', code: 'fr', badge: 'FR' },
        { label: 'German', code: 'de', badge: 'DE' },
        { label: 'Tamil', code: 'ta', badge: 'TA' },
      ];
      return (
        <View style={styles.configContainer}>
          <Text style={styles.configTitle}>SELECT TRANSLATION</Text>
          <View style={styles.langGrid}>
            {languages.map((lang) => {
              const isActive = selectedLang === lang.code;
              return (
                <TouchableOpacity
                  key={lang.code}
                  style={[
                    styles.optionBox,
                    isActive && styles.optionBoxActive,
                    highContrastActive && styles.optionBoxHighContrast,
                    highContrastActive && isActive && styles.optionBoxActiveHighContrast,
                  ]}
                  onPress={() => {
                    setSelectedLang(lang.code);
                    VoiceService.speakGuidance(`Selected ${lang.label}`);
                  }}
                  accessibilityRole="radio"
                  accessibilityState={{ checked: isActive }}
                  accessibilityLabel={`Select ${lang.label}`}
                >
                  {isActive && (
                    <View style={[styles.checkBadge, highContrastActive && styles.checkBadgeHighContrast]}>
                      <Check color={highContrastActive ? '#FFFF00' : COLORS.primary} size={11} strokeWidth={3.5} />
                    </View>
                  )}
                  <Text style={[
                    styles.langBadge, 
                    isActive && styles.langBadgeActive, 
                    highContrastActive && styles.textHighContrast,
                    highContrastActive && isActive && styles.textActiveHighContrast
                  ]}>
                    {lang.badge}
                  </Text>
                  <Text style={[
                    styles.langLabel, 
                    isActive && styles.langLabelActive, 
                    highContrastActive && styles.textHighContrastSecondary,
                    highContrastActive && isActive && styles.textActiveHighContrast
                  ]} numberOfLines={1}>
                    {lang.label}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>
      );
    }

    if (currentStep === 2) {
      // Accessibility config
      return (
        <View style={styles.configContainer}>
          <Text style={styles.configTitle}>PREFERENCES</Text>
          
          <TouchableOpacity
            style={[styles.toggleCard, highContrastActive && styles.toggleCardActive]}
            onPress={() => {
              const nextVal = !highContrastActive;
              setHighContrastActive(nextVal);
              VoiceService.speakGuidance(`High contrast mode ${nextVal ? 'enabled' : 'disabled'}`);
            }}
            accessibilityRole="checkbox"
            accessibilityState={{ checked: highContrastActive }}
            accessibilityLabel="Toggle High Contrast mode"
          >
            <View style={styles.toggleTextContainer}>
              <Text style={styles.toggleLabel}>High Contrast Colors</Text>
              <Text style={styles.toggleDesc}>Pure high-visibility blacks & bright yellows</Text>
            </View>
            <View style={[styles.customSwitch, highContrastActive && styles.customSwitchActive]}>
              <View style={[styles.customSwitchThumb, highContrastActive && styles.customSwitchThumbActive]} />
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.toggleCard, ttsActive && styles.toggleCardActive]}
            onPress={() => {
              const nextVal = !ttsActive;
              setTtsActive(nextVal);
              VoiceService.speakGuidance(`Neural speech ${nextVal ? 'enabled' : 'disabled'}`);
            }}
            accessibilityRole="checkbox"
            accessibilityState={{ checked: ttsActive }}
            accessibilityLabel="Toggle Neural Text to Speech voice output"
          >
            <View style={styles.toggleTextContainer}>
              <Text style={styles.toggleLabel}>Neural Audio Output</Text>
              <Text style={styles.toggleDesc}>Spoken guidance guidelines and voice results</Text>
            </View>
            <View style={[styles.customSwitch, ttsActive && styles.customSwitchActive]}>
              <View style={[styles.customSwitchThumb, ttsActive && styles.customSwitchThumbActive]} />
            </View>
          </TouchableOpacity>
        </View>
      );
    }

    return null;
  };

  const step = ONBOARDING_STEPS[currentStep];

  return (
    <View style={[styles.container, highContrastActive && styles.highContrastContainer]}>
      {/* Step Progress Bar */}
      <View style={styles.progressBar} accessibilityLabel={`Step ${currentStep + 1} of ${ONBOARDING_STEPS.length}`}>
        {ONBOARDING_STEPS.map((_, i) => (
          <View
            key={i}
            style={[
              styles.progressDot,
              i <= currentStep ? styles.progressDotActive : null,
              highContrastActive && i <= currentStep ? styles.progressDotHighContrast : null,
            ]}
          />
        ))}
      </View>

      {/* Main Slide Card */}
      <ScrollView contentContainerStyle={styles.cardContainer} showsVerticalScrollIndicator={false}>
        <View style={styles.iconWrapper}>
          {renderIcon()}
          <View style={styles.iconPulseRing} />
        </View>
        <Text style={styles.title} accessibilityRole="header">{step.title}</Text>
        <Text style={styles.description}>{step.desc}</Text>

        {renderConfigControls()}
      </ScrollView>

      {/* Bottom Nav Controls */}
      <View style={styles.navBar}>
        {currentStep > 0 ? (
          <TouchableOpacity
            style={styles.backBtn}
            onPress={handleBack}
            accessibilityRole="button"
            accessibilityLabel="Back to previous screen"
          >
            <ChevronLeft color={COLORS.textSecondary} size={18} strokeWidth={2.5} style={{ marginRight: 4 }} />
            <Text style={styles.backBtnText}>BACK</Text>
          </TouchableOpacity>
        ) : (
          <View style={styles.backBtnPlaceholder} />
        )}

        <TouchableOpacity
          style={[styles.nextBtn, highContrastActive && styles.nextBtnHighContrast]}
          onPress={handleNext}
          accessibilityRole="button"
          accessibilityLabel={currentStep === ONBOARDING_STEPS.length - 1 ? '✓ Finish and open app' : 'Next screen'}
        >
          <Text style={[styles.nextBtnText, highContrastActive && styles.nextBtnTextHighContrast]}>
            {currentStep === ONBOARDING_STEPS.length - 1 ? 'FINISH' : 'CONTINUE'}
          </Text>
          {currentStep < ONBOARDING_STEPS.length - 1 ? (
            <ChevronRight color="#0B0E14" size={16} strokeWidth={3} style={{ marginLeft: 4 }} />
          ) : (
            <Check color="#0B0E14" size={16} strokeWidth={3} style={{ marginLeft: 4 }} />
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
    paddingTop: 54,
    paddingBottom: 24,
    paddingHorizontal: SPACING.lg,
    justifyContent: 'space-between',
  },
  highContrastContainer: {
    backgroundColor: '#000000',
  },
  progressBar: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: SPACING.md,
    marginTop: 10,
  },
  progressDot: {
    height: 5,
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    marginHorizontal: 4,
    borderRadius: BORDER_RADIUS.round,
  },
  progressDotActive: {
    backgroundColor: COLORS.primary,
  },
  progressDotHighContrast: {
    backgroundColor: '#FFFF00', // pure accessibility yellow
  },
  cardContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: SPACING.lg,
  },
  iconWrapper: {
    width: 150,
    height: 150,
    borderRadius: 75,
    backgroundColor: 'rgba(240, 86, 200, 0.04)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.xl,
    borderWidth: 2,
    borderColor: 'rgba(240, 86, 200, 0.25)',
    position: 'relative',
  },
  iconPulseRing: {
    position: 'absolute',
    width: 172,
    height: 172,
    borderRadius: 86,
    borderWidth: 1.5,
    borderColor: 'rgba(240, 86, 200, 0.1)',
  },
  title: {
    fontSize: 28,
    fontWeight: TYPOGRAPHY.weight.bold,
    color: COLORS.text,
    textAlign: 'center',
    marginBottom: SPACING.md,
    letterSpacing: -0.5,
  },
  description: {
    fontSize: 15,
    color: COLORS.textSecondary,
    textAlign: 'center',
    lineHeight: 24,
    paddingHorizontal: SPACING.md,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  configContainer: {
    marginTop: SPACING.xl,
    width: '100%',
  },
  configTitle: {
    fontSize: 11,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.primary,
    marginBottom: SPACING.md,
    textAlign: 'center',
    letterSpacing: 1.8,
  },
  langGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    width: '100%',
  },
  optionBox: {
    width: '31.3%',
    height: 94,
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
    borderRadius: 16,
    borderWidth: 1.5,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: SPACING.md,
    padding: 8,
    position: 'relative',
  },
  optionBoxActive: {
    borderColor: 'rgba(240, 86, 200, 0.35)',
    backgroundColor: 'rgba(240, 86, 200, 0.12)',
  },
  optionBoxHighContrast: {
    borderColor: '#FFFFFF',
    backgroundColor: '#000000',
  },
  optionBoxActiveHighContrast: {
    borderColor: '#FFFF00',
    backgroundColor: '#000000',
    borderWidth: 3,
  },
  checkBadge: {
    position: 'absolute',
    top: 6,
    right: 6,
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: 'rgba(240, 86, 200, 0.08)',
    borderWidth: 1.2,
    borderColor: 'rgba(240, 86, 200, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkBadgeHighContrast: {
    backgroundColor: '#000000',
    borderColor: '#FFFF00',
    borderWidth: 1.5,
  },
  langBadge: {
    fontSize: 16,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.primary,
    marginBottom: 2,
    letterSpacing: -0.2,
  },
  langBadgeActive: {
    color: COLORS.primaryLight,
  },
  langLabel: {
    fontSize: 11,
    fontWeight: TYPOGRAPHY.weight.bold,
    color: COLORS.textSecondary,
    textAlign: 'center',
  },
  langLabelActive: {
    color: COLORS.text,
  },
  textHighContrast: {
    color: '#FFFFFF',
  },
  textHighContrastSecondary: {
    color: '#D1D5DB',
  },
  textActiveHighContrast: {
    color: '#FFFF00',
    fontWeight: TYPOGRAPHY.weight.heavy,
  },
  toggleCard: {
    width: '100%',
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
    borderWidth: 1.5,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 20,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: SPACING.md,
  },
  toggleCardActive: {
    borderColor: 'rgba(240, 86, 200, 0.25)',
    backgroundColor: 'rgba(240, 86, 200, 0.04)',
  },
  toggleTextContainer: {
    flex: 1,
    marginRight: SPACING.md,
  },
  toggleLabel: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: TYPOGRAPHY.weight.heavy,
  },
  toggleDesc: {
    color: COLORS.textSecondary,
    fontSize: 11,
    marginTop: 2,
    fontWeight: TYPOGRAPHY.weight.heavy,
    textTransform: 'uppercase',
    letterSpacing: 0.2,
  },
  customSwitch: {
    width: 46,
    height: 26,
    borderRadius: 13,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    padding: 2,
    justifyContent: 'center',
  },
  customSwitchActive: {
    backgroundColor: COLORS.primary,
  },
  customSwitchThumb: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: COLORS.textSecondary,
  },
  customSwitchThumbActive: {
    transform: [{ translateX: 20 }],
    backgroundColor: '#FFFFFF',
  },
  navBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: SPACING.md,
  },
  backBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  backBtnText: {
    color: COLORS.textSecondary,
    fontSize: 13,
    fontWeight: TYPOGRAPHY.weight.heavy,
  },
  backBtnPlaceholder: {
    width: 80,
  },
  nextBtn: {
    backgroundColor: COLORS.accent, // Neon Cyan
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 140,
    shadowColor: COLORS.accent,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 4,
  },
  nextBtnHighContrast: {
    backgroundColor: '#FFFF00', // Bright accessibility yellow
    shadowColor: '#FFFF00',
  },
  nextBtnText: {
    color: '#0B0E14', // Stark stark contrast contrast
    fontSize: 13,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 0.8,
  },
  nextBtnTextHighContrast: {
    color: '#000000',
  },
  guidanceCard: {
    marginTop: SPACING.xl,
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
    borderWidth: 1.5,
    borderColor: 'rgba(240, 86, 200, 0.2)',
    borderRadius: 20,
    padding: 20,
    width: '100%',
  },
  guidanceCardHighContrast: {
    borderColor: '#FFFF00',
    backgroundColor: '#000000',
  },
  guidanceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: SPACING.sm,
  },
  guidanceIconWrapper: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(240, 86, 200, 0.08)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  guidanceIconWrapperHighContrast: {
    backgroundColor: '#000000',
    borderWidth: 1,
    borderColor: '#FFFF00',
  },
  guidanceTitle: {
    fontSize: 14,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.primary,
    letterSpacing: 1.2,
  },
  guidanceTitleHighContrast: {
    color: '#FFFF00',
  },
  guidanceDesc: {
    fontSize: 13,
    color: COLORS.textSecondary,
    lineHeight: 20,
    fontWeight: TYPOGRAPHY.weight.semibold,
  },
  guidanceDescHighContrast: {
    color: '#FFFFFF',
  },
});
