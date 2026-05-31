/**
 * ═══════════════════════════════════
 * 📄 FILE 39b/42: mobile/app/index.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — App Entry Redirector
 * Checks Zustand store state for onboarding completion and handles routing.
 */

import React from 'react';
import { Redirect } from 'expo-router';
import { useAppStore } from '../store/useAppStore';

export default function Index() {
  const isOnboardingCompleted = useAppStore((state) => state.isOnboardingCompleted);

  if (!isOnboardingCompleted) {
    return <Redirect href="/onboarding" />;
  }

  return <Redirect href="/(tabs)/scanner" />;
}
