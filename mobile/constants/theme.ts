/**
 * ═══════════════════════════════════
 * 📄 FILE 24/42: mobile/constants/theme.ts
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Design System & Accessible Theme
 * Cohesive visual tokens featuring dark mode, high-contrast accessibility,
 * modern typography scale, and standard sizing constants.
 */

export const COLORS = {
  // Brand Colors (Cyber Obsidian Neon - Vibrant Magenta & Holographic Cyan)
  primary: '#F056C8',      // Ultra-Electric Neon Magenta (brand core glow)
  primaryDark: '#B91C85',  // Deep Magenta shadow
  primaryLight: '#FF88E4', // Soft Magenta aurora edge
  accent: '#06B6D4',       // Holographic Cyber Cyan (guidance & precision actions)
  accentDark: '#0891B2',   // Deep Cyber Cyan shadow
  accentLight: '#67E8F9',  // Laser Mint Cyan edge
  
  // Accessibility & UI Colors (Frosted glassmorphic obsidian optimized)
  background: '#040209',   // Ultimate Pitch Void Obsidian Black
  surface: 'rgba(18, 12, 32, 0.70)',      // Frosted Glass Dark Obsidian
  surfaceLight: 'rgba(32, 20, 56, 0.85)', // Radiant Frosted glass
  surfaceDark: '#07040E',  // Recessed void shadow
  
  // Text Colors (High-fidelity readability contrast)
  text: '#F8FAFC',         // Premium soft stark white
  textSecondary: '#D8B4FE', // Translucent neon lavender
  textMuted: '#9333EA',    // High-contrast accessible deep violet
  
  // Status Colors (Tactile warning indicators)
  success: '#10B981',      // Cybernetic Emerald Green
  warning: '#F59E0B',      // Dynamic Amber Gold
  error: '#EF4444',        // Deep Crimson Alarm Red
  info: '#06B6D4',         // Holographic cyber cyan
  
  // Confidence indicators (Visual overlay colors)
  confHigh: '#10B981',     // Glowing Emerald Green
  confMedium: '#F59E0B',   // Glowing Amber Gold
  confLow: '#EF4444',      // Glowing Crimson Red
 
  // Borders & Dividers (Glowing glass segments)
  border: 'rgba(240, 86, 200, 0.25)',      // Neon magenta glowing border segment
  borderLight: 'rgba(255, 255, 255, 0.08)', // Frosted glass panel separator
  borderCyan: 'rgba(6, 182, 212, 0.30)',    // Holographic cyber cyan border segment
  
  // Overlay colors
  overlay: 'rgba(4, 2, 9, 0.92)',
  overlayLight: 'rgba(255, 255, 255, 0.03)',
};

export const SPACING = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const TYPOGRAPHY = {
  // FontSize Scale
  size: {
    xs: 12,
    sm: 14,
    md: 16,
    lg: 18,
    xl: 20,
    xxl: 24,
    heading: 32,
    largeHeading: 40,
  },
  
  // FontWeight Scale
  weight: {
    regular: '400' as const,
    medium: '500' as const,
    semibold: '600' as const,
    bold: '700' as const,
    heavy: '900' as const,
  },
};

export const SHADOWS = {
  sm: {
    shadowColor: '#F056C8',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 6,
    elevation: 4,
  },
  md: {
    shadowColor: '#06B6D4',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.30,
    shadowRadius: 12,
    elevation: 8,
  },
  lg: {
    shadowColor: '#F056C8',
    shadowOffset: { width: 0, height: 16 },
    shadowOpacity: 0.45,
    shadowRadius: 24,
    elevation: 16,
  },
};

export const BORDER_RADIUS = {
  sm: 10,
  md: 20,
  lg: 30,
  round: 9999,
};

export const THEME = {
  colors: COLORS,
  spacing: SPACING,
  typography: TYPOGRAPHY,
  shadows: SHADOWS,
  borderRadius: BORDER_RADIUS,
};

export default THEME;
