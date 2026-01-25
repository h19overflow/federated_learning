/**
 * Animation Configuration Constants
 * Centralized GSAP animation settings for consistency
 */

export const ANIMATION_CONFIG = {
  duration: 0.8,
  staggerDelay: 0.15,
  ease: "power2.out",
} as const;

export const ANIMATION_TIMINGS = {
  fast: 0.3,
  normal: 0.5,
  slow: 0.8,
} as const;

export const ANIMATION_EASES = {
  easeIn: "power2.in",
  easeOut: "power2.out",
  easeInOut: "power2.inOut",
  back: "back.out(1.7)",
  elastic: "elastic.out(1, 0.5)",
} as const;

/**
 * Hero section entrance animation timeline
 */
export const heroEntranceTimeline = {
  badge: {
    opacity: 0,
    y: 20,
    duration: ANIMATION_CONFIG.duration,
  },
  title: {
    opacity: 0,
    y: 30,
    duration: ANIMATION_CONFIG.duration,
    offset: "-=0.4",
  },
  subtitle: {
    opacity: 0,
    y: 20,
    duration: ANIMATION_CONFIG.duration,
    offset: "-=0.3",
  },
  modeToggle: {
    opacity: 0,
    y: 20,
    duration: ANIMATION_CONFIG.duration,
    offset: "-=0.2",
  },
  contentCard: {
    opacity: 0,
    y: 60,
    stagger: ANIMATION_CONFIG.staggerDelay,
    duration: ANIMATION_CONFIG.duration,
    offset: "-=0.2",
  },
} as const;

/**
 * Result card animation
 */
export const resultCardAnimation = {
  opacity: 0,
  y: 40,
  duration: ANIMATION_CONFIG.duration,
  ease: ANIMATION_CONFIG.ease,
  stagger: ANIMATION_CONFIG.staggerDelay,
} as const;

/**
 * Batch results animation
 */
export const batchResultsAnimation = {
  opacity: 0,
  y: 40,
  duration: ANIMATION_CONFIG.duration,
  ease: ANIMATION_CONFIG.ease,
  stagger: ANIMATION_CONFIG.staggerDelay,
} as const;

/**
 * Upload area reset animation
 */
export const uploadAreaResetAnimation = {
  opacity: 0,
  scale: 0.95,
  duration: 0.5,
  ease: ANIMATION_EASES.back,
} as const;

/**
 * Check if user prefers reduced motion
 */
export const prefersReducedMotion = (): boolean => {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
};
