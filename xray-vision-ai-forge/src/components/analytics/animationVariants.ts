import { Variants } from "framer-motion";

/**
 * Container animation variant for staggered children animations.
 * Used on the main analytics container to orchestrate sequential appearance of child elements.
 * Creates a fade-in effect with staggered children appearing 0.1s apart.
 */
export const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
};

/**
 * Item animation variant for individual elements within containers.
 * Used for filter sections, charts, and cards to create a slide-up fade-in effect.
 * Applies spring physics for smooth, natural motion.
 */
export const itemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 15,
    },
  },
};

/**
 * Card hover animation variant for interactive cards.
 * Used on stat cards and chart containers to provide visual feedback on hover.
 * Creates a subtle lift and scale effect when user hovers over cards.
 */
export const cardHoverVariants: Variants = {
  rest: { scale: 1, y: 0 },
  hover: {
    scale: 1.02,
    y: -4,
    transition: {
      type: "spring",
      stiffness: 400,
      damping: 25,
    },
  },
};

/**
 * Table row animation variant for list items with custom delays.
 * Used in comparison tables and top runs table to create sequential row animations.
 * Accepts a custom index parameter to calculate staggered delays (0.05s per row).
 */
export const tableRowVariants: Variants = {
  hidden: { opacity: 0, x: -20 },
  visible: (i: number) => ({
    opacity: 1,
    x: 0,
    transition: {
      delay: i * 0.05,
      type: "spring",
      stiffness: 100,
      damping: 15,
    },
  }),
};
