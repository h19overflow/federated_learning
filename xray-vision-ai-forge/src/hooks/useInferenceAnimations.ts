/**
 * Inference Animations Hook
 * Manages GSAP animations for the inference page
 */

import { useEffect, useRef } from "react";
import gsap from "gsap";
import {
  prefersReducedMotion,
  heroEntranceTimeline,
  ANIMATION_CONFIG,
} from "@/utils/animationConfigs";

export const useInferenceAnimations = () => {
  const contextRef = useRef<gsap.Context | null>(null);

  /**
   * Play hero section entrance animation
   */
  const playHeroEntrance = () => {
    if (prefersReducedMotion()) return;

    contextRef.current = gsap.context(() => {
      const timeline = gsap.timeline({
        defaults: { ease: "expo.out" },
      });

      // Initial state for elements to prevent flash
      gsap.set([".hero-badge", ".hero-title", ".hero-subtitle", ".hero-mode-toggle", ".content-card"], {
        opacity: 0,
        y: 30,
      });

      timeline
        .to(".hero-badge", {
          opacity: 1,
          y: 0,
          duration: 1,
          ease: "back.out(1.7)",
        })
        .to(
          ".hero-title",
          {
            opacity: 1,
            y: 0,
            duration: 1.2,
          },
          "-=0.7"
        )
        .to(
          ".hero-subtitle",
          {
            opacity: 1,
            y: 0,
            duration: 1,
          },
          "-=0.8"
        )
        .to(
          ".hero-mode-toggle",
          {
            opacity: 1,
            y: 0,
            duration: 0.8,
            ease: "back.out(1.2)",
          },
          "-=0.6"
        )
        .to(
          ".content-card",
          {
            opacity: 1,
            y: 0,
            duration: 1,
            stagger: 0.1,
          },
          "-=0.4"
        );

      // Add a subtle floating effect to the hero section
      gsap.to(".hero-title", {
        y: -5,
        duration: 3,
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut",
        delay: 1.5,
      });
    });
  };

  /**
   * Cleanup animations on unmount
   */
  const cleanup = () => {
    if (contextRef.current) {
      contextRef.current.revert();
      contextRef.current = null;
    }
  };

  return {
    playHeroEntrance,
    cleanup,
  };
};
