import { useEffect, RefObject } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

const ANIMATION_CONFIG = {
  duration: 0.8,
  staggerDelay: 0.15,
  ease: "power2.out",
  scrollTriggerStart: "top 80%",
} as const;

export const useLandingAnimations = (mainRef: RefObject<HTMLElement | null>) => {
  useEffect(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    const scrollContainer = mainRef.current;
    if (!scrollContainer) return;

    ScrollTrigger.defaults({ scroller: scrollContainer });
    ScrollTrigger.refresh();

    const ctx = gsap.context(() => {
      // Hero Section Animations
      const heroTimeline = gsap.timeline({ defaults: { ease: ANIMATION_CONFIG.ease } });

      heroTimeline.from(".hero-trust-badge", { opacity: 0, y: 20, duration: ANIMATION_CONFIG.duration });
      heroTimeline.from(".hero-headline .word", { opacity: 0, y: 30, stagger: 0.1, duration: 0.6 }, "-=0.4");
      heroTimeline.from(".hero-subheadline", { opacity: 0, y: 20, duration: ANIMATION_CONFIG.duration }, "-=0.2");
      heroTimeline.from(".hero-cta-button", { opacity: 0, y: 30, stagger: ANIMATION_CONFIG.staggerDelay, duration: ANIMATION_CONFIG.duration }, "-=0.3");
      heroTimeline.from(".hero-scroll-indicator", { opacity: 0, y: -10, duration: ANIMATION_CONFIG.duration }, "-=0.2");

      // Features Section
      gsap.from(".features-title", {
        scrollTrigger: { trigger: ".features-section", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, y: 40, duration: ANIMATION_CONFIG.duration, ease: ANIMATION_CONFIG.ease,
      });
      gsap.from(".feature-card", {
        scrollTrigger: { trigger: ".features-grid", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, y: 60, stagger: ANIMATION_CONFIG.staggerDelay, duration: ANIMATION_CONFIG.duration, ease: ANIMATION_CONFIG.ease,
      });

      // Comparison Section
      gsap.from(".comparison-title", {
        scrollTrigger: { trigger: ".comparison-section", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, y: 40, duration: ANIMATION_CONFIG.duration, ease: ANIMATION_CONFIG.ease,
      });
      gsap.from(".centralized-card", {
        scrollTrigger: { trigger: ".comparison-cards", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, x: -80, duration: 1, ease: "power3.out",
      });
      gsap.from(".federated-card", {
        scrollTrigger: { trigger: ".comparison-cards", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, x: 80, duration: 1, ease: "power3.out",
      });
      gsap.from(".comparison-table-wrapper", {
        scrollTrigger: { trigger: ".comparison-table-wrapper", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, y: 40, duration: ANIMATION_CONFIG.duration, ease: ANIMATION_CONFIG.ease,
      });
      gsap.from(".comparison-table-row", {
        scrollTrigger: { trigger: ".comparison-table-wrapper", start: "top 70%", toggleActions: "play none none none" },
        opacity: 0, x: -20, stagger: 0.1, duration: 0.5, ease: ANIMATION_CONFIG.ease,
      });

      // How It Works Section
      gsap.from(".how-it-works-title", {
        scrollTrigger: { trigger: ".how-it-works-section", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, y: 40, duration: ANIMATION_CONFIG.duration, ease: ANIMATION_CONFIG.ease,
      });
      gsap.from(".step-card", {
        scrollTrigger: { trigger: ".how-it-works-section", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, y: 60, stagger: ANIMATION_CONFIG.staggerDelay, duration: ANIMATION_CONFIG.duration, ease: ANIMATION_CONFIG.ease,
      });
      gsap.from(".step-icon", {
        scrollTrigger: { trigger: ".how-it-works-section", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        scale: 0, stagger: ANIMATION_CONFIG.staggerDelay, duration: 0.6, ease: "back.out(1.7)", delay: 0.3,
      });

      // CTA Section
      gsap.from(".cta-icon", {
        scrollTrigger: { trigger: ".cta-section", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, rotation: 90, scale: 0.5, duration: 1, ease: "back.out(1.7)",
      });
      gsap.from(".cta-content", {
        scrollTrigger: { trigger: ".cta-section", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, y: 30, duration: ANIMATION_CONFIG.duration, ease: ANIMATION_CONFIG.ease, delay: 0.3,
      });
      gsap.from(".cta-button", {
        scrollTrigger: { trigger: ".cta-section", start: ANIMATION_CONFIG.scrollTriggerStart, toggleActions: "play none none none" },
        opacity: 0, y: 30, duration: ANIMATION_CONFIG.duration, ease: "back.out(1.7)", delay: 0.5,
      });
    });

    return () => {
      ctx.revert();
      ScrollTrigger.defaults({ scroller: window });
    };
  }, [mainRef]);
};
