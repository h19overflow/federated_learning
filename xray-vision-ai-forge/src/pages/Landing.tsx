import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import {
  ArrowRight,
  Shield,
  Lock,
  Activity,
  CheckCircle2,
  ChevronDown
} from 'lucide-react';
import { Header, Footer, WelcomeGuide } from '@/components/layout';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

// Register GSAP plugins
gsap.registerPlugin(ScrollTrigger);

// Animation configuration constants
const ANIMATION_CONFIG = {
  duration: 0.8,
  staggerDelay: 0.15,
  ease: 'power2.out',
  scrollTriggerStart: 'top 80%',
} as const;

const Landing = () => {
  const navigate = useNavigate();
  const comparisonRef = useRef<HTMLElement>(null);
  const mainRef = useRef<HTMLElement>(null); // Scroll container ref
  const [showWelcomeGuide, setShowWelcomeGuide] = useState(false);

  const handleGetStarted = () => {
    navigate('/experiment');
  };

  const scrollToComparison = () => {
    comparisonRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // GSAP Scroll-Triggered Animations
  useEffect(() => {
    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    // Wait for DOM to be ready and refs to be set
    const scrollContainer = mainRef.current;
    if (!scrollContainer) return;

    // Configure ScrollTrigger to use the custom scroll container
    ScrollTrigger.defaults({
      scroller: scrollContainer,
    });

    // Refresh ScrollTrigger after setting scroller
    ScrollTrigger.refresh();

    const ctx = gsap.context(() => {
      // Hero Section Animations - runs immediately on load
      const heroTimeline = gsap.timeline({ defaults: { ease: ANIMATION_CONFIG.ease } });

      // Trust badge: fade in + slide up
      heroTimeline.from('.hero-trust-badge', {
        opacity: 0,
        y: 20,
        duration: ANIMATION_CONFIG.duration,
      });

      // Main headline: staggered word reveal
      heroTimeline.from('.hero-headline .word', {
        opacity: 0,
        y: 30,
        stagger: 0.1,
        duration: 0.6,
      }, '-=0.4');

      // Subheadline: fade in after headline
      heroTimeline.from('.hero-subheadline', {
        opacity: 0,
        y: 20,
        duration: ANIMATION_CONFIG.duration,
      }, '-=0.2');

      // CTA buttons: slide up with stagger
      heroTimeline.from('.hero-cta-button', {
        opacity: 0,
        y: 30,
        stagger: ANIMATION_CONFIG.staggerDelay,
        duration: ANIMATION_CONFIG.duration,
      }, '-=0.3');

      // Scroll indicator: gentle bounce/pulse (delayed entrance)
      heroTimeline.from('.hero-scroll-indicator', {
        opacity: 0,
        y: -10,
        duration: ANIMATION_CONFIG.duration,
      }, '-=0.2');

      // Features Section - ScrollTriggered
      gsap.from('.features-title', {
        scrollTrigger: {
          trigger: '.features-section',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        y: 40,
        duration: ANIMATION_CONFIG.duration,
        ease: ANIMATION_CONFIG.ease,
      });

      gsap.from('.feature-card', {
        scrollTrigger: {
          trigger: '.features-grid',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        y: 60,
        stagger: ANIMATION_CONFIG.staggerDelay,
        duration: ANIMATION_CONFIG.duration,
        ease: ANIMATION_CONFIG.ease,
      });

      // Comparison Section - ScrollTriggered
      gsap.from('.comparison-title', {
        scrollTrigger: {
          trigger: '.comparison-section',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        y: 40,
        duration: ANIMATION_CONFIG.duration,
        ease: ANIMATION_CONFIG.ease,
      });

      // Centralized card: slide in from left
      gsap.from('.centralized-card', {
        scrollTrigger: {
          trigger: '.comparison-cards',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        x: -80,
        duration: 1,
        ease: 'power3.out',
      });

      // Federated card: slide in from right
      gsap.from('.federated-card', {
        scrollTrigger: {
          trigger: '.comparison-cards',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        x: 80,
        duration: 1,
        ease: 'power3.out',
      });

      // Comparison table: fade in rows sequentially
      gsap.from('.comparison-table-wrapper', {
        scrollTrigger: {
          trigger: '.comparison-table-wrapper',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        y: 40,
        duration: ANIMATION_CONFIG.duration,
        ease: ANIMATION_CONFIG.ease,
      });

      gsap.from('.comparison-table-row', {
        scrollTrigger: {
          trigger: '.comparison-table-wrapper',
          start: 'top 70%',
          toggleActions: 'play none none none',
        },
        opacity: 0,
        x: -20,
        stagger: 0.1,
        duration: 0.5,
        ease: ANIMATION_CONFIG.ease,
      });

      // How It Works Section - ScrollTriggered
      gsap.from('.how-it-works-title', {
        scrollTrigger: {
          trigger: '.how-it-works-section',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        y: 40,
        duration: ANIMATION_CONFIG.duration,
        ease: ANIMATION_CONFIG.ease,
      });

      // Step cards: sequential reveal
      gsap.from('.step-card', {
        scrollTrigger: {
          trigger: '.how-it-works-section',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        y: 60,
        stagger: ANIMATION_CONFIG.staggerDelay,
        duration: ANIMATION_CONFIG.duration,
        ease: ANIMATION_CONFIG.ease,
      });

      // Step icons: scale up animation
      gsap.from('.step-icon', {
        scrollTrigger: {
          trigger: '.how-it-works-section',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        scale: 0,
        stagger: ANIMATION_CONFIG.staggerDelay,
        duration: 0.6,
        ease: 'back.out(1.7)',
        delay: 0.3,
      });

      // CTA Section - ScrollTriggered
      gsap.from('.cta-icon', {
        scrollTrigger: {
          trigger: '.cta-section',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        rotation: 90,
        scale: 0.5,
        duration: 1,
        ease: 'back.out(1.7)',
      });

      gsap.from('.cta-content', {
        scrollTrigger: {
          trigger: '.cta-section',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        y: 30,
        duration: ANIMATION_CONFIG.duration,
        ease: ANIMATION_CONFIG.ease,
        delay: 0.3,
      });

      gsap.from('.cta-button', {
        scrollTrigger: {
          trigger: '.cta-section',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        opacity: 0,
        y: 30,
        duration: ANIMATION_CONFIG.duration,
        ease: 'back.out(1.7)',
        delay: 0.5,
      });
    });

    return () => {
      ctx.revert(); // Cleanup all GSAP animations
      ScrollTrigger.defaults({ scroller: window }); // Reset scroller default
    };
  }, []);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {showWelcomeGuide && (
        <WelcomeGuide onClose={() => setShowWelcomeGuide(false)} />
      )}

      <Header onShowHelp={() => setShowWelcomeGuide(true)} />

      <main ref={mainRef} className="flex-1 overflow-y-auto bg-hero-gradient">
        {/* Hero Section - Apple Style */}
        <section className="relative min-h-[90vh] flex flex-col items-center justify-center px-6 overflow-hidden">
          {/* Subtle background elements */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-[hsl(172_40%_85%)] rounded-full blur-[120px] opacity-30" />
            <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-[hsl(210_60%_90%)] rounded-full blur-[100px] opacity-25" />
          </div>

          {/* Hero AI Lung Visualization - Right side with soft rounded edges */}
          <div className="absolute right-[-5%] top-1/2 -translate-y-1/2 w-[50%] h-[80%] pointer-events-none hidden lg:block">
            <div
              className="relative w-full h-full"
              style={{
                maskImage: 'radial-gradient(ellipse 80% 70% at 70% 50%, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.4) 50%, transparent 80%)',
                WebkitMaskImage: 'radial-gradient(ellipse 80% 70% at 70% 50%, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.4) 50%, transparent 80%)'
              }}
            >
              {/* Image container - crops bottom-right Gemini logo */}
              <div className="absolute inset-0 overflow-hidden">
                <img
                  src="/images/Lungs.png"
                  alt="AI-powered neural network visualization of lung analysis"
                  className="w-[115%] h-[115%] object-cover object-left-top opacity-50"
                />
              </div>
              {/* Teal tint overlay for brand cohesion */}
              <div className="absolute inset-0 bg-[hsl(172_50%_60%)]/10 mix-blend-overlay" />
            </div>
          </div>

          <div className="relative z-10 max-w-4xl mx-auto text-center">
            {/* Trust badge */}
            <div className="hero-trust-badge inline-flex items-center gap-2 px-4 py-2 mb-8 rounded-full bg-white/60 backdrop-blur-sm border border-[hsl(172_30%_85%)] shadow-sm">
              <div className="w-2 h-2 rounded-full bg-[hsl(152_60%_42%)] animate-pulse" />
              <span className="text-sm font-medium text-[hsl(172_43%_25%)]">
                Powered by ResNet50 V2 & Flower Framework
              </span>
            </div>

            {/* Main headline */}
            <h1 className="hero-headline text-5xl md:text-7xl font-semibold tracking-tight text-[hsl(172_43%_15%)] mb-6">
              <span className="word inline-block">Medical</span>{' '}
              <span className="word inline-block">AI,</span>
              <br />
              <span className="word inline-block text-[hsl(172_63%_28%)]">Refined.</span>
            </h1>

            {/* Subheadline */}
            <p className="hero-subheadline text-xl md:text-2xl text-[hsl(215_15%_45%)] font-light max-w-2xl mx-auto mb-12 leading-relaxed">
              Train state-of-the-art pneumonia detection models with
              <span className="font-medium text-[hsl(172_43%_25%)]"> Centralized </span>
              or
              <span className="font-medium text-[hsl(172_43%_25%)]"> Federated Learning</span>.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                size="lg"
                className="hero-cta-button bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white text-lg px-10 py-7 rounded-2xl shadow-lg shadow-[hsl(172_63%_22%)]/20 transition-all duration-300 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/30 hover:-translate-y-0.5"
                onClick={handleGetStarted}
              >
                Start Training
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="hero-cta-button text-lg px-10 py-7 rounded-2xl border-2 border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)] transition-all duration-300"
                onClick={scrollToComparison}
              >
                Learn More
              </Button>
            </div>
          </div>

          {/* Scroll indicator */}
          <div className="hero-scroll-indicator absolute bottom-10 left-1/2 -translate-x-1/2">
            <button
              onClick={scrollToComparison}
              className="flex flex-col items-center gap-2 text-[hsl(215_15%_55%)] hover:text-[hsl(172_63%_28%)] transition-colors"
            >
              <span className="text-sm font-medium">Explore</span>
              <ChevronDown className="h-5 w-5 animate-bounce" />
            </button>
          </div>
        </section>

        {/* Features Section - Minimal Grid */}
        <section className="features-section py-32 px-6 bg-white relative overflow-hidden">
          {/* Subtle hexagon pattern background */}
          <div
            className="absolute inset-0 opacity-[0.35] pointer-events-none"
            style={{
              backgroundImage: 'url(/images/hexagon_grid.png)',
              backgroundSize: '400px 400px',
              backgroundRepeat: 'repeat',
              maskImage: 'radial-gradient(ellipse at center, rgba(0,0,0,0.5) 0%, transparent 70%)',
              WebkitMaskImage: 'radial-gradient(ellipse at center, rgba(0,0,0,0.5) 0%, transparent 70%)'
            }}
          />
          <div className="max-w-6xl mx-auto relative z-10">
            <div className="features-title text-center mb-20">
              <h2 className="text-4xl md:text-5xl font-semibold text-[hsl(172_43%_15%)] mb-6">
                Why Choose Our Platform?
              </h2>
              <p className="text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto">
                Enterprise-grade AI training with institutional trust.
              </p>
            </div>

            <div className="features-grid grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  icon: (
                    <svg className="w-10 h-10" viewBox="0 0 40 40" fill="none">
                      <circle cx="20" cy="20" r="18" stroke="hsl(172 63% 22%)" strokeWidth="2" fill="hsl(168 40% 95%)" />
                      <path d="M12 20c0-4.4 3.6-8 8-8s8 3.6 8 8-3.6 8-8 8" stroke="hsl(172 63% 22%)" strokeWidth="2" strokeLinecap="round" />
                      <circle cx="20" cy="20" r="3" fill="hsl(172 63% 22%)" />
                    </svg>
                  ),
                  title: 'Advanced AI Models',
                  description: 'Pre-trained ResNet50 V2 architecture, fine-tuned for medical imaging with exceptional accuracy.'
                },
                {
                  icon: <Shield className="w-10 h-10 text-[hsl(172_63%_22%)]" />,
                  title: 'Privacy-Preserving',
                  description: 'Federated learning ensures patient data remains private and secure, never leaving local devices.'
                },
                {
                  icon: <Activity className="w-10 h-10 text-[hsl(172_63%_22%)]" />,
                  title: 'Real-Time Monitoring',
                  description: 'Watch training metrics and performance live with elegant, interactive visualizations.'
                }
              ].map((feature, index) => (
                <div
                  key={index}
                  className="feature-card group p-8 rounded-3xl bg-[hsl(168_25%_98%)] border border-[hsl(168_20%_92%)] hover:bg-white hover:shadow-xl hover:shadow-[hsl(172_40%_85%)]/30 transition-all duration-500 hover:-translate-y-1"
                >
                  <div className="mb-6 p-4 rounded-2xl bg-white inline-block shadow-sm group-hover:shadow-md transition-shadow">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)] mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-[hsl(215_15%_45%)] leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Comparison Section - Clean & Sophisticated */}
        <section ref={comparisonRef} className="comparison-section py-32 px-6 bg-trust-gradient relative overflow-hidden">
          <div className="absolute inset-0 noise-overlay" />
          {/* Subtle hexagon pattern overlay */}
          <div
            className="absolute inset-0 opacity-20 pointer-events-none"
            style={{
              backgroundImage: 'url(/images/hexagon_grid.png)',
              backgroundSize: '350px 350px',
              backgroundRepeat: 'repeat',
              mixBlendMode: 'multiply'
            }}
          />

          <div className="relative z-10 max-w-6xl mx-auto">
            <div className="comparison-title text-center mb-20">
              <h2 className="text-4xl md:text-5xl font-semibold text-[hsl(172_43%_15%)] mb-6">
                Choose Your Approach
              </h2>
              <p className="text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto">
                Two powerful methodologies, one exceptional outcome.
              </p>
            </div>

            <div className="comparison-cards grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Centralized Learning Card */}
              <div className="centralized-card relative group">
                <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-[2rem] blur-xl opacity-50 group-hover:opacity-70 transition-opacity" />
                <div className="relative bg-white/90 backdrop-blur-sm rounded-[2rem] p-10 border border-[hsl(172_30%_88%)] shadow-lg hover:shadow-2xl transition-all duration-500">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-8">
                    <div>
                      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[hsl(172_40%_94%)] text-[hsl(172_50%_35%)] text-sm font-medium mb-4">
                        Traditional
                      </div>
                      <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)]">
                        Centralized Learning
                      </h3>
                    </div>
                    <div className="p-3 rounded-2xl bg-[hsl(172_40%_94%)]">
                      <svg className="w-8 h-8 text-[hsl(172_50%_35%)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <rect x="4" y="4" width="16" height="16" rx="2" />
                        <path d="M4 9h16M9 4v16" />
                      </svg>
                    </div>
                  </div>

                  {/* Centralized Learning Illustration */}
                  <div className="centralized-diagram relative h-64 mb-8 rounded-2xl bg-gradient-to-br from-[hsl(172_30%_97%)] to-[hsl(168_40%_93%)] overflow-hidden">
                    {/* Real centralized learning image */}
                    <img
                      src="/images/centralizied.png"
                      alt="Centralized learning diagram showing hospitals sending data to central server with warning indicators"
                      className="absolute inset-0 w-full h-full object-contain p-2"
                    />
                    {/* Subtle overlay for blending */}
                    <div className="absolute inset-0 bg-gradient-to-t from-[hsl(168_40%_93%)]/60 via-transparent to-transparent" />
                  </div>

                  {/* Benefits */}
                  <div className="space-y-4">
                    <h4 className="text-sm font-semibold text-[hsl(152_60%_35%)] uppercase tracking-wide flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4" />
                      Advantages
                    </h4>
                    <ul className="space-y-2">
                      {['Faster training time', 'Simpler implementation', 'Easier debugging', 'Ideal for small datasets'].map((item, i) => (
                        <li key={i} className="flex items-center gap-3 text-[hsl(215_15%_40%)]">
                          <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)]" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Considerations */}
                  <div className="mt-6 pt-6 border-t border-[hsl(172_20%_92%)] space-y-4">
                    <h4 className="text-sm font-semibold text-[hsl(35_70%_45%)] uppercase tracking-wide flex items-center gap-2">
                      <Lock className="w-4 h-4" />
                      Considerations
                    </h4>
                    <ul className="space-y-2">
                      {['Requires data centralization', 'Privacy concerns with sensitive data', 'Single point of failure'].map((item, i) => (
                        <li key={i} className="flex items-center gap-3 text-[hsl(215_15%_50%)]">
                          <span className="w-1.5 h-1.5 rounded-full bg-[hsl(35_70%_50%)]" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              {/* Federated Learning Card */}
              <div className="federated-card relative group">
                <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_90%)] to-[hsl(168_40%_85%)] rounded-[2rem] blur-xl opacity-50 group-hover:opacity-70 transition-opacity" />
                <div className="relative bg-white/90 backdrop-blur-sm rounded-[2rem] p-10 border border-[hsl(172_30%_88%)] shadow-lg hover:shadow-2xl transition-all duration-500">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-8">
                    <div>
                      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[hsl(172_40%_92%)] text-[hsl(172_63%_28%)] text-sm font-medium mb-4">
                        Privacy-First
                      </div>
                      <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)]">
                        Federated Learning
                      </h3>
                    </div>
                    <div className="p-3 rounded-2xl bg-[hsl(172_40%_94%)]">
                      <svg className="w-8 h-8 text-[hsl(172_63%_28%)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <circle cx="12" cy="12" r="3" />
                        <circle cx="5" cy="5" r="2" />
                        <circle cx="19" cy="5" r="2" />
                        <circle cx="5" cy="19" r="2" />
                        <circle cx="19" cy="19" r="2" />
                        <path d="M12 9V7M12 17v-2M9 12H7m10 0h-2" />
                        <path d="M6.5 6.5l3 3m5 5l3 3M17.5 6.5l-3 3m-5 5l-3 3" />
                      </svg>
                    </div>
                  </div>

                  {/* Federated Learning Illustration */}
                  <div className="federated-diagram relative h-64 mb-8 rounded-2xl bg-gradient-to-br from-[hsl(172_30%_97%)] to-[hsl(168_40%_93%)] overflow-hidden">
                    {/* Real distributed learning image */}
                    <img
                      src="/images/distributed_learning.png"
                      alt="Federated learning diagram showing hospitals with privacy shields connected to central aggregation server"
                      className="absolute inset-0 w-full h-full object-contain p-2"
                    />
                    {/* Subtle overlay for blending */}
                    <div className="absolute inset-0 bg-gradient-to-t from-[hsl(168_40%_93%)]/60 via-transparent to-transparent" />

                    {/* Hidden SVG - kept for animation particles only */}
                    <svg viewBox="0 0 300 200" className="absolute inset-0 w-full h-full opacity-0 pointer-events-none" aria-hidden="true">
                      <defs>
                        {/* Gradient for upward gradient packets */}
                        <linearGradient id="gradientUp" x1="0%" y1="100%" x2="0%" y2="0%">
                          <stop offset="0%" stopColor="hsl(172 60% 50%)" />
                          <stop offset="100%" stopColor="hsl(172 60% 35%)" />
                        </linearGradient>

                        {/* Gradient for downward model updates */}
                        <linearGradient id="gradientDown" x1="0%" y1="0%" x2="0%" y2="100%">
                          <stop offset="0%" stopColor="hsl(152 60% 50%)" />
                          <stop offset="100%" stopColor="hsl(152 60% 35%)" />
                        </linearGradient>

                        {/* Glow for server */}
                        <filter id="glowGreen">
                          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                          <feMerge>
                            <feMergeNode in="coloredBlur"/>
                            <feMergeNode in="SourceGraphic"/>
                          </feMerge>
                        </filter>
                      </defs>

                      {/* Global server at center top */}
                      <g transform="translate(150, 45)">
                        {/* Pulsing glow */}
                        <circle className="pulse-ring-fed" r="30" fill="hsl(172 45% 85%)" opacity="0.5">
                          <animate attributeName="r" values="30;36;30" dur="2s" repeatCount="indefinite" />
                          <animate attributeName="opacity" values="0.5;0.25;0.5" dur="2s" repeatCount="indefinite" />
                        </circle>
                        {/* Server */}
                        <rect x="-35" y="-18" width="70" height="36" rx="8" fill="white" stroke="hsl(172 50% 65%)" strokeWidth="2.5" filter="url(#glowGreen)" />
                        {/* Aggregation icon */}
                        <circle cx="0" cy="0" r="8" fill="none" stroke="hsl(172 63% 35%)" strokeWidth="1.5" />
                        <circle cx="-10" cy="-6" r="3" fill="hsl(172 63% 35%)" opacity="0.6" />
                        <circle cx="10" cy="-6" r="3" fill="hsl(172 63% 35%)" opacity="0.6" />
                        <circle cx="0" cy="8" r="3" fill="hsl(172 63% 35%)" opacity="0.6" />
                        <text x="0" y="25" textAnchor="middle" fontSize="10" fontWeight="600" fill="hsl(172 43% 25%)">Global Server</text>
                      </g>

                      {/* Hospital nodes with lock indicators */}
                      {[0, 1, 2].map((i) => (
                        <g key={i} transform={`translate(${50 + i * 100}, 155)`} className="hospital-node">
                          {/* Hospital building */}
                          <rect x="-18" y="-18" width="36" height="36" rx="6" fill="white" stroke="hsl(172 40% 75%)" strokeWidth="2" />
                          {/* Database icon */}
                          <path d="M-10,-10 L10,-10 M-10,-3 L10,-3 M-10,4 L10,4" stroke="hsl(172 50% 40%)" strokeWidth="1.5" strokeLinecap="round" />

                          {/* Lock indicator (data stays local) */}
                          <g transform="translate(12, -12)">
                            <circle r="7" fill="hsl(152 60% 42%)" />
                            {/* Lock icon */}
                            <path d="M-2,0 L-2,-2 A2,2 0 0,1 2,-2 L2,0" stroke="white" strokeWidth="1.2" fill="none" strokeLinecap="round" />
                            <rect x="-2.5" y="0" width="5" height="4" rx="0.5" fill="white" />
                          </g>

                          {/* Data stays here indicator */}
                          <circle cx="0" cy="6" r="2" fill="hsl(152 60% 42%)" className="animate-pulse" />
                        </g>
                      ))}

                      {/* Bidirectional flow paths */}
                      {[0, 1, 2].map((pathIndex) => {
                        const startX = 50 + pathIndex * 100;
                        const startY = 137;
                        const endX = 150;
                        const endY = 63;

                        return (
                          <g key={pathIndex}>
                            {/* Upward path (gradients to server) - offset left */}
                            <line
                              x1={startX - 6}
                              y1={startY}
                              x2={endX - 6}
                              y2={endY}
                              stroke="hsl(172 50% 75%)"
                              strokeWidth="2"
                              strokeDasharray="5 5"
                              opacity="0.4"
                            />

                            {/* Downward path (model updates back) - offset right */}
                            <line
                              x1={endX + 6}
                              y1={endY}
                              x2={startX + 6}
                              y2={startY}
                              stroke="hsl(152 50% 70%)"
                              strokeWidth="2"
                              strokeDasharray="5 5"
                              opacity="0.4"
                            />

                            {/* Upward gradient particles (local gradients) */}
                            {[0, 1].map((particleIndex) => (
                              <circle
                                key={`up-${particleIndex}`}
                                className={`gradient-particle-up gradient-particle-up-${pathIndex}-${particleIndex}`}
                                r="3.5"
                                fill="url(#gradientUp)"
                                stroke="white"
                                strokeWidth="1"
                                strokeDasharray="2 1"
                                opacity="0"
                              >
                                <animateMotion
                                  dur="3s"
                                  repeatCount="indefinite"
                                  begin={`${pathIndex * 0.4 + particleIndex * 1.5}s`}
                                  path={`M ${startX - 6} ${startY} L ${endX - 6} ${endY}`}
                                />
                                <animate
                                  attributeName="opacity"
                                  values="0;1;1;0"
                                  keyTimes="0;0.1;0.85;1"
                                  dur="3s"
                                  repeatCount="indefinite"
                                  begin={`${pathIndex * 0.4 + particleIndex * 1.5}s`}
                                />
                              </circle>
                            ))}

                            {/* Downward model update particles */}
                            {[0, 1].map((particleIndex) => (
                              <g key={`down-${particleIndex}`}>
                                <circle
                                  className={`model-particle-down model-particle-down-${pathIndex}-${particleIndex}`}
                                  r="3.5"
                                  fill="url(#gradientDown)"
                                  stroke="white"
                                  strokeWidth="1"
                                  opacity="0"
                                >
                                  <animateMotion
                                    dur="3s"
                                    repeatCount="indefinite"
                                    begin={`${pathIndex * 0.4 + particleIndex * 1.5 + 0.3}s`}
                                    path={`M ${endX + 6} ${endY} L ${startX + 6} ${startY}`}
                                  />
                                  <animate
                                    attributeName="opacity"
                                    values="0;1;1;0"
                                    keyTimes="0;0.1;0.85;1"
                                    dur="3s"
                                    repeatCount="indefinite"
                                    begin={`${pathIndex * 0.4 + particleIndex * 1.5 + 0.3}s`}
                                  />
                                </circle>
                                {/* Model icon inside particle */}
                                <g opacity="0">
                                  <circle r="2" fill="white" opacity="0.9" />
                                  <path d="M-1,-1 L1,1 M1,-1 L-1,1" stroke="hsl(152 60% 35%)" strokeWidth="0.8" />
                                  <animateMotion
                                    dur="3s"
                                    repeatCount="indefinite"
                                    begin={`${pathIndex * 0.4 + particleIndex * 1.5 + 0.3}s`}
                                    path={`M ${endX + 6} ${endY} L ${startX + 6} ${startY}`}
                                  />
                                  <animate
                                    attributeName="opacity"
                                    values="0;0.9;0.9;0"
                                    keyTimes="0;0.1;0.85;1"
                                    dur="3s"
                                    repeatCount="indefinite"
                                    begin={`${pathIndex * 0.4 + particleIndex * 1.5 + 0.3}s`}
                                  />
                                </g>
                              </g>
                            ))}
                          </g>
                        );
                      })}

                      {/* Directional arrows */}
                      {[0, 1, 2].map((i) => {
                        const startX = 50 + i * 100;
                        return (
                          <g key={i} opacity="0.5">
                            {/* Upward arrow */}
                            <polygon
                              points={`${startX - 6},${95} ${startX - 9},${100} ${startX - 3},${100}`}
                              fill="hsl(172 50% 60%)"
                            />
                            {/* Downward arrow */}
                            <polygon
                              points={`${startX + 6},${100} ${startX + 3},${95} ${startX + 9},${95}`}
                              fill="hsl(152 50% 55%)"
                            />
                          </g>
                        );
                      })}

                      {/* Federated Learning Step Labels */}
                      {/* Step 1: Local Training - at hospital nodes */}
                      <g transform="translate(50, 185)">
                        <rect x="-28" y="-8" width="56" height="16" rx="4" fill="hsl(172 40% 92%)" stroke="hsl(172 50% 75%)" strokeWidth="1" />
                        <text x="0" y="4" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(172 43% 30%)">1. Local Train</text>
                      </g>

                      {/* Step 2: Send Gradients - on the left upward path */}
                      <g transform="translate(25, 100)">
                        <rect x="-30" y="-8" width="60" height="16" rx="4" fill="hsl(172 45% 90%)" stroke="hsl(172 50% 70%)" strokeWidth="1" />
                        <text x="0" y="4" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(172 50% 28%)">2. Send Grads</text>
                      </g>

                      {/* Step 3: Aggregate - at the global server */}
                      <g transform="translate(150, 12)">
                        <rect x="-26" y="-7" width="52" height="14" rx="4" fill="hsl(152 50% 88%)" stroke="hsl(152 55% 65%)" strokeWidth="1" />
                        <text x="0" y="3" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(152 60% 28%)">3. Aggregate</text>
                      </g>

                      {/* Step 4: Send Updated Model - on the right downward path */}
                      <g transform="translate(275, 100)">
                        <rect x="-35" y="-8" width="70" height="16" rx="4" fill="hsl(152 45% 90%)" stroke="hsl(152 55% 70%)" strokeWidth="1" />
                        <text x="0" y="4" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(152 55% 28%)">4. Send Model</text>
                      </g>

                      {/* Cycle indicator - shows it's a repeating process */}
                      <g transform="translate(150, 185)">
                        <rect x="-22" y="-8" width="44" height="16" rx="4" fill="hsl(172 35% 95%)" stroke="hsl(172 40% 80%)" strokeWidth="1" strokeDasharray="3 2" />
                        <text x="0" y="4" textAnchor="middle" fontSize="7" fontWeight="500" fill="hsl(172 40% 40%)">↻ Repeat</text>
                      </g>
                    </svg>
                  </div>

                  {/* Benefits */}
                  <div className="space-y-4">
                    <h4 className="text-sm font-semibold text-[hsl(152_60%_35%)] uppercase tracking-wide flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4" />
                      Advantages
                    </h4>
                    <ul className="space-y-2">
                      {['Data privacy preserved', 'HIPAA/GDPR compliant', 'Distributed computation', 'Highly scalable'].map((item, i) => (
                        <li key={i} className="flex items-center gap-3 text-[hsl(215_15%_40%)]">
                          <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)]" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Considerations */}
                  <div className="mt-6 pt-6 border-t border-[hsl(172_20%_92%)] space-y-4">
                    <h4 className="text-sm font-semibold text-[hsl(35_70%_45%)] uppercase tracking-wide flex items-center gap-2">
                      <Lock className="w-4 h-4" />
                      Considerations
                    </h4>
                    <ul className="space-y-2">
                      {['Longer training time', 'More complex setup', 'Network overhead'].map((item, i) => (
                        <li key={i} className="flex items-center gap-3 text-[hsl(215_15%_50%)]">
                          <span className="w-1.5 h-1.5 rounded-full bg-[hsl(35_70%_50%)]" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Comparison Table */}
            <div className="comparison-table-wrapper mt-16 bg-white rounded-[2rem] p-8 shadow-lg border border-[hsl(210_15%_92%)]">
              <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)] mb-8 text-center">
                Quick Comparison
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="comparison-table-row border-b border-[hsl(210_15%_92%)]">
                      <th className="py-4 px-6 text-left text-sm font-semibold text-[hsl(172_43%_20%)]">Feature</th>
                      <th className="py-4 px-6 text-center text-sm font-semibold text-[hsl(210_60%_40%)]">Centralized</th>
                      <th className="py-4 px-6 text-center text-sm font-semibold text-[hsl(172_63%_28%)]">Federated</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[hsl(210_15%_95%)]">
                    {[
                      { feature: 'Data Privacy', centralized: { text: 'Data must be shared', status: 'warning' }, federated: { text: 'Data stays local', status: 'success' } },
                      { feature: 'Training Speed', centralized: { text: 'Fast', status: 'success' }, federated: { text: 'Slower', status: 'warning' } },
                      { feature: 'Setup Complexity', centralized: { text: 'Simple', status: 'success' }, federated: { text: 'Complex', status: 'warning' } },
                      { feature: 'Compliance', centralized: { text: 'Requires safeguards', status: 'warning' }, federated: { text: 'Built-in privacy', status: 'success' } },
                      { feature: 'Model Quality', centralized: { text: 'Excellent', status: 'success' }, federated: { text: 'Comparable', status: 'success' } },
                      { feature: 'Scalability', centralized: { text: 'Limited by server', status: 'warning' }, federated: { text: 'Highly scalable', status: 'success' } },
                    ].map((row, index) => (
                      <tr key={index} className="comparison-table-row hover:bg-[hsl(168_25%_98%)] transition-colors">
                        <td className="py-4 px-6 text-sm font-medium text-[hsl(172_43%_20%)]">{row.feature}</td>
                        <td className="py-4 px-6 text-center">
                          <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
                            row.centralized.status === 'success'
                              ? 'bg-[hsl(152_50%_95%)] text-[hsl(152_60%_30%)]'
                              : 'bg-[hsl(35_60%_95%)] text-[hsl(35_70%_35%)]'
                          }`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${
                              row.centralized.status === 'success' ? 'bg-[hsl(152_60%_42%)]' : 'bg-[hsl(35_70%_50%)]'
                            }`} />
                            {row.centralized.text}
                          </span>
                        </td>
                        <td className="py-4 px-6 text-center">
                          <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
                            row.federated.status === 'success'
                              ? 'bg-[hsl(152_50%_95%)] text-[hsl(152_60%_30%)]'
                              : 'bg-[hsl(35_60%_95%)] text-[hsl(35_70%_35%)]'
                          }`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${
                              row.federated.status === 'success' ? 'bg-[hsl(152_60%_42%)]' : 'bg-[hsl(35_70%_50%)]'
                            }`} />
                            {row.federated.text}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works - Horizontal Scroll Section */}
        <section className="how-it-works-section py-32 bg-white relative overflow-hidden">
          {/* Centered lungs diagnostic visual as background */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div
              className="w-[600px] h-[600px] opacity-[0.08]"
              style={{
                maskImage: 'radial-gradient(circle at center, rgba(0,0,0,1) 0%, rgba(0,0,0,0.5) 40%, transparent 70%)',
                WebkitMaskImage: 'radial-gradient(circle at center, rgba(0,0,0,1) 0%, rgba(0,0,0,0.5) 40%, transparent 70%)'
              }}
            >
              <div className="w-full h-full overflow-hidden">
                <img
                  src="/images/lungs2.png"
                  alt=""
                  className="w-[120%] h-[120%] object-cover object-left-top"
                  style={{ filter: 'saturate(0.5) contrast(1.1)' }}
                />
              </div>
            </div>
          </div>

          <div className="relative z-10">
            <div className="how-it-works-title text-center mb-16 px-6">
              <h2 className="text-4xl md:text-5xl font-semibold text-[hsl(172_43%_15%)] mb-6">
                How It Works
              </h2>
              <p className="text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto">
                From data upload to model comparison — a streamlined ML pipeline.
              </p>
            </div>

            {/* Simple 4-card grid */}
            <div className="max-w-6xl mx-auto px-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {[
                  {
                    step: '01',
                    title: 'Upload Dataset',
                    description: 'Upload chest X-ray images with NORMAL/PNEUMONIA folder structure.',
                    icon: (
                      <svg className="w-8 h-8" viewBox="0 0 40 40" fill="none">
                        <rect x="5" y="10" width="30" height="25" rx="4" stroke="currentColor" strokeWidth="2" />
                        <path d="M5 17h30" stroke="currentColor" strokeWidth="2" />
                        <circle cx="11" cy="13.5" r="1.5" fill="currentColor" />
                        <circle cx="17" cy="13.5" r="1.5" fill="currentColor" />
                        <path d="M15 27l5-5 4 4 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    ),
                    color: 'hsl(172 63% 35%)'
                  },
                  {
                    step: '02',
                    title: 'Configure Training',
                    description: 'Set hyperparameters and choose Centralized or Federated mode.',
                    icon: (
                      <svg className="w-8 h-8" viewBox="0 0 40 40" fill="none">
                        <circle cx="20" cy="20" r="14" stroke="currentColor" strokeWidth="2" />
                        <path d="M20 10v5m0 10v5m-10-10h5m10 0h5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                        <circle cx="20" cy="20" r="4" fill="currentColor" />
                      </svg>
                    ),
                    color: 'hsl(200 70% 45%)'
                  },
                  {
                    step: '03',
                    title: 'Train Model',
                    description: 'PyTorch Lightning training with real-time metrics via WebSocket.',
                    icon: (
                      <svg className="w-8 h-8" viewBox="0 0 40 40" fill="none">
                        <path d="M5 30l8-8 6 6 16-16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                        <circle cx="33" cy="14" r="4" fill="currentColor" />
                        <path d="M5 35h30" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.3" />
                      </svg>
                    ),
                    color: 'hsl(152 60% 40%)'
                  },
                  {
                    step: '04',
                    title: 'Compare Results',
                    description: 'Analyze Centralized vs Federated performance with detailed metrics.',
                    icon: (
                      <svg className="w-8 h-8" viewBox="0 0 40 40" fill="none">
                        <rect x="5" y="8" width="13" height="24" rx="2" stroke="currentColor" strokeWidth="2" />
                        <rect x="22" y="8" width="13" height="24" rx="2" stroke="currentColor" strokeWidth="2" />
                        <path d="M9 14h5M9 19h5M9 24h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                        <path d="M26 14h5M26 19h5M26 24h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                        <path d="M18 16l2 2 2-2M18 24l2-2 2 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    ),
                    color: 'hsl(260 60% 55%)'
                  }
                ].map((item, index) => (
                  <div
                    key={index}
                    className="step-card relative group"
                  >
                    <div className="h-full p-6 rounded-2xl bg-white/80 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-lg hover:shadow-xl hover:shadow-[hsl(172_40%_85%)]/30 transition-all duration-300 hover:-translate-y-1">
                      {/* Step indicator */}
                      <div className="flex items-center gap-3 mb-4">
                        <div
                          className="text-xs font-bold tracking-widest px-2.5 py-0.5 rounded-full"
                          style={{
                            backgroundColor: `${item.color}15`,
                            color: item.color
                          }}
                        >
                          STEP {item.step}
                        </div>
                      </div>

                      {/* Icon */}
                      <div
                        className="step-icon mb-4 w-14 h-14 rounded-xl flex items-center justify-center transition-all duration-300 group-hover:scale-105"
                        style={{
                          backgroundColor: `${item.color}12`,
                          color: item.color
                        }}
                      >
                        {item.icon}
                      </div>

                      <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-2">
                        {item.title}
                      </h3>
                      <p className="text-[hsl(215_15%_45%)] text-sm leading-relaxed">
                        {item.description}
                      </p>
                    </div>

                    {/* Connector arrow (hidden on last card and on mobile) */}
                    {index < 3 && (
                      <div className="hidden lg:block absolute top-1/2 -right-3 transform -translate-y-1/2 z-10">
                        <svg className="w-5 h-5 text-[hsl(172_40%_75%)]" viewBox="0 0 24 24" fill="none">
                          <path d="M9 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section - Refined */}
        <section className="cta-section py-32 px-6 bg-[hsl(172_63%_22%)] relative overflow-hidden">
          {/* Background elements */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-[hsl(172_55%_28%)] rounded-full blur-[150px] opacity-50" />
            <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-[hsl(172_70%_18%)] rounded-full blur-[120px] opacity-40" />
          </div>

          {/* Health Orb Background Image - centered and covering section */}
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            {/* Full-section centered orb */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div
                className="w-full h-full max-w-[1200px] max-h-[800px]"
                style={{
                  maskImage: 'radial-gradient(ellipse 60% 80% at 50% 50%, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.3) 50%, transparent 80%)',
                  WebkitMaskImage: 'radial-gradient(ellipse 60% 80% at 50% 50%, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.3) 50%, transparent 80%)'
                }}
              >
                {/* Crop bottom-right Gemini logo by oversizing and repositioning */}
                <div className="w-full h-full overflow-hidden">
                  <img
                    src="/images/health_orb.png"
                    alt=""
                    className="w-[115%] h-[120%] object-cover object-left-top opacity-40 mix-blend-soft-light"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="relative z-10 max-w-3xl mx-auto text-center">
            {/* Medical cross icon - now enhanced with orb imagery visible behind */}
            <div className="cta-icon mb-8 mx-auto w-20 h-20 rounded-3xl bg-white/10 backdrop-blur flex items-center justify-center">
              <svg className="w-10 h-10 text-white" viewBox="0 0 40 40" fill="none">
                <path d="M20 8v24M8 20h24" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                <circle cx="20" cy="20" r="16" stroke="currentColor" strokeWidth="2" opacity="0.3" />
              </svg>
            </div>

            <div className="cta-content">
              <h2 className="text-4xl md:text-5xl font-semibold text-white mb-6">
                Ready to Train Your Model?
              </h2>
              <p className="text-xl text-white/80 mb-12 max-w-xl mx-auto">
                Start detecting pneumonia with state-of-the-art machine learning, powered by privacy-preserving technology.
              </p>
            </div>

            <Button
              size="lg"
              className="cta-button bg-white text-[hsl(172_63%_22%)] hover:bg-white/90 text-lg px-12 py-7 rounded-2xl shadow-xl shadow-black/20 transition-all duration-300 hover:shadow-2xl hover:-translate-y-0.5 font-semibold"
              onClick={handleGetStarted}
            >
              Get Started Now
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default Landing;
