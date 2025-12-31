import React, { useEffect, useRef } from 'react';
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
import Header from '@/components/Header';
import Footer from '@/components/Footer';
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
          trigger: '.steps-grid',
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
          trigger: '.steps-grid',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        scale: 0,
        stagger: ANIMATION_CONFIG.staggerDelay,
        duration: 0.6,
        ease: 'back.out(1.7)',
        delay: 0.3,
      });

      // Connector lines: animate width
      gsap.from('.step-connector', {
        scrollTrigger: {
          trigger: '.steps-grid',
          start: ANIMATION_CONFIG.scrollTriggerStart,
          toggleActions: 'play none none none',
        },
        scaleX: 0,
        transformOrigin: 'left center',
        stagger: ANIMATION_CONFIG.staggerDelay,
        duration: 0.5,
        ease: ANIMATION_CONFIG.ease,
        delay: 0.5,
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
      <Header />

      <main ref={mainRef} className="flex-1 overflow-y-auto bg-hero-gradient">
        {/* Hero Section - Apple Style */}
        <section className="relative min-h-[90vh] flex flex-col items-center justify-center px-6 overflow-hidden">
          {/* Subtle background elements */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-[hsl(172_40%_85%)] rounded-full blur-[120px] opacity-30" />
            <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-[hsl(210_60%_90%)] rounded-full blur-[100px] opacity-25" />
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
        <section className="features-section py-32 px-6 bg-white">
          <div className="max-w-6xl mx-auto">
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
                <div className="absolute inset-0 bg-gradient-to-br from-[hsl(210_80%_95%)] to-[hsl(210_60%_90%)] rounded-[2rem] blur-xl opacity-50 group-hover:opacity-70 transition-opacity" />
                <div className="relative bg-white/90 backdrop-blur-sm rounded-[2rem] p-10 border border-[hsl(210_30%_90%)] shadow-lg hover:shadow-2xl transition-all duration-500">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-8">
                    <div>
                      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[hsl(210_80%_95%)] text-[hsl(210_60%_45%)] text-sm font-medium mb-4">
                        Traditional
                      </div>
                      <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)]">
                        Centralized Learning
                      </h3>
                    </div>
                    <div className="p-3 rounded-2xl bg-[hsl(210_80%_96%)]">
                      <svg className="w-8 h-8 text-[hsl(210_60%_45%)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <rect x="4" y="4" width="16" height="16" rx="2" />
                        <path d="M4 9h16M9 4v16" />
                      </svg>
                    </div>
                  </div>

                  {/* Animated Flow Diagram - Centralized */}
                  <div className="centralized-diagram relative h-56 mb-8 rounded-2xl bg-gradient-to-br from-[hsl(210_40%_98%)] to-[hsl(210_50%_95%)] overflow-hidden">
                    <svg viewBox="0 0 300 200" className="w-full h-full" aria-label="Centralized learning data flow diagram showing patient data flowing from hospitals to central server">
                      <defs>
                        {/* Gradient for data particles */}
                        <linearGradient id="dataGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                          <stop offset="0%" stopColor="hsl(210 60% 60%)" />
                          <stop offset="100%" stopColor="hsl(210 60% 45%)" />
                        </linearGradient>

                        {/* Pulsing glow for server */}
                        <filter id="glow">
                          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                          <feMerge>
                            <feMergeNode in="coloredBlur"/>
                            <feMergeNode in="SourceGraphic"/>
                          </feMerge>
                        </filter>
                      </defs>

                      {/* Hospital nodes at bottom */}
                      {[0, 1, 2].map((i) => (
                        <g key={i} transform={`translate(${75 + i * 75}, 150)`} className="data-node">
                          {/* Hospital building */}
                          <rect x="-15" y="-15" width="30" height="30" rx="6" fill="white" stroke="hsl(210 60% 75%)" strokeWidth="2" />
                          {/* Database icon */}
                          <path d="M-8,-8 L8,-8 M-8,-2 L8,-2 M-8,4 L8,4" stroke="hsl(210 60% 45%)" strokeWidth="1.5" strokeLinecap="round" />
                          {/* Unlocked indicator (privacy concern) */}
                          <g transform="translate(10, -10)">
                            <circle r="6" fill="hsl(35 70% 50%)" opacity="0.9" />
                            <path d="M-2,-1 L-2,-3 A2,2 0 0,1 2,-3 L2,-2" stroke="white" strokeWidth="1.2" fill="none" strokeLinecap="round" />
                            <rect x="-2.5" y="-1" width="5" height="4" rx="1" fill="white" />
                          </g>
                        </g>
                      ))}

                      {/* Central server at top */}
                      <g transform="translate(150, 40)">
                        {/* Pulsing glow */}
                        <circle className="pulse-ring" r="35" fill="hsl(210 60% 88%)" opacity="0.4">
                          <animate attributeName="r" values="35;40;35" dur="2s" repeatCount="indefinite" />
                          <animate attributeName="opacity" values="0.4;0.2;0.4" dur="2s" repeatCount="indefinite" />
                        </circle>
                        {/* Server box */}
                        <rect x="-40" y="-20" width="80" height="40" rx="8" fill="white" stroke="hsl(210 60% 70%)" strokeWidth="2.5" filter="url(#glow)" />
                        {/* Server rack lines */}
                        <path d="M-30,-8 L30,-8 M-30,0 L30,0 M-30,8 L30,8" stroke="hsl(210 60% 50%)" strokeWidth="1.5" />
                        {/* Activity indicator */}
                        <circle cx="-25" cy="-8" r="2" fill="hsl(210 60% 50%)" className="animate-pulse" />
                        <circle cx="-25" cy="0" r="2" fill="hsl(210 60% 50%)" className="animate-pulse" style={{ animationDelay: '0.3s' }} />
                        <circle cx="-25" cy="8" r="2" fill="hsl(210 60% 50%)" className="animate-pulse" style={{ animationDelay: '0.6s' }} />
                        {/* Label below the server box */}
                        <text x="0" y="35" textAnchor="middle" fontSize="10" fontWeight="600" fill="hsl(210 60% 35%)">Central Server</text>
                      </g>

                      {/* Animated data particles flowing upward */}
                      {[0, 1, 2].map((pathIndex) => (
                        <g key={pathIndex}>
                          {/* Flow path (invisible) */}
                          <path
                            id={`flow-path-${pathIndex}`}
                            d={`M ${75 + pathIndex * 75} 135 L ${75 + pathIndex * 75} 80`}
                            fill="none"
                            stroke="none"
                          />

                          {/* Path line */}
                          <line
                            x1={75 + pathIndex * 75}
                            y1="135"
                            x2={75 + pathIndex * 75}
                            y2="80"
                            stroke="hsl(210 60% 80%)"
                            strokeWidth="2"
                            strokeDasharray="4 4"
                            opacity="0.5"
                          />

                          {/* Data particles */}
                          {[0, 1, 2].map((particleIndex) => (
                            <g key={particleIndex}>
                              <circle
                                className={`data-particle data-particle-${pathIndex}-${particleIndex}`}
                                r="4"
                                fill="url(#dataGradient)"
                                stroke="white"
                                strokeWidth="1"
                                opacity="0"
                              >
                                {/* Initial position at bottom */}
                                <animateMotion
                                  dur="2.5s"
                                  repeatCount="indefinite"
                                  begin={`${pathIndex * 0.3 + particleIndex * 0.8}s`}
                                  path={`M ${75 + pathIndex * 75} 135 L ${75 + pathIndex * 75} 80`}
                                />
                                {/* Fade in/out */}
                                <animate
                                  attributeName="opacity"
                                  values="0;1;1;0"
                                  keyTimes="0;0.1;0.85;1"
                                  dur="2.5s"
                                  repeatCount="indefinite"
                                  begin={`${pathIndex * 0.3 + particleIndex * 0.8}s`}
                                />
                              </circle>
                              {/* Document icon inside particle */}
                              <g opacity="0">
                                <rect
                                  width="4"
                                  height="5"
                                  rx="0.5"
                                  fill="white"
                                  opacity="0.8"
                                />
                                <animateMotion
                                  dur="2.5s"
                                  repeatCount="indefinite"
                                  begin={`${pathIndex * 0.3 + particleIndex * 0.8}s`}
                                  path={`M ${75 + pathIndex * 75 - 2} ${135 - 2.5} L ${75 + pathIndex * 75 - 2} ${80 - 2.5}`}
                                />
                                <animate
                                  attributeName="opacity"
                                  values="0;0.8;0.8;0"
                                  keyTimes="0;0.1;0.85;1"
                                  dur="2.5s"
                                  repeatCount="indefinite"
                                  begin={`${pathIndex * 0.3 + particleIndex * 0.8}s`}
                                />
                              </g>
                            </g>
                          ))}
                        </g>
                      ))}

                      {/* Convergence arrows */}
                      {[0, 1, 2].map((i) => (
                        <g key={i} opacity="0.6">
                          <path
                            d={`M ${75 + i * 75} 80 Q ${75 + i * 75} 60 150 60 L 150 60`}
                            fill="none"
                            stroke="hsl(210 60% 75%)"
                            strokeWidth="1.5"
                            strokeDasharray="3 3"
                          />
                          <polygon
                            points="148,55 155,60 148,65"
                            fill="hsl(210 60% 75%)"
                          />
                        </g>
                      ))}

                      {/* Centralized Learning Step Labels */}
                      {/* Step 1: Collect Data - at data source nodes */}
                      <g transform="translate(75, 175)">
                        <rect x="-30" y="-8" width="60" height="16" rx="4" fill="hsl(210 50% 92%)" stroke="hsl(210 55% 75%)" strokeWidth="1" />
                        <text x="0" y="4" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(210 55% 30%)">1. Collect Data</text>
                      </g>

                      {/* Step 2: Upload to Server - on the upward path */}
                      <g transform="translate(25, 105)">
                        <rect x="-28" y="-8" width="56" height="16" rx="4" fill="hsl(210 50% 90%)" stroke="hsl(210 55% 70%)" strokeWidth="1" />
                        <text x="0" y="4" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(210 55% 28%)">2. Upload</text>
                      </g>

                      {/* Step 3: Train Model - at the central server */}
                      <g transform="translate(150, 12)">
                        <rect x="-28" y="-7" width="56" height="14" rx="4" fill="hsl(210 55% 88%)" stroke="hsl(210 60% 65%)" strokeWidth="1" />
                        <text x="0" y="3" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(210 60% 28%)">3. Train Model</text>
                      </g>

                      {/* Step 4: Deploy - on the right side */}
                      <g transform="translate(275, 105)">
                        <rect x="-25" y="-8" width="50" height="16" rx="4" fill="hsl(210 50% 90%)" stroke="hsl(210 55% 70%)" strokeWidth="1" />
                        <text x="0" y="4" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(210 55% 28%)">4. Deploy</text>
                      </g>

                      {/* Privacy warning indicator */}
                      <g transform="translate(150, 175)">
                        <rect x="-32" y="-8" width="64" height="16" rx="4" fill="hsl(35 70% 92%)" stroke="hsl(35 65% 70%)" strokeWidth="1" />
                        <text x="0" y="4" textAnchor="middle" fontSize="7" fontWeight="500" fill="hsl(35 70% 35%)">⚠ Data Leaves</text>
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
                      {['Faster training time', 'Simpler implementation', 'Easier debugging', 'Ideal for small datasets'].map((item, i) => (
                        <li key={i} className="flex items-center gap-3 text-[hsl(215_15%_40%)]">
                          <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)]" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Considerations */}
                  <div className="mt-6 pt-6 border-t border-[hsl(210_15%_92%)] space-y-4">
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

                  {/* Animated Flow Diagram - Federated */}
                  <div className="federated-diagram relative h-56 mb-8 rounded-2xl bg-gradient-to-br from-[hsl(172_30%_97%)] to-[hsl(168_40%_93%)] overflow-hidden">
                    <svg viewBox="0 0 300 200" className="w-full h-full" aria-label="Federated learning diagram showing model updates flowing between global server and local hospitals while data stays secure">
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

        {/* How It Works - Apple Style Steps */}
        <section className="how-it-works-section py-32 px-6 bg-white">
          <div className="max-w-6xl mx-auto">
            <div className="how-it-works-title text-center mb-20">
              <h2 className="text-4xl md:text-5xl font-semibold text-[hsl(172_43%_15%)] mb-6">
                How It Works
              </h2>
              <p className="text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto">
                Four simple steps to train your AI model.
              </p>
            </div>

            <div className="steps-grid grid grid-cols-1 md:grid-cols-4 gap-6">
              {[
                {
                  step: '01',
                  title: 'Upload Dataset',
                  description: 'Upload your chest X-ray images in standard format.',
                  icon: (
                    <svg className="w-8 h-8" viewBox="0 0 32 32" fill="none">
                      <rect x="4" y="8" width="24" height="20" rx="3" stroke="currentColor" strokeWidth="2" />
                      <path d="M4 14h24" stroke="currentColor" strokeWidth="2" />
                      <circle cx="9" cy="11" r="1.5" fill="currentColor" />
                      <circle cx="14" cy="11" r="1.5" fill="currentColor" />
                      <path d="M12 22l4-4 3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )
                },
                {
                  step: '02',
                  title: 'Configure',
                  description: 'Choose training mode and fine-tune parameters.',
                  icon: (
                    <svg className="w-8 h-8" viewBox="0 0 32 32" fill="none">
                      <circle cx="16" cy="16" r="12" stroke="currentColor" strokeWidth="2" />
                      <path d="M16 8v4m0 8v4m-8-8h4m8 0h4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                      <circle cx="16" cy="16" r="3" fill="currentColor" />
                    </svg>
                  )
                },
                {
                  step: '03',
                  title: 'Train',
                  description: 'Watch your model learn with real-time metrics.',
                  icon: (
                    <svg className="w-8 h-8" viewBox="0 0 32 32" fill="none">
                      <path d="M4 24l7-7 5 5 12-12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      <circle cx="26" cy="10" r="3" fill="currentColor" />
                    </svg>
                  )
                },
                {
                  step: '04',
                  title: 'Compare & Analyze',
                  description: 'Compare performance across training modes.',
                  icon: (
                    <svg className="w-8 h-8" viewBox="0 0 32 32" fill="none">
                      <rect x="4" y="18" width="6" height="10" rx="1" stroke="currentColor" strokeWidth="2" fill="none" />
                      <rect x="13" y="12" width="6" height="16" rx="1" stroke="currentColor" strokeWidth="2" fill="none" />
                      <rect x="22" y="8" width="6" height="20" rx="1" stroke="currentColor" strokeWidth="2" fill="none" />
                      <path d="M7 18L16 12L25 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )
                }
              ].map((item, index) => (
                <div
                  key={index}
                  className="step-card relative group"
                >
                  <div className="p-8 rounded-3xl bg-[hsl(168_25%_98%)] border border-[hsl(168_20%_92%)] hover:bg-white hover:shadow-xl hover:shadow-[hsl(172_40%_85%)]/30 transition-all duration-500 hover:-translate-y-1 text-center">
                    {/* Step number */}
                    <div className="text-xs font-bold text-[hsl(172_63%_35%)] tracking-widest mb-4">
                      STEP {item.step}
                    </div>

                    {/* Icon */}
                    <div className="step-icon mb-6 mx-auto w-16 h-16 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center text-[hsl(172_63%_28%)] group-hover:bg-[hsl(172_63%_22%)] group-hover:text-white transition-all duration-300">
                      {item.icon}
                    </div>

                    <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)] mb-3">
                      {item.title}
                    </h3>
                    <p className="text-[hsl(215_15%_45%)] text-sm leading-relaxed">
                      {item.description}
                    </p>
                  </div>

                  {/* Connector line */}
                  {index < 3 && (
                    <div className="step-connector hidden md:block absolute top-1/2 -right-3 w-6 h-px bg-gradient-to-r from-[hsl(172_40%_80%)] to-[hsl(172_40%_90%)]" />
                  )}
                </div>
              ))}
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

          <div className="relative z-10 max-w-3xl mx-auto text-center">
            {/* Medical cross icon */}
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
