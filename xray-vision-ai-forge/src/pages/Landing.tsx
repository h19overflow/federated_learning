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

const Landing = () => {
  const navigate = useNavigate();
  const comparisonRef = useRef<HTMLElement>(null);

  const handleGetStarted = () => {
    navigate('/experiment');
  };

  const scrollToComparison = () => {
    comparisonRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header />

      <main className="flex-1 overflow-y-auto bg-hero-gradient">
        {/* Hero Section - Apple Style */}
        <section className="relative min-h-[90vh] flex flex-col items-center justify-center px-6 overflow-hidden">
          {/* Subtle background elements */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-[hsl(172_40%_85%)] rounded-full blur-[120px] opacity-30" />
            <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-[hsl(210_60%_90%)] rounded-full blur-[100px] opacity-25" />
          </div>

          <div className="relative z-10 max-w-4xl mx-auto text-center">
            {/* Trust badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 mb-8 rounded-full bg-white/60 backdrop-blur-sm border border-[hsl(172_30%_85%)] shadow-sm opacity-0 animate-fade-in stagger-1" style={{ animationFillMode: 'forwards' }}>
              <div className="w-2 h-2 rounded-full bg-[hsl(152_60%_42%)] animate-pulse" />
              <span className="text-sm font-medium text-[hsl(172_43%_25%)]">
                Powered by ResNet50 V2 & Flower Framework
              </span>
            </div>

            {/* Main headline */}
            <h1 className="text-5xl md:text-7xl font-semibold tracking-tight text-[hsl(172_43%_15%)] mb-6 opacity-0 animate-fade-in stagger-2" style={{ animationFillMode: 'forwards' }}>
              Medical AI,
              <br />
              <span className="text-[hsl(172_63%_28%)]">Refined.</span>
            </h1>

            {/* Subheadline */}
            <p className="text-xl md:text-2xl text-[hsl(215_15%_45%)] font-light max-w-2xl mx-auto mb-12 leading-relaxed opacity-0 animate-fade-in stagger-3" style={{ animationFillMode: 'forwards' }}>
              Train state-of-the-art pneumonia detection models with
              <span className="font-medium text-[hsl(172_43%_25%)]"> Centralized </span>
              or
              <span className="font-medium text-[hsl(172_43%_25%)]"> Federated Learning</span>.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center opacity-0 animate-fade-in stagger-4" style={{ animationFillMode: 'forwards' }}>
              <Button
                size="lg"
                className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white text-lg px-10 py-7 rounded-2xl shadow-lg shadow-[hsl(172_63%_22%)]/20 transition-all duration-300 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/30 hover:-translate-y-0.5"
                onClick={handleGetStarted}
              >
                Start Training
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="text-lg px-10 py-7 rounded-2xl border-2 border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)] transition-all duration-300"
                onClick={scrollToComparison}
              >
                Learn More
              </Button>
            </div>
          </div>

          {/* Scroll indicator */}
          <div className="absolute bottom-10 left-1/2 -translate-x-1/2 opacity-0 animate-fade-in stagger-5" style={{ animationFillMode: 'forwards' }}>
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
        <section className="py-32 px-6 bg-white">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-20">
              <h2 className="text-4xl md:text-5xl font-semibold text-[hsl(172_43%_15%)] mb-6">
                Why Choose Our Platform?
              </h2>
              <p className="text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto">
                Enterprise-grade AI training with institutional trust.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
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
                  className="group p-8 rounded-3xl bg-[hsl(168_25%_98%)] border border-[hsl(168_20%_92%)] hover:bg-white hover:shadow-xl hover:shadow-[hsl(172_40%_85%)]/30 transition-all duration-500 hover:-translate-y-1"
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
        <section ref={comparisonRef} className="py-32 px-6 bg-trust-gradient relative overflow-hidden">
          <div className="absolute inset-0 noise-overlay" />

          <div className="relative z-10 max-w-6xl mx-auto">
            <div className="text-center mb-20">
              <h2 className="text-4xl md:text-5xl font-semibold text-[hsl(172_43%_15%)] mb-6">
                Choose Your Approach
              </h2>
              <p className="text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto">
                Two powerful methodologies, one exceptional outcome.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Centralized Learning Card */}
              <div className="relative group">
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

                  {/* Visual Diagram */}
                  <div className="relative h-48 mb-8 rounded-2xl bg-gradient-to-br from-[hsl(210_40%_98%)] to-[hsl(210_50%_95%)] flex items-center justify-center overflow-hidden">
                    <div className="flex flex-col items-center">
                      {/* Data sources */}
                      <div className="flex gap-6 mb-4">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="flex flex-col items-center">
                            <div className="w-10 h-10 rounded-xl bg-[hsl(210_60%_88%)] flex items-center justify-center">
                              <svg className="w-5 h-5 text-[hsl(210_60%_45%)]" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M20 6H4V4h16v2zm-2 6H6V8h12v4zm-4 6H10v-4h4v4z"/>
                              </svg>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Arrows */}
                      <div className="flex gap-12 mb-4">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="h-8 w-px bg-gradient-to-b from-[hsl(210_60%_75%)] to-[hsl(210_60%_60%)]" />
                        ))}
                      </div>

                      {/* Central server */}
                      <div className="relative">
                        <div className="absolute -inset-3 bg-[hsl(210_60%_88%)] rounded-2xl blur-sm" />
                        <div className="relative px-6 py-3 bg-white rounded-xl border-2 border-[hsl(210_60%_70%)] shadow-md">
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-[hsl(210_60%_50%)] animate-pulse" />
                            <span className="text-sm font-medium text-[hsl(210_60%_35%)]">Central Server</span>
                          </div>
                        </div>
                      </div>
                    </div>
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
              <div className="relative group">
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

                  {/* Visual Diagram */}
                  <div className="relative h-48 mb-8 rounded-2xl bg-gradient-to-br from-[hsl(172_30%_97%)] to-[hsl(168_40%_93%)] flex items-center justify-center overflow-hidden">
                    <div className="flex flex-col items-center">
                      {/* Global server */}
                      <div className="relative mb-4">
                        <div className="absolute -inset-3 bg-[hsl(172_45%_85%)] rounded-2xl blur-sm" />
                        <div className="relative px-6 py-3 bg-white rounded-xl border-2 border-[hsl(172_50%_65%)] shadow-md">
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-[hsl(172_63%_35%)] animate-pulse" />
                            <span className="text-sm font-medium text-[hsl(172_43%_25%)]">Global Server</span>
                          </div>
                        </div>
                      </div>

                      {/* Bidirectional arrows */}
                      <div className="flex gap-8 mb-4">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="flex flex-col items-center gap-1">
                            <ArrowRight className="w-3 h-3 text-[hsl(172_50%_55%)] rotate-90" />
                            <div className="h-4 w-px bg-gradient-to-b from-[hsl(172_50%_70%)] to-[hsl(172_50%_55%)]" />
                            <ArrowRight className="w-3 h-3 text-[hsl(172_50%_55%)] -rotate-90" />
                          </div>
                        ))}
                      </div>

                      {/* Local clients */}
                      <div className="flex gap-6">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="relative">
                            <div className="p-3 rounded-xl bg-white border border-[hsl(172_40%_85%)] shadow-sm">
                              <div className="flex flex-col items-center gap-1">
                                <svg className="w-5 h-5 text-[hsl(172_50%_40%)]" viewBox="0 0 24 24" fill="currentColor">
                                  <path d="M20 6H4V4h16v2zm-2 6H6V8h12v4zm-4 6H10v-4h4v4z"/>
                                </svg>
                                <div className="w-2 h-2 rounded-full bg-[hsl(152_60%_42%)]" />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
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
            <div className="mt-16 bg-white rounded-[2rem] p-8 shadow-lg border border-[hsl(210_15%_92%)]">
              <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)] mb-8 text-center">
                Quick Comparison
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-[hsl(210_15%_92%)]">
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
                      <tr key={index} className="hover:bg-[hsl(168_25%_98%)] transition-colors">
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
        <section className="py-32 px-6 bg-white">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-20">
              <h2 className="text-4xl md:text-5xl font-semibold text-[hsl(172_43%_15%)] mb-6">
                How It Works
              </h2>
              <p className="text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto">
                Four simple steps to train your AI model.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
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
                  title: 'Analyze',
                  description: 'Review metrics and export your trained model.',
                  icon: (
                    <svg className="w-8 h-8" viewBox="0 0 32 32" fill="none">
                      <rect x="4" y="4" width="24" height="24" rx="3" stroke="currentColor" strokeWidth="2" />
                      <path d="M4 12h24M12 12v16" stroke="currentColor" strokeWidth="2" />
                      <path d="M18 18l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )
                }
              ].map((item, index) => (
                <div
                  key={index}
                  className="relative group"
                >
                  <div className="p-8 rounded-3xl bg-[hsl(168_25%_98%)] border border-[hsl(168_20%_92%)] hover:bg-white hover:shadow-xl hover:shadow-[hsl(172_40%_85%)]/30 transition-all duration-500 hover:-translate-y-1 text-center">
                    {/* Step number */}
                    <div className="text-xs font-bold text-[hsl(172_63%_35%)] tracking-widest mb-4">
                      STEP {item.step}
                    </div>

                    {/* Icon */}
                    <div className="mb-6 mx-auto w-16 h-16 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center text-[hsl(172_63%_28%)] group-hover:bg-[hsl(172_63%_22%)] group-hover:text-white transition-all duration-300">
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
                    <div className="hidden md:block absolute top-1/2 -right-3 w-6 h-px bg-gradient-to-r from-[hsl(172_40%_80%)] to-[hsl(172_40%_90%)]" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA Section - Refined */}
        <section className="py-32 px-6 bg-[hsl(172_63%_22%)] relative overflow-hidden">
          {/* Background elements */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-[hsl(172_55%_28%)] rounded-full blur-[150px] opacity-50" />
            <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-[hsl(172_70%_18%)] rounded-full blur-[120px] opacity-40" />
          </div>

          <div className="relative z-10 max-w-3xl mx-auto text-center">
            {/* Medical cross icon */}
            <div className="mb-8 mx-auto w-20 h-20 rounded-3xl bg-white/10 backdrop-blur flex items-center justify-center">
              <svg className="w-10 h-10 text-white" viewBox="0 0 40 40" fill="none">
                <path d="M20 8v24M8 20h24" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                <circle cx="20" cy="20" r="16" stroke="currentColor" strokeWidth="2" opacity="0.3" />
              </svg>
            </div>

            <h2 className="text-4xl md:text-5xl font-semibold text-white mb-6">
              Ready to Train Your Model?
            </h2>
            <p className="text-xl text-white/80 mb-12 max-w-xl mx-auto">
              Start detecting pneumonia with state-of-the-art machine learning, powered by privacy-preserving technology.
            </p>

            <Button
              size="lg"
              className="bg-white text-[hsl(172_63%_22%)] hover:bg-white/90 text-lg px-12 py-7 rounded-2xl shadow-xl shadow-black/20 transition-all duration-300 hover:shadow-2xl hover:-translate-y-0.5 font-semibold"
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
