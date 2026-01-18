import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { ArrowRight, ArrowLeft } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface WelcomeGuideProps {
  onClose: () => void;
}

/**
 * Welcome Guide component for onboarding new users
 * Redesigned to match Landing page aesthetic - Apple-inspired, glass morphism, deep teal
 */
const WelcomeGuide = ({ onClose }: WelcomeGuideProps) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [open, setOpen] = useState(true);
  const [direction, setDirection] = useState(1); // 1 = forward, -1 = backward

  // Custom SVG icons matching Landing page style
  const icons = {
    welcome: (
      <svg className="w-12 h-12" viewBox="0 0 48 48" fill="none">
        <circle cx="24" cy="24" r="20" stroke="hsl(172 63% 22%)" strokeWidth="2" fill="hsl(168 40% 95%)" />
        <path d="M16 24c0-4.4 3.6-8 8-8s8 3.6 8 8-3.6 8-8 8" stroke="hsl(172 63% 22%)" strokeWidth="2" strokeLinecap="round" />
        <circle cx="24" cy="24" r="4" fill="hsl(172 63% 22%)" />
        <path d="M24 8v4M24 36v4M8 24h4M36 24h4" stroke="hsl(172 40% 70%)" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
    upload: (
      <svg className="w-12 h-12" viewBox="0 0 48 48" fill="none">
        <rect x="8" y="14" width="32" height="26" rx="4" stroke="hsl(172 63% 22%)" strokeWidth="2" fill="hsl(168 40% 95%)" />
        <path d="M8 22h32" stroke="hsl(172 63% 22%)" strokeWidth="2" />
        <circle cx="14" cy="18" r="2" fill="hsl(172 63% 35%)" />
        <circle cx="20" cy="18" r="2" fill="hsl(172 63% 35%)" />
        <path d="M24 28v8M20 32l4-4 4 4" stroke="hsl(172 63% 22%)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    configure: (
      <svg className="w-12 h-12" viewBox="0 0 48 48" fill="none">
        <circle cx="24" cy="24" r="16" stroke="hsl(172 63% 22%)" strokeWidth="2" fill="hsl(168 40% 95%)" />
        <path d="M24 12v6m0 12v6m-12-12h6m12 0h6" stroke="hsl(172 63% 22%)" strokeWidth="2" strokeLinecap="round" />
        <circle cx="24" cy="24" r="5" fill="hsl(172 63% 22%)" />
        <circle cx="24" cy="24" r="8" stroke="hsl(172 40% 70%)" strokeWidth="1" strokeDasharray="3 3" />
      </svg>
    ),
    train: (
      <svg className="w-12 h-12" viewBox="0 0 48 48" fill="none">
        <path d="M8 36l10-10 8 8 18-18" stroke="hsl(172 63% 22%)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
        <circle cx="40" cy="16" r="5" fill="hsl(152 60% 42%)" />
        <path d="M8 40h32" stroke="hsl(172 30% 80%)" strokeWidth="1.5" strokeLinecap="round" />
        <circle cx="18" cy="26" r="3" fill="hsl(168 40% 95%)" stroke="hsl(172 63% 22%)" strokeWidth="1.5" />
        <circle cx="26" cy="34" r="3" fill="hsl(168 40% 95%)" stroke="hsl(172 63% 22%)" strokeWidth="1.5" />
      </svg>
    ),
    analyze: (
      <svg className="w-12 h-12" viewBox="0 0 48 48" fill="none">
        <rect x="6" y="10" width="16" height="28" rx="3" stroke="hsl(172 63% 22%)" strokeWidth="2" fill="hsl(168 40% 95%)" />
        <rect x="26" y="10" width="16" height="28" rx="3" stroke="hsl(172 63% 22%)" strokeWidth="2" fill="hsl(168 40% 95%)" />
        <path d="M10 18h8M10 24h8M10 30h5" stroke="hsl(172 50% 50%)" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M30 18h8M30 24h8M30 30h5" stroke="hsl(172 50% 50%)" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M22 20l2 2 2-2M22 28l2-2 2 2" stroke="hsl(152 60% 42%)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  };

  const steps = [
    {
      title: 'Welcome to XRay Vision AI Forge',
      description: 'Your platform for state-of-the-art pneumonia detection',
      content: (
        <div className="space-y-6">
          <p className="text-[hsl(215_15%_40%)] leading-relaxed text-lg">
            Train deep learning models for pneumonia detection from chest X-rays.
            Choose between <span className="font-semibold text-[hsl(172_43%_25%)]">Centralized</span> or
            privacy-preserving <span className="font-semibold text-[hsl(172_43%_25%)]">Federated Learning</span>.
          </p>
          <div className="grid grid-cols-2 gap-4">
            {[
              { label: 'Upload & Prepare', sublabel: 'Your dataset' },
              { label: 'Configure', sublabel: 'Training parameters' },
              { label: 'Monitor', sublabel: 'In real-time' },
              { label: 'Analyze', sublabel: 'Compare results' },
            ].map((item, i) => (
              <div
                key={i}
                className="group p-4 rounded-2xl bg-white/60 backdrop-blur-sm border border-[hsl(172_30%_88%)] hover:bg-white hover:shadow-lg hover:shadow-[hsl(172_40%_85%)]/30 transition-all duration-300"
              >
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-[hsl(172_63%_35%)] group-hover:scale-125 transition-transform" />
                  <div>
                    <span className="font-medium text-[hsl(172_43%_20%)]">{item.label}</span>
                    <span className="text-[hsl(215_15%_50%)] text-sm block">{item.sublabel}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ),
      icon: icons.welcome,
    },
    {
      title: 'Step 1: Upload Dataset',
      description: 'Start by uploading your chest X-ray dataset',
      content: (
        <div className="space-y-6">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-[1.5rem] blur-xl opacity-40" />
            <div className="relative bg-white/80 backdrop-blur-sm rounded-[1.5rem] p-6 border border-[hsl(172_30%_88%)]">
              <h4 className="font-semibold text-[hsl(172_43%_15%)] text-lg mb-4">Dataset Format</h4>
              <p className="text-[hsl(215_15%_45%)] mb-4">Upload a ZIP file containing:</p>
              <ul className="space-y-3">
                {[
                  { title: 'Images folder', desc: 'X-ray images organized by class' },
                  { title: 'Metadata CSV', desc: 'Image labels and information' },
                ].map((item, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-lg bg-[hsl(172_40%_94%)] flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-[hsl(172_63%_35%)]" />
                    </div>
                    <div>
                      <span className="font-medium text-[hsl(172_43%_20%)]">{item.title}</span>
                      <span className="text-[hsl(215_15%_50%)] text-sm block">{item.desc}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <div className="p-4 rounded-xl bg-[hsl(35_60%_96%)] border border-[hsl(35_50%_85%)]">
            <p className="text-sm text-[hsl(35_50%_30%)]">
              <span className="font-semibold text-[hsl(35_70%_35%)]">Pro tip:</span> Configure the train/validation split to control how much data is used for training vs. testing.
            </p>
          </div>
        </div>
      ),
      icon: icons.upload,
    },
    {
      title: 'Step 2: Configure Experiment',
      description: 'Set up your training parameters',
      content: (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            {[
              {
                title: 'Centralized',
                badge: 'Traditional',
                badgeColor: 'hsl(210 60% 40%)',
                desc: 'Training on a single machine with all data collected centrally',
                benefits: ['Faster training', 'Simpler setup'],
              },
              {
                title: 'Federated',
                badge: 'Privacy-First',
                badgeColor: 'hsl(172 63% 28%)',
                desc: 'Distributed training preserving data privacy at each node',
                benefits: ['Data stays local', 'HIPAA compliant'],
              },
            ].map((mode, i) => (
              <div key={i} className="relative group">
                <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-2xl blur-lg opacity-30" />
                <div className="relative h-full bg-white/80 backdrop-blur-sm rounded-2xl p-5 border border-[hsl(172_30%_88%)]">
                  <div
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium mb-3"
                    style={{ backgroundColor: `${mode.badgeColor}15`, color: mode.badgeColor }}
                  >
                    {mode.badge}
                  </div>
                  <h4 className="font-semibold text-[hsl(172_43%_15%)] mb-2">{mode.title}</h4>
                  <p className="text-sm text-[hsl(215_15%_50%)] mb-3">{mode.desc}</p>
                  <ul className="space-y-1.5">
                    {mode.benefits.map((b, j) => (
                      <li key={j} className="flex items-center gap-2 text-sm text-[hsl(152_60%_30%)]">
                        <svg className="w-3.5 h-3.5" viewBox="0 0 14 14" fill="none">
                          <circle cx="7" cy="7" r="6" stroke="hsl(152 60% 42%)" strokeWidth="1.5" />
                          <path d="M4 7l2 2 4-4" stroke="hsl(152 60% 42%)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        {b}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
          <div className="p-4 rounded-xl bg-[hsl(172_40%_96%)] border border-[hsl(172_35%_88%)]">
            <p className="text-sm text-[hsl(172_35%_30%)]">
              <span className="font-semibold text-[hsl(172_45%_25%)]">New to ML?</span> Start with the default values — they're optimized for most use cases.
            </p>
          </div>
        </div>
      ),
      icon: icons.configure,
    },
    {
      title: 'Step 3: Monitor Training',
      description: 'Watch your model learn in real-time',
      content: (
        <div className="space-y-6">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-[1.5rem] blur-xl opacity-40" />
            <div className="relative bg-white/80 backdrop-blur-sm rounded-[1.5rem] p-6 border border-[hsl(172_30%_88%)]">
              <h4 className="font-semibold text-[hsl(172_43%_15%)] text-lg mb-5">Real-time Monitoring</h4>
              <div className="space-y-4">
                {[
                  { title: 'Live Metrics', desc: 'Track loss, accuracy, and other metrics as they update' },
                  { title: 'Progress Bars', desc: 'See epoch and round progress at a glance' },
                  { title: 'Status Messages', desc: 'Get detailed training updates and notifications' },
                ].map((feature, i) => (
                  <div key={i} className="flex items-start gap-4 p-3 rounded-xl">
                    <div className="w-10 h-10 rounded-xl bg-[hsl(172_40%_94%)] flex items-center justify-center flex-shrink-0">
                      <div className="w-2 h-2 rounded-full bg-[hsl(172_63%_35%)]" />
                    </div>
                    <div>
                      <span className="font-medium text-[hsl(172_43%_20%)]">{feature.title}</span>
                      <span className="text-[hsl(215_15%_50%)] text-sm block">{feature.desc}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="p-4 rounded-xl bg-[hsl(172_40%_96%)] border border-[hsl(172_35%_88%)]">
            <p className="text-sm text-[hsl(172_35%_30%)]">
              <span className="font-semibold text-[hsl(172_45%_25%)]">Pro tip:</span> Training can take a while. Feel free to minimize — we'll keep you updated!
            </p>
          </div>
        </div>
      ),
      icon: icons.train,
    },
    {
      title: 'Step 4: Analyze Results',
      description: "Understand your model's performance",
      content: (
        <div className="space-y-6">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-[1.5rem] blur-xl opacity-40" />
            <div className="relative bg-white/80 backdrop-blur-sm rounded-[1.5rem] p-6 border border-[hsl(172_30%_88%)]">
              <h4 className="font-semibold text-[hsl(172_43%_15%)] text-lg mb-5">Comprehensive Analytics</h4>
              <div className="grid grid-cols-3 gap-4 mb-6">
                {[
                  { label: 'Accuracy', value: '93%', color: 'hsl(152 60% 42%)' },
                  { label: 'Recall', value: '91%', color: 'hsl(172 63% 28%)' },
                  { label: 'F1 Score', value: '92%', color: 'hsl(200 70% 45%)' },
                ].map((metric, i) => (
                  <div key={i} className="text-center p-4 rounded-xl bg-white/60 border border-[hsl(172_25%_90%)]">
                    <div className="text-2xl font-bold" style={{ color: metric.color }}>{metric.value}</div>
                    <div className="text-xs text-[hsl(215_15%_50%)] mt-1">{metric.label}</div>
                  </div>
                ))}
              </div>
              <ul className="space-y-3">
                {[
                  { title: 'Performance Metrics', desc: 'Accuracy, precision, recall, F1-score, AUROC' },
                  { title: 'Training Curves', desc: 'Visualize loss and accuracy over time' },
                  { title: 'Confusion Matrix', desc: 'See detailed classification patterns' },
                ].map((item, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-lg bg-[hsl(152_50%_94%)] flex items-center justify-center flex-shrink-0 mt-0.5">
                      <svg className="w-3 h-3 text-[hsl(152_60%_35%)]" viewBox="0 0 12 12" fill="none">
                        <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                    <div>
                      <span className="font-medium text-[hsl(172_43%_20%)]">{item.title}</span>
                      <span className="text-[hsl(215_15%_50%)] text-sm block">{item.desc}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <div className="p-5 rounded-xl bg-[hsl(172_63%_22%)] text-white relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-[hsl(172_55%_28%)] to-transparent opacity-50" />
            <div className="relative flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center">
                <svg className="w-5 h-5 text-white" viewBox="0 0 20 20" fill="none">
                  <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </div>
              <div>
                <span className="font-semibold block">Ready to start?</span>
                <span className="text-white/80 text-sm">Click "Get Started" to begin your first experiment!</span>
              </div>
            </div>
          </div>
        </div>
      ),
      icon: icons.analyze,
    },
  ];

  const currentStepData = steps[currentStep];
  const isLastStep = currentStep === steps.length - 1;
  const isFirstStep = currentStep === 0;

  const handleNext = () => {
    if (isLastStep) {
      handleClose();
    } else {
      setDirection(1);
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setDirection(-1);
      setCurrentStep(currentStep - 1);
    }
  };

  const handleStepClick = (index: number) => {
    setDirection(index > currentStep ? 1 : -1);
    setCurrentStep(index);
  };

  const handleClose = () => {
    setOpen(false);
    setTimeout(onClose, 200);
  };

  // Motion variants
  const contentVariants = {
    enter: (dir: number) => ({
      x: dir > 0 ? 30 : -30,
      opacity: 0,
    }),
    center: {
      x: 0,
      opacity: 1,
    },
    exit: (dir: number) => ({
      x: dir > 0 ? -30 : 30,
      opacity: 0,
    }),
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && handleClose()}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto p-0 bg-transparent border-none shadow-none">
        {/* Outer glow effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_90%)] to-[hsl(168_40%_85%)] rounded-[2rem] blur-2xl opacity-50" />

        {/* Main dialog container - glass morphism */}
        <div className="relative bg-white/95 backdrop-blur-xl rounded-[2rem] border border-[hsl(168_20%_90%)] shadow-2xl shadow-[hsl(172_40%_70%)]/20 overflow-hidden">
          {/* Decorative background elements */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-[hsl(172_40%_85%)] rounded-full blur-[100px] opacity-30 pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-48 h-48 bg-[hsl(210_60%_90%)] rounded-full blur-[80px] opacity-25 pointer-events-none" />

          <div className="relative p-8">
            <DialogHeader>
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-5">
                  {/* Icon container with glow */}
                  <div className="relative">
                    <div className="absolute inset-0 bg-[hsl(172_50%_80%)] rounded-2xl blur-lg opacity-50" />
                    <div className="relative p-4 rounded-2xl bg-white shadow-lg border border-[hsl(168_20%_92%)]">
                      {currentStepData.icon}
                    </div>
                  </div>
                  <div>
                    <DialogTitle className="text-2xl font-semibold text-[hsl(172_43%_15%)] tracking-tight">
                      {currentStepData.title}
                    </DialogTitle>
                    <DialogDescription className="mt-1.5 text-[hsl(215_15%_45%)] text-base">
                      {currentStepData.description}
                    </DialogDescription>
                  </div>
                </div>
              </div>
            </DialogHeader>

            {/* Step content with Motion animation */}
            <div className="py-4 min-h-[300px]">
              <AnimatePresence mode="wait" custom={direction}>
                <motion.div
                  key={currentStep}
                  custom={direction}
                  variants={contentVariants}
                  initial="enter"
                  animate="center"
                  exit="exit"
                  transition={{ duration: 0.25, ease: 'easeOut' }}
                >
                  {currentStepData.content}
                </motion.div>
              </AnimatePresence>
            </div>

            {/* Progress indicator */}
            <div className="flex justify-center gap-2 py-6">
              {steps.map((_, index) => (
                <button
                  key={index}
                  onClick={() => handleStepClick(index)}
                  aria-label={`Go to step ${index + 1}`}
                  className={`rounded-full transition-all duration-300 ${
                    index === currentStep
                      ? 'w-10 h-2.5 bg-[hsl(172_63%_22%)] shadow-md shadow-[hsl(172_63%_22%)]/30'
                      : index < currentStep
                        ? 'w-2.5 h-2.5 bg-[hsl(152_60%_42%)]'
                        : 'w-2.5 h-2.5 bg-[hsl(210_15%_85%)] hover:bg-[hsl(210_15%_75%)]'
                  }`}
                />
              ))}
            </div>

            <DialogFooter className="flex justify-between items-center sm:justify-between gap-4 pt-2">
              <div className="flex gap-2">
                {!isFirstStep && (
                  <Button
                    variant="outline"
                    onClick={handlePrevious}
                    className="rounded-xl px-5 border-2 border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)] hover:border-[hsl(172_40%_70%)]"
                  >
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Previous
                  </Button>
                )}
              </div>
              <div className="flex gap-3">
                {!isLastStep && (
                  <Button
                    variant="ghost"
                    onClick={handleClose}
                    className="rounded-xl px-5 text-[hsl(215_15%_45%)] hover:bg-[hsl(168_25%_94%)] hover:text-[hsl(172_43%_25%)]"
                  >
                    Skip Tour
                  </Button>
                )}
                <Button
                  onClick={handleNext}
                  className="rounded-xl px-6 py-5 bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white shadow-lg shadow-[hsl(172_63%_22%)]/25"
                >
                  {isLastStep ? 'Get Started' : (
                    <>
                      Next <ArrowRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>
              </div>
            </DialogFooter>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default WelcomeGuide;
