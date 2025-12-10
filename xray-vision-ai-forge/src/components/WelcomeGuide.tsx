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
import {
  Upload,
  Settings,
  Play,
  BarChart3,
  CheckCircle2,
  ArrowRight,
  ArrowLeft,
  X
} from 'lucide-react';

interface WelcomeGuideProps {
  onClose: () => void;
}

/**
 * Welcome Guide component for onboarding new users
 * Redesigned with Clinical Clarity theme - glass morphism, teal accents
 */
const WelcomeGuide = ({ onClose }: WelcomeGuideProps) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [open, setOpen] = useState(true);

  const steps = [
    {
      title: 'Welcome to XRay Vision AI Forge',
      description: 'Your platform for training pneumonia detection models using state-of-the-art machine learning',
      content: (
        <div className="space-y-5">
          <p className="text-[hsl(215_15%_45%)] leading-relaxed">
            This platform helps you train deep learning models for pneumonia detection from chest X-rays.
            You can choose between traditional centralized training or privacy-preserving federated learning.
          </p>
          <div className="bg-[hsl(210_100%_97%)] border border-[hsl(210_60%_85%)] rounded-xl p-5">
            <h4 className="font-semibold text-[hsl(210_70%_30%)] mb-3">What you'll learn:</h4>
            <ul className="space-y-2 text-sm text-[hsl(210_50%_35%)]">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[hsl(210_80%_50%)]" />
                How to upload and prepare your dataset
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[hsl(210_80%_50%)]" />
                Configure training parameters
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[hsl(210_80%_50%)]" />
                Monitor training in real-time
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[hsl(210_80%_50%)]" />
                Analyze and compare results
              </li>
            </ul>
          </div>
        </div>
      ),
      icon: <CheckCircle2 className="h-10 w-10 text-[hsl(172_63%_28%)]" />,
    },
    {
      title: 'Step 1: Upload Dataset',
      description: 'Start by uploading your chest X-ray dataset',
      content: (
        <div className="space-y-5">
          <div className="flex items-start gap-4 bg-[hsl(168_25%_97%)] p-5 rounded-xl border border-[hsl(168_20%_92%)]">
            <div className="p-2.5 rounded-xl bg-[hsl(172_40%_92%)]">
              <Upload className="h-6 w-6 text-[hsl(172_63%_28%)]" />
            </div>
            <div className="flex-1">
              <h4 className="font-semibold text-[hsl(172_43%_20%)] mb-2">Dataset Format</h4>
              <p className="text-sm text-[hsl(215_15%_45%)] mb-3">
                Upload a ZIP file containing:
              </p>
              <ul className="text-sm text-[hsl(215_15%_45%)] space-y-2">
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)] mt-1.5" />
                  <span><strong className="text-[hsl(172_43%_20%)]">Images folder:</strong> X-ray images organized by class</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)] mt-1.5" />
                  <span><strong className="text-[hsl(172_43%_20%)]">Metadata CSV:</strong> Image labels and information</span>
                </li>
              </ul>
            </div>
          </div>
          <div className="bg-[hsl(35_60%_96%)] border border-[hsl(35_50%_80%)] rounded-xl p-4">
            <p className="text-sm text-[hsl(35_50%_30%)]">
              <strong className="text-[hsl(35_70%_35%)]">Tip:</strong> Configure the train/validation split to control how much data is used for training vs. testing.
            </p>
          </div>
        </div>
      ),
      icon: <Upload className="h-10 w-10 text-[hsl(172_63%_28%)]" />,
    },
    {
      title: 'Step 2: Configure Experiment',
      description: 'Set up your training parameters',
      content: (
        <div className="space-y-5">
          <div className="flex items-start gap-4 bg-[hsl(168_25%_97%)] p-5 rounded-xl border border-[hsl(168_20%_92%)]">
            <div className="p-2.5 rounded-xl bg-[hsl(172_40%_92%)]">
              <Settings className="h-6 w-6 text-[hsl(172_63%_28%)]" />
            </div>
            <div className="flex-1">
              <h4 className="font-semibold text-[hsl(172_43%_20%)] mb-3">Training Modes</h4>
              <div className="space-y-4">
                <div className="text-sm">
                  <strong className="text-[hsl(172_63%_28%)]">Centralized</strong>
                  <p className="text-[hsl(215_15%_45%)]">Traditional training on a single machine</p>
                </div>
                <div className="text-sm">
                  <strong className="text-[hsl(172_63%_28%)]">Federated</strong>
                  <p className="text-[hsl(215_15%_45%)]">Distributed training across multiple clients while preserving privacy</p>
                </div>
              </div>
            </div>
          </div>
          <div className="bg-[hsl(168_40%_95%)] border border-[hsl(168_35%_80%)] rounded-xl p-4">
            <p className="text-sm text-[hsl(168_35%_30%)]">
              <strong className="text-[hsl(168_45%_25%)]">New to ML?</strong> Start with the default values - they're optimized for most use cases!
            </p>
          </div>
        </div>
      ),
      icon: <Settings className="h-10 w-10 text-[hsl(172_63%_28%)]" />,
    },
    {
      title: 'Step 3: Monitor Training',
      description: 'Watch your model learn in real-time',
      content: (
        <div className="space-y-5">
          <div className="flex items-start gap-4 bg-[hsl(168_25%_97%)] p-5 rounded-xl border border-[hsl(168_20%_92%)]">
            <div className="p-2.5 rounded-xl bg-[hsl(172_40%_92%)]">
              <Play className="h-6 w-6 text-[hsl(172_63%_28%)]" />
            </div>
            <div className="flex-1">
              <h4 className="font-semibold text-[hsl(172_43%_20%)] mb-3">Real-time Monitoring</h4>
              <ul className="text-sm text-[hsl(215_15%_45%)] space-y-2">
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)] mt-1.5" />
                  <span><strong className="text-[hsl(172_43%_20%)]">Live metrics:</strong> Track loss, accuracy, and other metrics</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)] mt-1.5" />
                  <span><strong className="text-[hsl(172_43%_20%)]">Progress bars:</strong> See epoch and round progress</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)] mt-1.5" />
                  <span><strong className="text-[hsl(172_43%_20%)]">Status messages:</strong> Get detailed training updates</span>
                </li>
              </ul>
            </div>
          </div>
          <div className="bg-[hsl(172_40%_96%)] border border-[hsl(172_40%_80%)] rounded-xl p-4">
            <p className="text-sm text-[hsl(172_35%_30%)]">
              <strong className="text-[hsl(172_43%_22%)]">Pro tip:</strong> Training can take a while. Feel free to minimize the window - we'll keep you updated!
            </p>
          </div>
        </div>
      ),
      icon: <Play className="h-10 w-10 text-[hsl(172_63%_28%)]" />,
    },
    {
      title: 'Step 4: Analyze Results',
      description: "Understand your model's performance",
      content: (
        <div className="space-y-5">
          <div className="flex items-start gap-4 bg-[hsl(168_25%_97%)] p-5 rounded-xl border border-[hsl(168_20%_92%)]">
            <div className="p-2.5 rounded-xl bg-[hsl(172_40%_92%)]">
              <BarChart3 className="h-6 w-6 text-[hsl(172_63%_28%)]" />
            </div>
            <div className="flex-1">
              <h4 className="font-semibold text-[hsl(172_43%_20%)] mb-3">Comprehensive Analytics</h4>
              <ul className="text-sm text-[hsl(215_15%_45%)] space-y-2">
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)] mt-1.5" />
                  <span><strong className="text-[hsl(172_43%_20%)]">Performance metrics:</strong> Accuracy, precision, recall, F1-score, AUROC</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)] mt-1.5" />
                  <span><strong className="text-[hsl(172_43%_20%)]">Training curves:</strong> Visualize loss and accuracy over time</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)] mt-1.5" />
                  <span><strong className="text-[hsl(172_43%_20%)]">Confusion matrix:</strong> See classification patterns</span>
                </li>
              </ul>
            </div>
          </div>
          <div className="bg-[hsl(172_50%_95%)] border border-[hsl(172_50%_80%)] rounded-xl p-4">
            <p className="text-sm text-[hsl(172_40%_28%)]">
              <strong className="text-[hsl(172_50%_22%)]">Ready to start?</strong> Click "Get Started" to begin your first experiment!
            </p>
          </div>
        </div>
      ),
      icon: <BarChart3 className="h-10 w-10 text-[hsl(172_63%_28%)]" />,
    },
  ];

  const currentStepData = steps[currentStep];
  const isLastStep = currentStep === steps.length - 1;
  const isFirstStep = currentStep === 0;

  const handleNext = () => {
    if (isLastStep) {
      handleClose();
    } else {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleClose = () => {
    setOpen(false);
    setTimeout(onClose, 200);
  };

  const handleSkip = () => {
    handleClose();
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => {
      if (!isOpen) handleClose();
    }}>
      <DialogContent
        className="sm:max-w-2xl max-h-[90vh] overflow-y-auto bg-white/95 backdrop-blur-xl border-[hsl(168_20%_90%)] rounded-2xl shadow-xl"
        style={{ animation: 'fadeIn 0.3s ease-out' }}
      >
        <DialogHeader>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-2xl bg-[hsl(172_40%_94%)]">
                {currentStepData.icon}
              </div>
              <div>
                <DialogTitle className="text-2xl font-semibold text-[hsl(172_43%_15%)]">
                  {currentStepData.title}
                </DialogTitle>
                <DialogDescription className="mt-1 text-[hsl(215_15%_50%)]">
                  {currentStepData.description}
                </DialogDescription>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleSkip}
              className="absolute right-4 top-4 rounded-full hover:bg-[hsl(168_25%_94%)] text-[hsl(215_15%_55%)] hover:text-[hsl(172_43%_25%)]"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        <div className="py-4" style={{ animation: 'fadeIn 0.3s ease-out' }}>
          {currentStepData.content}
        </div>

        {/* Progress indicator */}
        <div className="flex justify-center gap-2 py-4">
          {steps.map((_, index) => (
            <button
              key={index}
              onClick={() => setCurrentStep(index)}
              className={`rounded-full transition-all duration-300 ${
                index === currentStep
                  ? 'w-8 h-2 bg-[hsl(172_63%_28%)]'
                  : index < currentStep
                    ? 'w-2 h-2 bg-[hsl(172_50%_50%)]'
                    : 'w-2 h-2 bg-[hsl(210_15%_85%)]'
              }`}
            />
          ))}
        </div>

        <DialogFooter className="flex justify-between items-center sm:justify-between gap-4">
          <div className="flex gap-2">
            {!isFirstStep && (
              <Button
                variant="outline"
                onClick={handlePrevious}
                className="rounded-xl border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)]"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Previous
              </Button>
            )}
          </div>
          <div className="flex gap-2">
            {!isLastStep && (
              <Button
                variant="ghost"
                onClick={handleSkip}
                className="rounded-xl text-[hsl(215_15%_50%)] hover:bg-[hsl(168_25%_94%)] hover:text-[hsl(172_43%_25%)]"
              >
                Skip Tour
              </Button>
            )}
            <Button
              onClick={handleNext}
              className="rounded-xl bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white shadow-md shadow-[hsl(172_63%_22%)]/20 transition-all hover:shadow-lg hover:-translate-y-0.5"
            >
              {isLastStep ? (
                'Get Started'
              ) : (
                <>
                  Next <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default WelcomeGuide;
