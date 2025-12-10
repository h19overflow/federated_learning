import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { 
  Upload, 
  Settings, 
  Play, 
  BarChart3,
  CheckCircle2,
  ArrowRight,
  X
} from 'lucide-react';

interface WelcomeGuideProps {
  onClose: () => void;
}

const WelcomeGuide = ({ onClose }: WelcomeGuideProps) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [open, setOpen] = useState(true);

  const steps = [
    {
      title: 'Welcome to XRay Vision AI Forge! 👋',
      description: 'Your platform for training pneumonia detection models using state-of-the-art machine learning',
      content: (
        <div className="space-y-4">
          <p className="text-muted-foreground">
            This platform helps you train deep learning models for pneumonia detection from chest X-rays.
            You can choose between traditional centralized training or privacy-preserving federated learning.
          </p>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2">What you'll learn:</h4>
            <ul className="list-disc list-inside space-y-1 text-sm text-blue-800">
              <li>How to upload and prepare your dataset</li>
              <li>Configure training parameters</li>
              <li>Monitor training in real-time</li>
              <li>Analyze and compare results</li>
            </ul>
          </div>
        </div>
      ),
      icon: <CheckCircle2 className="h-12 w-12 text-medical" />,
    },
    {
      title: 'Step 1: Upload Dataset 📁',
      description: 'Start by uploading your chest X-ray dataset',
      content: (
        <div className="space-y-4">
          <div className="flex items-start gap-4 bg-gray-50 p-4 rounded-lg">
            <Upload className="h-8 w-8 text-medical flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h4 className="font-semibold mb-2">Dataset Format</h4>
              <p className="text-sm text-muted-foreground mb-3">
                Upload a ZIP file containing:
              </p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• <strong>Images folder:</strong> X-ray images organized by class</li>
                <li>• <strong>Metadata CSV:</strong> Image labels and information</li>
              </ul>
            </div>
          </div>
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
            <p className="text-sm text-yellow-900">
              <strong>Tip:</strong> Configure the train/validation split to control how much data is used for training vs. testing.
            </p>
          </div>
        </div>
      ),
      icon: <Upload className="h-12 w-12 text-medical" />,
    },
    {
      title: 'Step 2: Configure Experiment ⚙️',
      description: 'Set up your training parameters',
      content: (
        <div className="space-y-4">
          <div className="flex items-start gap-4 bg-gray-50 p-4 rounded-lg">
            <Settings className="h-8 w-8 text-medical flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h4 className="font-semibold mb-2">Training Modes</h4>
              <div className="space-y-3">
                <div className="text-sm">
                  <strong className="text-medical">Centralized:</strong>
                  <p className="text-muted-foreground">Traditional training on a single machine</p>
                </div>
                <div className="text-sm">
                  <strong className="text-medical">Federated:</strong>
                  <p className="text-muted-foreground">Distributed training across multiple clients while preserving privacy</p>
                </div>
                <div className="text-sm">
                  <strong className="text-medical">Both:</strong>
                  <p className="text-muted-foreground">Compare both approaches side-by-side</p>
                </div>
              </div>
            </div>
          </div>
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
            <p className="text-sm text-purple-900">
              <strong>New to ML?</strong> Start with the default values - they're optimized for most use cases!
            </p>
          </div>
        </div>
      ),
      icon: <Settings className="h-12 w-12 text-medical" />,
    },
    {
      title: 'Step 3: Monitor Training 🚀',
      description: 'Watch your model learn in real-time',
      content: (
        <div className="space-y-4">
          <div className="flex items-start gap-4 bg-gray-50 p-4 rounded-lg">
            <Play className="h-8 w-8 text-medical flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h4 className="font-semibold mb-2">Real-time Monitoring</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <strong>Live metrics:</strong> Track loss, accuracy, and other metrics</li>
                <li>• <strong>Progress bars:</strong> See epoch and round progress</li>
                <li>• <strong>Federated updates:</strong> Watch global model aggregation</li>
                <li>• <strong>Status messages:</strong> Get detailed training updates</li>
              </ul>
            </div>
          </div>
          <div className="bg-teal-50 border border-teal-200 rounded-lg p-3">
            <p className="text-sm text-teal-900">
              <strong>Pro tip:</strong> Training can take a while. Feel free to minimize the window - we'll keep you updated!
            </p>
          </div>
        </div>
      ),
      icon: <Play className="h-12 w-12 text-medical" />,
    },
    {
      title: 'Step 4: Analyze Results 📊',
      description: 'Understand your model\'s performance',
      content: (
        <div className="space-y-4">
          <div className="flex items-start gap-4 bg-gray-50 p-4 rounded-lg">
            <BarChart3 className="h-8 w-8 text-medical flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h4 className="font-semibold mb-2">Comprehensive Analytics</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <strong>Performance metrics:</strong> Accuracy, precision, recall, F1-score, AUROC</li>
                <li>• <strong>Training curves:</strong> Visualize loss and accuracy over time</li>
                <li>• <strong>Confusion matrix:</strong> See classification patterns</li>
                <li>• <strong>ROC curve:</strong> Evaluate model discrimination ability</li>
              </ul>
            </div>
          </div>
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <p className="text-sm text-green-900">
              <strong>Ready to start?</strong> Click "Get Started" to begin your first experiment!
            </p>
          </div>
        </div>
      ),
      icon: <BarChart3 className="h-12 w-12 text-medical" />,
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
    setTimeout(onClose, 200); // Wait for animation
  };

  const handleSkip = () => {
    handleClose();
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => {
      if (!isOpen) handleClose();
    }}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              {currentStepData.icon}
              <div>
                <DialogTitle className="text-2xl">{currentStepData.title}</DialogTitle>
                <DialogDescription className="mt-2">
                  {currentStepData.description}
                </DialogDescription>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleSkip}
              className="absolute right-4 top-4"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        <div className="py-4">
          {currentStepData.content}
        </div>

        {/* Progress indicator */}
        <div className="flex justify-center gap-2 py-4">
          {steps.map((_, index) => (
            <div
              key={index}
              className={`h-2 rounded-full transition-all ${
                index === currentStep
                  ? 'w-8 bg-medical'
                  : index < currentStep
                  ? 'w-2 bg-medical/50'
                  : 'w-2 bg-gray-300'
              }`}
            />
          ))}
        </div>

        <DialogFooter className="flex justify-between items-center sm:justify-between">
          <div className="flex gap-2">
            {!isFirstStep && (
              <Button variant="outline" onClick={handlePrevious}>
                Previous
              </Button>
            )}
          </div>
          <div className="flex gap-2">
            {!isLastStep && (
              <Button variant="ghost" onClick={handleSkip}>
                Skip Tour
              </Button>
            )}
            <Button onClick={handleNext} className="bg-medical hover:bg-medical-dark">
              {isLastStep ? (
                <>Get Started</>
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

