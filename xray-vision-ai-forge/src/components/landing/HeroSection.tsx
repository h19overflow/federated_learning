import { ArrowRight, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeroSectionProps {
  onGetStarted: () => void;
  onLearnMore: () => void;
}

export const HeroSection = ({ onGetStarted, onLearnMore }: HeroSectionProps) => (
  <section className="relative min-h-[90vh] flex flex-col items-center justify-center px-6 overflow-hidden">
    {/* Background elements */}
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-[hsl(172_40%_85%)] rounded-full blur-[120px] opacity-30" />
      <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-[hsl(210_60%_90%)] rounded-full blur-[100px] opacity-25" />
    </div>

    {/* Hero AI Lung Visualization */}
    <div className="absolute right-[-5%] top-1/2 -translate-y-1/2 w-[50%] h-[80%] pointer-events-none hidden lg:block">
      <div
        className="relative w-full h-full"
        style={{
          maskImage: "radial-gradient(ellipse 80% 70% at 70% 50%, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.4) 50%, transparent 80%)",
          WebkitMaskImage: "radial-gradient(ellipse 80% 70% at 70% 50%, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.4) 50%, transparent 80%)",
        }}
      >
        <div className="absolute inset-0 overflow-hidden">
          <img
            src="/images/Lungs.png"
            alt="AI-powered neural network visualization of lung analysis"
            className="w-[115%] h-[115%] object-cover object-left-top opacity-50"
          />
        </div>
        <div className="absolute inset-0 bg-[hsl(172_50%_60%)]/10 mix-blend-overlay" />
      </div>
    </div>

    <div className="relative z-10 max-w-4xl mx-auto text-center">
      {/* Trust badge */}
      <div className="hero-trust-badge inline-flex items-center gap-2 px-4 py-2 mb-8 rounded-full bg-white/60 backdrop-blur-sm border border-[hsl(172_30%_85%)] shadow-sm">
        <div className="w-2 h-2 rounded-full bg-[hsl(152_60%_42%)] animate-pulse" />
        <span className="text-sm font-medium text-[hsl(172_43%_25%)]">Powered by ResNet50 V2 & Flower Framework</span>
      </div>

      {/* Main headline */}
      <h1 className="hero-headline text-5xl md:text-7xl font-semibold tracking-tight text-[hsl(172_43%_15%)] mb-6">
        <span className="word inline-block">Medical</span>{" "}
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
          onClick={onGetStarted}
        >
          Start Training
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
        <Button
          size="lg"
          variant="outline"
          className="hero-cta-button text-lg px-10 py-7 rounded-2xl border-2 border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)] transition-all duration-300"
          onClick={onLearnMore}
        >
          Learn More
        </Button>
      </div>
    </div>

    {/* Scroll indicator */}
    <div className="hero-scroll-indicator absolute bottom-10 left-1/2 -translate-x-1/2">
      <button onClick={onLearnMore} className="flex flex-col items-center gap-2 text-[hsl(215_15%_55%)] hover:text-[hsl(172_63%_28%)] transition-colors">
        <span className="text-sm font-medium">Explore</span>
        <ChevronDown className="h-5 w-5 animate-bounce" />
      </button>
    </div>
  </section>
);
