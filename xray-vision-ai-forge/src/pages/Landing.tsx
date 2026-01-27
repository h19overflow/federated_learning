import { useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Header, Footer, WelcomeGuide } from "@/components/layout";
import {
  useLandingAnimations,
  HeroSection,
  FeaturesSection,
  ComparisonSection,
  HowItWorksSection,
  CTASection,
} from "@/components/landing";

const Landing = () => {
  const navigate = useNavigate();
  const comparisonRef = useRef<HTMLElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  const [showWelcomeGuide, setShowWelcomeGuide] = useState(false);

  const handleGetStarted = useCallback(() => navigate("/experiment"), [navigate]);

  const scrollToComparison = useCallback(() => {
    comparisonRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useLandingAnimations(mainRef);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {showWelcomeGuide && <WelcomeGuide onClose={() => setShowWelcomeGuide(false)} />}
      <Header onShowHelp={() => setShowWelcomeGuide(true)} />

      <main ref={mainRef} className="flex-1 overflow-y-auto bg-hero-gradient">
        <HeroSection onGetStarted={handleGetStarted} onLearnMore={scrollToComparison} />
        <FeaturesSection />
        <ComparisonSection comparisonRef={comparisonRef} />
        <HowItWorksSection />
        <CTASection onGetStarted={handleGetStarted} />
      </main>

      <Footer />
    </div>
  );
};

export default Landing;
