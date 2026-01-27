import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CTASectionProps {
  onGetStarted: () => void;
}

export const CTASection = ({ onGetStarted }: CTASectionProps) => (
  <section className="cta-section py-32 px-6 bg-[hsl(172_63%_22%)] relative overflow-hidden">
    {/* Background elements */}
    <div className="absolute inset-0 overflow-hidden">
      <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-[hsl(172_55%_28%)] rounded-full blur-[150px] opacity-50" />
      <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-[hsl(172_70%_18%)] rounded-full blur-[120px] opacity-40" />
    </div>

    {/* Health Orb Background */}
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      <div className="absolute inset-0 flex items-center justify-center">
        <div
          className="w-full h-full max-w-[1200px] max-h-[800px]"
          style={{
            maskImage: "radial-gradient(ellipse 60% 80% at 50% 50%, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.3) 50%, transparent 80%)",
            WebkitMaskImage: "radial-gradient(ellipse 60% 80% at 50% 50%, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.3) 50%, transparent 80%)",
          }}
        >
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
      {/* Medical cross icon */}
      <div className="cta-icon mb-8 mx-auto w-20 h-20 rounded-3xl bg-white/10 backdrop-blur flex items-center justify-center">
        <svg className="w-10 h-10 text-white" viewBox="0 0 40 40" fill="none">
          <path d="M20 8v24M8 20h24" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
          <circle cx="20" cy="20" r="16" stroke="currentColor" strokeWidth="2" opacity="0.3" />
        </svg>
      </div>

      <div className="cta-content">
        <h2 className="text-4xl md:text-5xl font-semibold text-white mb-6">Ready to Train Your Model?</h2>
        <p className="text-xl text-white/80 mb-12 max-w-xl mx-auto">
          Start detecting pneumonia with state-of-the-art machine learning, powered by privacy-preserving technology.
        </p>
      </div>

      <Button
        size="lg"
        className="cta-button bg-white text-[hsl(172_63%_22%)] hover:bg-white/90 text-lg px-12 py-7 rounded-2xl shadow-xl shadow-black/20 transition-all duration-300 hover:shadow-2xl hover:-translate-y-0.5 font-semibold"
        onClick={onGetStarted}
      >
        Get Started Now
        <ArrowRight className="ml-2 h-5 w-5" />
      </Button>
    </div>
  </section>
);
