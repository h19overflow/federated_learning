import { RefObject } from "react";
import { CentralizedCard } from "./CentralizedCard";
import { FederatedCard } from "./FederatedCard";
import { ComparisonTable } from "./ComparisonTable";

interface ComparisonSectionProps {
  comparisonRef: RefObject<HTMLElement | null>;
}

export const ComparisonSection = ({ comparisonRef }: ComparisonSectionProps) => (
  <section ref={comparisonRef} className="comparison-section py-32 px-6 bg-trust-gradient relative overflow-hidden">
    <div className="absolute inset-0 noise-overlay" />
    <div
      className="absolute inset-0 opacity-20 pointer-events-none"
      style={{
        backgroundImage: "url(/images/hexagon_grid.png)",
        backgroundSize: "350px 350px",
        backgroundRepeat: "repeat",
        mixBlendMode: "multiply",
      }}
    />

    <div className="relative z-10 max-w-6xl mx-auto">
      <div className="comparison-title text-center mb-20">
        <h2 className="text-4xl md:text-5xl font-semibold text-[hsl(172_43%_15%)] mb-6">Choose Your Approach</h2>
        <p className="text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto">Two powerful methodologies, one exceptional outcome.</p>
      </div>

      <div className="comparison-cards grid grid-cols-1 lg:grid-cols-2 gap-8">
        <CentralizedCard />
        <FederatedCard />
      </div>

      <ComparisonTable />
    </div>
  </section>
);
