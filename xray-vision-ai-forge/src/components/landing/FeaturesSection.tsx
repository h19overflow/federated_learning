import { features } from "./constants";

export const FeaturesSection = () => (
  <section className="features-section py-32 px-6 bg-white relative overflow-hidden">
    {/* Subtle hexagon pattern background */}
    <div
      className="absolute inset-0 opacity-[0.35] pointer-events-none"
      style={{
        backgroundImage: "url(/images/hexagon_grid.png)",
        backgroundSize: "400px 400px",
        backgroundRepeat: "repeat",
        maskImage: "radial-gradient(ellipse at center, rgba(0,0,0,0.5) 0%, transparent 70%)",
        WebkitMaskImage: "radial-gradient(ellipse at center, rgba(0,0,0,0.5) 0%, transparent 70%)",
      }}
    />
    <div className="max-w-6xl mx-auto relative z-10">
      <div className="features-title text-center mb-20">
        <h2 className="text-4xl md:text-5xl font-semibold text-[hsl(172_43%_15%)] mb-6">Why Choose Our Platform?</h2>
        <p className="text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto">Enterprise-grade AI training with institutional trust.</p>
      </div>

      <div className="features-grid grid grid-cols-1 md:grid-cols-3 gap-8">
        {features.map((feature, index) => (
          <div
            key={index}
            className="feature-card group p-8 rounded-3xl bg-[hsl(168_25%_98%)] border border-[hsl(168_20%_92%)] hover:bg-white hover:shadow-xl hover:shadow-[hsl(172_40%_85%)]/30 transition-all duration-500 hover:-translate-y-1"
          >
            <div className="mb-6 p-4 rounded-2xl bg-white inline-block shadow-sm group-hover:shadow-md transition-shadow">
              {feature.icon}
            </div>
            <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)] mb-3">{feature.title}</h3>
            <p className="text-[hsl(215_15%_45%)] leading-relaxed">{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  </section>
);
