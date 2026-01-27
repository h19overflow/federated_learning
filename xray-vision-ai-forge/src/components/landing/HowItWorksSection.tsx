import { howItWorksSteps } from "./constants";

export const HowItWorksSection = () => (
  <section className="how-it-works-section py-32 bg-white relative overflow-hidden">
    {/* Background lungs visual */}
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
      <div
        className="w-[600px] h-[600px] opacity-[0.08]"
        style={{
          maskImage: "radial-gradient(circle at center, rgba(0,0,0,1) 0%, rgba(0,0,0,0.5) 40%, transparent 70%)",
          WebkitMaskImage: "radial-gradient(circle at center, rgba(0,0,0,1) 0%, rgba(0,0,0,0.5) 40%, transparent 70%)",
        }}
      >
        <div className="w-full h-full overflow-hidden">
          <img
            src="/images/lungs2.png"
            alt=""
            className="w-[120%] h-[120%] object-cover object-left-top"
            style={{ filter: "saturate(0.5) contrast(1.1)" }}
          />
        </div>
      </div>
    </div>

    <div className="relative z-10">
      <div className="how-it-works-title text-center mb-16 px-6">
        <h2 className="text-4xl md:text-5xl font-semibold text-[hsl(172_43%_15%)] mb-6">How It Works</h2>
        <p className="text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto">
          From data upload to model comparison — a streamlined ML pipeline.
        </p>
      </div>

      <div className="max-w-6xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {howItWorksSteps.map((item, index) => (
            <div key={index} className="step-card relative group">
              <div className="h-full p-6 rounded-2xl bg-white/80 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-lg hover:shadow-xl hover:shadow-[hsl(172_40%_85%)]/30 transition-all duration-300 hover:-translate-y-1">
                <div className="flex items-center gap-3 mb-4">
                  <div
                    className="text-xs font-bold tracking-widest px-2.5 py-0.5 rounded-full"
                    style={{ backgroundColor: `${item.color}15`, color: item.color }}
                  >
                    STEP {item.step}
                  </div>
                </div>

                <div
                  className="step-icon mb-4 w-14 h-14 rounded-xl flex items-center justify-center transition-all duration-300 group-hover:scale-105"
                  style={{ backgroundColor: `${item.color}12`, color: item.color }}
                >
                  {item.icon}
                </div>

                <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-2">{item.title}</h3>
                <p className="text-[hsl(215_15%_45%)] text-sm leading-relaxed">{item.description}</p>
              </div>

              {index < 3 && (
                <div className="hidden lg:block absolute top-1/2 -right-3 transform -translate-y-1/2 z-10">
                  <svg className="w-5 h-5 text-[hsl(172_40%_75%)]" viewBox="0 0 24 24" fill="none">
                    <path d="M9 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  </section>
);
