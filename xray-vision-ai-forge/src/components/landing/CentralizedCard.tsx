import { CheckCircle2, Lock } from "lucide-react";
import { centralizedBenefits, centralizedConsiderations } from "./constants";

export const CentralizedCard = () => (
  <div className="centralized-card relative group">
    <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_92%)] to-[hsl(168_40%_87%)] rounded-[2rem] blur-xl opacity-50 group-hover:opacity-70 transition-opacity" />
    <div className="relative bg-white/90 backdrop-blur-sm rounded-[2rem] p-10 border border-[hsl(172_30%_88%)] shadow-lg hover:shadow-2xl transition-all duration-500">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[hsl(172_40%_94%)] text-[hsl(172_50%_35%)] text-sm font-medium mb-4">
            Traditional
          </div>
          <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)]">Centralized Learning</h3>
        </div>
        <div className="p-3 rounded-2xl bg-[hsl(172_40%_94%)]">
          <svg className="w-8 h-8 text-[hsl(172_50%_35%)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="4" y="4" width="16" height="16" rx="2" />
            <path d="M4 9h16M9 4v16" />
          </svg>
        </div>
      </div>

      {/* Image */}
      <div className="centralized-diagram relative h-64 mb-8 rounded-2xl bg-gradient-to-br from-[hsl(172_30%_97%)] to-[hsl(168_40%_93%)] overflow-hidden">
        <img
          src="/images/centralizied.png"
          alt="Centralized learning diagram showing hospitals sending data to central server"
          className="absolute inset-0 w-full h-full object-contain p-2"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-[hsl(168_40%_93%)]/60 via-transparent to-transparent" />
      </div>

      {/* Benefits */}
      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-[hsl(152_60%_35%)] uppercase tracking-wide flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4" />
          Advantages
        </h4>
        <ul className="space-y-2">
          {centralizedBenefits.map((item, i) => (
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
          {centralizedConsiderations.map((item, i) => (
            <li key={i} className="flex items-center gap-3 text-[hsl(215_15%_50%)]">
              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(35_70%_50%)]" />
              {item}
            </li>
          ))}
        </ul>
      </div>
    </div>
  </div>
);
