import { CheckCircle2, Lock } from "lucide-react";
import { federatedBenefits, federatedConsiderations } from "./constants";

// Federated Learning SVG Animation Component
const FederatedDiagram = () => (
  <svg viewBox="0 0 300 200" className="absolute inset-0 w-full h-full">
    <defs>
      <linearGradient id="gradientUp" x1="0%" y1="100%" x2="0%" y2="0%">
        <stop offset="0%" stopColor="hsl(172 60% 50%)" />
        <stop offset="100%" stopColor="hsl(172 60% 35%)" />
      </linearGradient>
      <linearGradient id="gradientDown" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor="hsl(152 60% 50%)" />
        <stop offset="100%" stopColor="hsl(152 60% 35%)" />
      </linearGradient>
      <filter id="glowGreen">
        <feGaussianBlur stdDeviation="3" result="coloredBlur" />
        <feMerge>
          <feMergeNode in="coloredBlur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>

    {/* Global server */}
    <g transform="translate(150, 45)">
      <circle className="pulse-ring-fed" r="30" fill="hsl(172 45% 85%)" opacity="0.5">
        <animate attributeName="r" values="30;36;30" dur="2s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.5;0.25;0.5" dur="2s" repeatCount="indefinite" />
      </circle>
      <rect x="-35" y="-18" width="70" height="36" rx="8" fill="white" stroke="hsl(172 50% 65%)" strokeWidth="2.5" filter="url(#glowGreen)" />
      <circle cx="0" cy="0" r="8" fill="none" stroke="hsl(172 63% 35%)" strokeWidth="1.5" />
      <circle cx="-10" cy="-6" r="3" fill="hsl(172 63% 35%)" opacity="0.6" />
      <circle cx="10" cy="-6" r="3" fill="hsl(172 63% 35%)" opacity="0.6" />
      <circle cx="0" cy="8" r="3" fill="hsl(172 63% 35%)" opacity="0.6" />
      <text x="0" y="25" textAnchor="middle" fontSize="10" fontWeight="600" fill="hsl(172 43% 25%)">Global Server</text>
    </g>

    {/* Hospital nodes */}
    {[0, 1, 2].map((i) => (
      <g key={i} transform={`translate(${50 + i * 100}, 155)`}>
        <rect x="-18" y="-18" width="36" height="36" rx="6" fill="white" stroke="hsl(172 40% 75%)" strokeWidth="2" />
        <path d="M-10,-10 L10,-10 M-10,-3 L10,-3 M-10,4 L10,4" stroke="hsl(172 50% 40%)" strokeWidth="1.5" strokeLinecap="round" />
        <g transform="translate(12, -12)">
          <circle r="7" fill="hsl(152 60% 42%)" />
          <path d="M-2,0 L-2,-2 A2,2 0 0,1 2,-2 L2,0" stroke="white" strokeWidth="1.2" fill="none" strokeLinecap="round" />
          <rect x="-2.5" y="0" width="5" height="4" rx="0.5" fill="white" />
        </g>
        <circle cx="0" cy="6" r="2" fill="hsl(152 60% 42%)" className="animate-pulse" />
      </g>
    ))}

    {/* Bidirectional flow paths with animated particles */}
    {[0, 1, 2].map((pathIndex) => {
      const startX = 50 + pathIndex * 100;
      const startY = 137;
      const endX = 150;
      const endY = 63;
      return (
        <g key={pathIndex}>
          <line x1={startX - 6} y1={startY} x2={endX - 6} y2={endY} stroke="hsl(172 50% 75%)" strokeWidth="2" strokeDasharray="5 5" opacity="0.4" />
          <line x1={endX + 6} y1={endY} x2={startX + 6} y2={startY} stroke="hsl(152 50% 70%)" strokeWidth="2" strokeDasharray="5 5" opacity="0.4" />
          {[0, 1].map((p) => (
            <circle key={`up-${p}`} r="3.5" fill="url(#gradientUp)" stroke="white" strokeWidth="1" opacity="0">
              <animateMotion dur="3s" repeatCount="indefinite" begin={`${pathIndex * 0.4 + p * 1.5}s`} path={`M ${startX - 6} ${startY} L ${endX - 6} ${endY}`} />
              <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.85;1" dur="3s" repeatCount="indefinite" begin={`${pathIndex * 0.4 + p * 1.5}s`} />
            </circle>
          ))}
          {[0, 1].map((p) => (
            <circle key={`down-${p}`} r="3.5" fill="url(#gradientDown)" stroke="white" strokeWidth="1" opacity="0">
              <animateMotion dur="3s" repeatCount="indefinite" begin={`${pathIndex * 0.4 + p * 1.5 + 0.3}s`} path={`M ${endX + 6} ${endY} L ${startX + 6} ${startY}`} />
              <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.85;1" dur="3s" repeatCount="indefinite" begin={`${pathIndex * 0.4 + p * 1.5 + 0.3}s`} />
            </circle>
          ))}
        </g>
      );
    })}

    {/* Directional arrows */}
    {[0, 1, 2].map((i) => {
      const startX = 50 + i * 100;
      return (
        <g key={i} opacity="0.5">
          <polygon points={`${startX - 6},95 ${startX - 9},100 ${startX - 3},100`} fill="hsl(172 50% 60%)" />
          <polygon points={`${startX + 6},100 ${startX + 3},95 ${startX + 9},95`} fill="hsl(152 50% 55%)" />
        </g>
      );
    })}

    {/* Step labels */}
    <g transform="translate(50, 185)">
      <rect x="-28" y="-8" width="56" height="16" rx="4" fill="hsl(172 40% 92%)" stroke="hsl(172 50% 75%)" strokeWidth="1" />
      <text x="0" y="4" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(172 43% 30%)">1. Local Train</text>
    </g>
    <g transform="translate(25, 100)">
      <rect x="-30" y="-8" width="60" height="16" rx="4" fill="hsl(172 45% 90%)" stroke="hsl(172 50% 70%)" strokeWidth="1" />
      <text x="0" y="4" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(172 50% 28%)">2. Send Grads</text>
    </g>
    <g transform="translate(150, 12)">
      <rect x="-26" y="-7" width="52" height="14" rx="4" fill="hsl(152 50% 88%)" stroke="hsl(152 55% 65%)" strokeWidth="1" />
      <text x="0" y="3" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(152 60% 28%)">3. Aggregate</text>
    </g>
    <g transform="translate(275, 100)">
      <rect x="-35" y="-8" width="70" height="16" rx="4" fill="hsl(152 45% 90%)" stroke="hsl(152 55% 70%)" strokeWidth="1" />
      <text x="0" y="4" textAnchor="middle" fontSize="8" fontWeight="600" fill="hsl(152 55% 28%)">4. Send Model</text>
    </g>
    <g transform="translate(150, 185)">
      <rect x="-22" y="-8" width="44" height="16" rx="4" fill="hsl(172 35% 95%)" stroke="hsl(172 40% 80%)" strokeWidth="1" strokeDasharray="3 2" />
      <text x="0" y="4" textAnchor="middle" fontSize="7" fontWeight="500" fill="hsl(172 40% 40%)">↻ Repeat</text>
    </g>
  </svg>
);

export const FederatedCard = () => (
  <div className="federated-card relative group">
    <div className="absolute inset-0 bg-gradient-to-br from-[hsl(172_50%_90%)] to-[hsl(168_40%_85%)] rounded-[2rem] blur-xl opacity-50 group-hover:opacity-70 transition-opacity" />
    <div className="relative bg-white/90 backdrop-blur-sm rounded-[2rem] p-10 border border-[hsl(172_30%_88%)] shadow-lg hover:shadow-2xl transition-all duration-500">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[hsl(172_40%_92%)] text-[hsl(172_63%_28%)] text-sm font-medium mb-4">Privacy-First</div>
          <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)]">Federated Learning</h3>
        </div>
        <div className="p-3 rounded-2xl bg-[hsl(172_40%_94%)]">
          <svg className="w-8 h-8 text-[hsl(172_63%_28%)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="12" cy="12" r="3" />
            <circle cx="5" cy="5" r="2" />
            <circle cx="19" cy="5" r="2" />
            <circle cx="5" cy="19" r="2" />
            <circle cx="19" cy="19" r="2" />
            <path d="M12 9V7M12 17v-2M9 12H7m10 0h-2" />
            <path d="M6.5 6.5l3 3m5 5l3 3M17.5 6.5l-3 3m-5 5l-3 3" />
          </svg>
        </div>
      </div>

      {/* Diagram */}
      <div className="federated-diagram relative h-64 mb-8 rounded-2xl bg-gradient-to-br from-[hsl(172_30%_97%)] to-[hsl(168_40%_93%)] overflow-hidden">
        <img src="/images/distributed_learning.png" alt="Federated learning diagram" className="absolute inset-0 w-full h-full object-contain p-2" />
        <div className="absolute inset-0 bg-gradient-to-t from-[hsl(168_40%_93%)]/60 via-transparent to-transparent" />
        <div className="absolute inset-0 opacity-0 pointer-events-none" aria-hidden="true">
          <FederatedDiagram />
        </div>
      </div>

      {/* Benefits */}
      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-[hsl(152_60%_35%)] uppercase tracking-wide flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4" />
          Advantages
        </h4>
        <ul className="space-y-2">
          {federatedBenefits.map((item, i) => (
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
          {federatedConsiderations.map((item, i) => (
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
