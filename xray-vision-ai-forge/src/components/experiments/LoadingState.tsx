import { Loader2 } from "lucide-react";

export const LoadingState = () => (
  <div className="flex flex-col items-center justify-center py-32 animate-fade-in">
    <div className="relative w-20 h-20 mb-8">
      <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-[hsl(172_63%_22%)]/10 to-[hsl(172_63%_22%)]/5 blur-xl" />
      <div className="relative w-full h-full rounded-3xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(168_25%_96%)] flex items-center justify-center border border-[hsl(172_40%_88%)] shadow-lg">
        <Loader2 className="h-9 w-9 text-[hsl(172_63%_35%)] animate-spin" />
      </div>
    </div>
    <p className="text-lg font-semibold text-[hsl(172_43%_20%)] mb-2">Loading experiments</p>
    <p className="text-sm text-[hsl(215_15%_55%)]">Fetching your training runs...</p>
  </div>
);
