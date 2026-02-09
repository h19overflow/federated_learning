import React from "react";
import { Button } from "@/components/ui/button";
import { Download, FileText } from "lucide-react";

interface DownloadSectionProps {
  handleDownload: (format: string) => void;
}

const DownloadSection: React.FC<DownloadSectionProps> = ({ handleDownload }) => {
  return (
    <div className="px-6 py-6 md:px-8 md:py-8 bg-gradient-to-br from-[hsl(168_25%_99%)] via-white to-[hsl(172_30%_98%)] border-t border-[hsl(210_15%_92%)]">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
        {/* Left: Title and Description */}
        <div className="flex-1 space-y-2">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-[hsl(172_40%_95%)] to-white border border-[hsl(172_30%_88%)] shadow-sm">
              <Download className="h-5 w-5 text-[hsl(172_63%_40%)]" />
            </div>
            <h3 className="text-base font-bold text-[hsl(172_43%_18%)]">
              Download Results
            </h3>
          </div>
          <p className="text-sm text-[hsl(215_15%_50%)] pl-[52px] leading-relaxed">
            Export epoch-by-epoch training and validation metrics as CSV for further analysis in your preferred tools.
          </p>
        </div>

        {/* Right: Download Button */}
        <Button
          variant="outline"
          onClick={() => handleDownload("csv")}
          className="group relative overflow-hidden rounded-xl border-[hsl(172_30%_82%)] bg-white text-[hsl(172_43%_25%)] hover:text-[hsl(172_63%_25%)] hover:border-[hsl(172_40%_72%)] hover:shadow-lg hover:shadow-[hsl(172_30%_90%)]/50 transition-all duration-300 px-5 py-5 h-auto"
        >
          <span className="absolute inset-0 bg-gradient-to-r from-[hsl(172_40%_97%)] to-[hsl(168_30%_98%)] opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          <span className="relative flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-[hsl(172_40%_95%)] group-hover:bg-[hsl(172_40%_92%)] transition-colors">
              <FileText className="h-4 w-4 text-[hsl(172_63%_40%)]" />
            </div>
            <span className="font-semibold">Export to CSV</span>
          </span>
        </Button>
      </div>
    </div>
  );
};

export default DownloadSection;
