import React from "react";
import { MetadataDisplay } from "@/components/analytics";
import type { ExperimentResults } from "@/types/api";

interface DetailsTabProps {
  centralizedResults: ExperimentResults | null;
  federatedResults: ExperimentResults | null;
}

const DetailsTab: React.FC<DetailsTabProps> = ({
  centralizedResults,
  federatedResults,
}) => {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {centralizedResults && (
          <MetadataDisplay
            metadata={centralizedResults.metadata}
            title="Centralized Experiment"
          />
        )}
        {federatedResults && (
          <MetadataDisplay
            metadata={federatedResults.metadata}
            title="Federated Experiment"
          />
        )}
      </div>
    </div>
  );
};

export default DetailsTab;
