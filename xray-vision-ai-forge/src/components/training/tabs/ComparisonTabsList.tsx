import React from "react";
import { TabsList, TabsTrigger } from "@/components/ui/tabs";

const ComparisonTabsList: React.FC = () => {
  return (
    <TabsList className="grid grid-cols-4 w-full max-w-lg mx-auto bg-[hsl(168_20%_95%)] p-1 rounded-xl">
      <TabsTrigger value="comparison" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
        Comparison
      </TabsTrigger>
      <TabsTrigger value="centralized" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
        Centralized
      </TabsTrigger>
      <TabsTrigger value="federated" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
        Federated
      </TabsTrigger>
      <TabsTrigger value="details" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
        Details
      </TabsTrigger>
    </TabsList>
  );
};

export default ComparisonTabsList;
