import { comparisonData } from "./constants";

export const ComparisonTable = () => (
  <div className="comparison-table-wrapper mt-16 bg-white rounded-[2rem] p-8 shadow-lg border border-[hsl(210_15%_92%)]">
    <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)] mb-8 text-center">Quick Comparison</h3>
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="comparison-table-row border-b border-[hsl(210_15%_92%)]">
            <th className="py-4 px-6 text-left text-sm font-semibold text-[hsl(172_43%_20%)]">Feature</th>
            <th className="py-4 px-6 text-center text-sm font-semibold text-[hsl(210_60%_40%)]">Centralized</th>
            <th className="py-4 px-6 text-center text-sm font-semibold text-[hsl(172_63%_28%)]">Federated</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[hsl(210_15%_95%)]">
          {comparisonData.map((row, index) => (
            <tr key={index} className="comparison-table-row hover:bg-[hsl(168_25%_98%)] transition-colors">
              <td className="py-4 px-6 text-sm font-medium text-[hsl(172_43%_20%)]">{row.feature}</td>
              <td className="py-4 px-6 text-center">
                <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
                  row.centralized.status === "success" ? "bg-[hsl(152_50%_95%)] text-[hsl(152_60%_30%)]" : "bg-[hsl(35_60%_95%)] text-[hsl(35_70%_35%)]"
                }`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${row.centralized.status === "success" ? "bg-[hsl(152_60%_42%)]" : "bg-[hsl(35_70%_50%)]"}`} />
                  {row.centralized.text}
                </span>
              </td>
              <td className="py-4 px-6 text-center">
                <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
                  row.federated.status === "success" ? "bg-[hsl(152_50%_95%)] text-[hsl(152_60%_30%)]" : "bg-[hsl(35_60%_95%)] text-[hsl(35_70%_35%)]"
                }`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${row.federated.status === "success" ? "bg-[hsl(152_60%_42%)]" : "bg-[hsl(35_70%_50%)]"}`} />
                  {row.federated.text}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);
