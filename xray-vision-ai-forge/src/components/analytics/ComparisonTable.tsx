import { motion } from "framer-motion";
import { TrendingUp, ArrowUp } from "lucide-react";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { ModeStatistics } from "./types";
import { formatPercentage, formatDuration } from "@/utils/formatters";
import {
  cardHoverVariants,
  tableRowVariants,
  itemVariants,
} from "./animationVariants";

/**
 * Props for the ComparisonTable component.
 */
interface ComparisonTableProps {
  /** Aggregated statistics for centralized training mode */
  centralizedData: ModeStatistics;
  /** Aggregated statistics for federated training mode */
  federatedData: ModeStatistics;
}

/**
 * Determines which training mode performs better for a given metric.
 *
 * @param centValue - The centralized mode metric value
 * @param fedValue - The federated mode metric value
 * @returns "centralized" if centralized is better, "federated" if federated is better, "tie" if equal
 */
const getWinner = (
  centValue: number,
  fedValue: number
): "centralized" | "federated" | "tie" => {
  if (Math.abs(centValue - fedValue) < 0.001) return "tie";
  return centValue > fedValue ? "centralized" : "federated";
};

/**
 * Comparison table component for displaying centralized vs federated training metrics.
 *
 * This component renders a side-by-side comparison of key performance metrics
 * between centralized and federated training modes, highlighting the winner for each metric.
 * Includes animations for rows and hover effects on the card.
 */
export const ComparisonTable = ({
  centralizedData,
  federatedData,
}: ComparisonTableProps) => {
  const comparisonMetrics = [
    {
      label: "Precision",
      cent: centralizedData.avg_precision,
      fed: federatedData.avg_precision,
    },
    {
      label: "Recall",
      cent: centralizedData.avg_recall,
      fed: federatedData.avg_recall,
    },
    {
      label: "F1 Score",
      cent: centralizedData.avg_f1,
      fed: federatedData.avg_f1,
    },
    {
      label: "Avg Duration",
      cent: centralizedData.avg_duration_minutes,
      fed: federatedData.avg_duration_minutes,
      isDuration: true,
    },
  ];

  return (
    <motion.div variants={itemVariants} initial="rest" whileHover="hover">
      <motion.div variants={cardHoverVariants}>
        <Card className="p-6 transition-shadow hover:shadow-xl">
          <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Centralized vs Federated Comparison
          </h2>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="font-semibold">Metric</TableHead>
                <TableHead className="font-semibold text-center">
                  Centralized
                </TableHead>
                <TableHead className="font-semibold text-center">
                  Federated
                </TableHead>
                <TableHead className="font-semibold text-center">
                  Winner
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {comparisonMetrics.map((row, i) => (
                <motion.tr
                  key={row.label}
                  custom={i}
                  variants={tableRowVariants}
                  initial="hidden"
                  animate="visible"
                  className="hover:bg-[hsl(168,20%,98%)] transition-colors"
                >
                  <TableCell className="font-medium">{row.label}</TableCell>
                  <TableCell className="text-center">
                    {row.isDuration
                      ? formatDuration(row.cent)
                      : formatPercentage(row.cent)}
                  </TableCell>
                  <TableCell className="text-center">
                    {row.isDuration
                      ? formatDuration(row.fed)
                      : formatPercentage(row.fed)}
                  </TableCell>
                  <TableCell className="text-center">
                    {(() => {
                      const winner = row.isDuration
                        ? getWinner(row.fed, row.cent)
                        : getWinner(row.cent, row.fed);
                      if (winner === "tie") return null;
                      return (
                        <motion.div
                          initial={{ scale: 0, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ delay: i * 0.1 + 0.3 }}
                        >
                          <Badge
                            className={
                              winner === "centralized"
                                ? "bg-[hsl(172,63%,28%)] hover:bg-[hsl(172,63%,22%)]"
                                : "bg-[hsl(210,60%,50%)] hover:bg-[hsl(210,60%,45%)]"
                            }
                          >
                            {winner === "centralized" ? "Cent." : "Fed."}{" "}
                            <ArrowUp className="w-3 h-3 ml-1 inline" />
                          </Badge>
                        </motion.div>
                      );
                    })()}
                  </TableCell>
                </motion.tr>
              ))}
            </TableBody>
          </Table>
        </Card>
      </motion.div>
    </motion.div>
  );
};
