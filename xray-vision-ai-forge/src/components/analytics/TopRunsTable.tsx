import { motion, AnimatePresence } from "framer-motion";
import { Trophy } from "lucide-react";
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
import { TopRun, SortMetric, SortDirection } from "./types";
import { formatPercentage, formatDuration, formatDate } from "@/utils/formatters";
import { cardHoverVariants, tableRowVariants, itemVariants } from "./animationVariants";

/**
 * Props for the TopRunsTable component.
 */
interface TopRunsTableProps {
  /** Array of top performing runs sorted by selected metric */
  sortedTopRuns: TopRun[];
  /** Currently selected sort metric */
  sortBy: SortMetric;
  /** Current sort direction (ascending or descending) */
  sortDirection: SortDirection;
  /** Handler function to change sort metric */
  handleSort: (metric: SortMetric) => void;
  /** Function to get sort icon for a given metric */
  getSortIcon: (metric: SortMetric) => React.ReactNode;
}

/**
 * Table component displaying top performing training runs with sortable metrics.
 *
 * Features:
 * - Sortable columns for accuracy, precision, recall, and F1 score
 * - Animated row entrance with staggered delays
 * - Color-coded badges for training mode and status
 * - Formatted metrics and dates
 *
 * @param props - Component properties
 * @returns Rendered top runs table component
 */
export function TopRunsTable({
  sortedTopRuns,
  sortBy,
  sortDirection,
  handleSort,
  getSortIcon,
}: TopRunsTableProps) {
  return (
    <motion.div variants={itemVariants} initial="rest" whileHover="hover">
      <motion.div variants={cardHoverVariants}>
        <Card className="p-6 transition-shadow hover:shadow-xl">
          <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
            <Trophy className="w-5 h-5" />
            Top Performing Runs
          </h2>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="font-semibold w-12">#</TableHead>
                <TableHead className="font-semibold">Run ID</TableHead>
                <TableHead className="font-semibold">Mode</TableHead>
                <TableHead
                  className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                  onClick={() => handleSort("best_accuracy")}
                >
                  Accuracy {getSortIcon("best_accuracy")}
                </TableHead>
                <TableHead
                  className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                  onClick={() => handleSort("best_precision")}
                >
                  Precision {getSortIcon("best_precision")}
                </TableHead>
                <TableHead
                  className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                  onClick={() => handleSort("best_recall")}
                >
                  Recall {getSortIcon("best_recall")}
                </TableHead>
                <TableHead
                  className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                  onClick={() => handleSort("best_f1")}
                >
                  F1 Score {getSortIcon("best_f1")}
                </TableHead>
                <TableHead className="font-semibold text-right">
                  Duration
                </TableHead>
                <TableHead className="font-semibold">Status</TableHead>
                <TableHead className="font-semibold">Date</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <AnimatePresence mode="wait">
                {sortedTopRuns.map((run, index) => (
                  <motion.tr
                    key={run.run_id}
                    custom={index}
                    variants={tableRowVariants}
                    initial="hidden"
                    animate="visible"
                    exit={{ opacity: 0, x: -20 }}
                    className="hover:bg-[hsl(168,20%,98%)] transition-colors"
                  >
                    <TableCell className="font-medium text-[hsl(215,15%,50%)]">
                      {index + 1}
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {run.run_id}
                    </TableCell>
                    <TableCell>
                      <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: index * 0.05 + 0.2 }}
                      >
                        <Badge
                          className={
                            run.training_mode === "centralized"
                              ? "bg-[hsl(172,63%,28%)] hover:bg-[hsl(172,63%,22%)]"
                              : "bg-[hsl(210,60%,50%)] hover:bg-[hsl(210,60%,45%)]"
                          }
                        >
                          {run.training_mode === "centralized"
                            ? "Centralized"
                            : "Federated"}
                        </Badge>
                      </motion.div>
                    </TableCell>
                    <TableCell className="text-right font-medium">
                      {formatPercentage(run.best_accuracy)}
                    </TableCell>
                    <TableCell className="text-right font-medium">
                      {formatPercentage(run.best_precision)}
                    </TableCell>
                    <TableCell className="text-right font-medium">
                      {formatPercentage(run.best_recall)}
                    </TableCell>
                    <TableCell className="text-right font-medium">
                      {formatPercentage(run.best_f1)}
                    </TableCell>
                    <TableCell className="text-right text-[hsl(215,15%,50%)]">
                      {formatDuration(run.duration_minutes)}
                    </TableCell>
                    <TableCell>
                      <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: index * 0.05 + 0.3 }}
                      >
                        <Badge
                          className={
                            run.status === "completed"
                              ? "bg-[hsl(168,40%,45%)] hover:bg-[hsl(168,40%,40%)]"
                              : run.status === "failed"
                                ? "bg-red-500 hover:bg-red-600"
                                : "bg-yellow-500 hover:bg-yellow-600"
                          }
                        >
                          {run.status === "completed"
                            ? "Completed"
                            : run.status === "failed"
                              ? "Failed"
                              : "In Progress"}
                        </Badge>
                      </motion.div>
                    </TableCell>
                    <TableCell className="text-[hsl(215,15%,50%)]">
                      {formatDate(run.start_time)}
                    </TableCell>
                  </motion.tr>
                ))}
              </AnimatePresence>
            </TableBody>
          </Table>
        </Card>
      </motion.div>
    </motion.div>
  );
}
