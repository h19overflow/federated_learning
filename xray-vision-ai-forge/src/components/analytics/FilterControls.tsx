import React, { startTransition } from "react";
import { motion } from "framer-motion";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { itemVariants } from "./animationVariants";

interface FilterControlsProps {
  trainingMode: string;
  setTrainingMode: (value: string) => void;
  status: string;
  setStatus: (value: string) => void;
  days: string;
  setDays: (value: string) => void;
}

export function FilterControls({
  trainingMode,
  setTrainingMode,
  status,
  setStatus,
  days,
  setDays,
}: FilterControlsProps) {
  return (
    <motion.div variants={itemVariants} className="flex flex-wrap gap-4">
      <div className="flex-1 min-w-[200px]">
        <label className="text-sm font-medium text-[hsl(172,43%,15%)] mb-2 block">
          Training Mode
        </label>
        <Select
          value={trainingMode}
          onValueChange={(value) =>
            startTransition(() => setTrainingMode(value))
          }
        >
          <SelectTrigger className="w-full transition-all hover:border-[hsl(172,63%,28%)]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Modes</SelectItem>
            <SelectItem value="centralized">Centralized</SelectItem>
            <SelectItem value="federated">Federated</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="flex-1 min-w-[200px]">
        <label className="text-sm font-medium text-[hsl(172,43%,15%)] mb-2 block">
          Status
        </label>
        <Select
          value={status}
          onValueChange={(value) => startTransition(() => setStatus(value))}
        >
          <SelectTrigger className="w-full transition-all hover:border-[hsl(172,63%,28%)]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="completed">Completed</SelectItem>
            <SelectItem value="failed">Failed</SelectItem>
            <SelectItem value="in_progress">In Progress</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="flex-1 min-w-[200px]">
        <label className="text-sm font-medium text-[hsl(172,43%,15%)] mb-2 block">
          Date Range
        </label>
        <Select
          value={days}
          onValueChange={(value) => startTransition(() => setDays(value))}
        >
          <SelectTrigger className="w-full transition-all hover:border-[hsl(172,63%,28%)]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Time</SelectItem>
            <SelectItem value="7">Last 7 Days</SelectItem>
            <SelectItem value="30">Last 30 Days</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </motion.div>
  );
}
