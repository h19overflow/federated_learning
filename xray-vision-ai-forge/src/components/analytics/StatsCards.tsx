import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { LucideIcon } from "lucide-react";
import { Card } from "@/components/ui/card";
import { itemVariants, cardHoverVariants } from "./animationVariants";

interface StatCard {
  icon: LucideIcon;
  label: string;
  value: number | string;
  color: string;
}

interface StatsCardsProps {
  statsCards: StatCard[];
}

export function StatsCards({ statsCards }: StatsCardsProps): JSX.Element {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <AnimatePresence mode="wait">
        {statsCards.map((card, index) => (
          <motion.div
            key={card.label}
            custom={index}
            variants={itemVariants}
            initial="rest"
            whileHover="hover"
          >
            <motion.div variants={cardHoverVariants}>
              <Card className="p-6 border-[hsl(210,15%,92%)] transition-shadow hover:shadow-xl">
                <div className="flex items-center gap-4">
                  <motion.div
                    className="p-3 rounded-xl bg-[hsl(168,20%,95%)]"
                    whileHover={{ rotate: 5, scale: 1.1 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <card.icon
                      className="w-6 h-6"
                      style={{ color: card.color }}
                    />
                  </motion.div>
                  <div>
                    <p className="text-sm text-[hsl(215,15%,50%)]">
                      {card.label}
                    </p>
                    <motion.p
                      className="text-3xl font-bold text-[hsl(172,43%,15%)]"
                      initial={{ scale: 0.5, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{
                        delay: index * 0.1 + 0.2,
                        type: "spring",
                      }}
                    >
                      {card.value}
                    </motion.p>
                  </div>
                </div>
              </Card>
            </motion.div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
