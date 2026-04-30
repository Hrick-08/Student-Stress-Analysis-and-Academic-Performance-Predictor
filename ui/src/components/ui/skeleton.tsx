import { motion } from "framer-motion"
import * as React from "react"
import { cn } from "@/lib/utils"

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  animate?: boolean
}

const Skeleton = React.forwardRef<HTMLDivElement, SkeletonProps>(
  ({ className, animate = true, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-md bg-white/10",
        animate && "animate-pulse",
        className
      )}
      {...props}
    />
  )
)
Skeleton.displayName = "Skeleton"

const SkeletonCard = ({ className }: { className?: string }) => (
  <motion.div
    className={cn("rounded-xl border border-white/10 bg-card/50 p-6", className)}
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ duration: 0.3 }}
  >
    <Skeleton className="h-6 w-1/3 mb-4" />
    <Skeleton className="h-4 w-full mb-2" />
    <Skeleton className="h-4 w-4/5 mb-2" />
    <Skeleton className="h-4 w-2/3" />
  </motion.div>
)

const SkeletonChart = ({ className }: { className?: string }) => (
  <motion.div
    className={cn("rounded-xl border border-white/10 bg-card/50 p-6", className)}
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ duration: 0.3 }}
  >
    <Skeleton className="h-8 w-1/2 mb-6" />
    <div className="flex items-end justify-center gap-3 h-[200px]">
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <motion.div
          key={i}
          className="w-8 rounded-t-md bg-white/10"
          initial={{ height: "0%" }}
          animate={{ height: `${20 + i * 12}%` }}
          transition={{ duration: 0.5, delay: i * 0.1 }}
        />
      ))}
    </div>
  </motion.div>
)

const SkeletonForm = ({ className }: { className?: string }) => (
  <motion.div
    className={cn("rounded-xl border border-white/10 bg-card/50 p-6", className)}
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ duration: 0.3 }}
  >
    <Skeleton className="h-8 w-1/3 mb-6" />
    {[1, 2, 3, 4, 5].map((i) => (
      <div key={i} className="mb-4">
        <Skeleton className="h-4 w-1/4 mb-2" />
        <Skeleton className="h-10 w-full" />
      </div>
    ))}
  </motion.div>
)

export { Skeleton, SkeletonCard, SkeletonChart, SkeletonForm }