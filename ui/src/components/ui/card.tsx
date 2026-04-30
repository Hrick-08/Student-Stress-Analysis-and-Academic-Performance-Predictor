import { cva, type VariantProps } from "class-variance-authority"
import { motion, type HTMLMotionProps } from "framer-motion"
import * as React from "react"
import { cn } from "@/lib/utils"

const cardVariants = cva(
  "rounded-xl border border-white/10 bg-card/80 backdrop-blur-sm text-card-foreground shadow-lg transition-all duration-300",
  {
    variants: {
      variant: {
        default: "hover:border-white/20",
        glow: "hover:border-primary/50 hover:shadow-primary/20 hover:shadow-xl",
        glass: "glass-card hover:bg-card/90",
      },
      hover3d: {
        false: "",
        true: "perspective-1000",
      },
    },
    defaultVariants: {
      variant: "default",
      hover3d: false,
    },
  }
)

export interface MotionCardProps
  extends Omit<HTMLMotionProps<"div">, "children">,
    VariantProps<typeof cardVariants> {
  children?: React.ReactNode
}

const MotionCard = React.forwardRef<HTMLDivElement, MotionCardProps>(
  ({ className, variant, hover3d, children, ...props }, ref) => {
    if (hover3d) {
      return (
        <motion.div
          ref={ref}
          className={cn(cardVariants({ variant, hover3d: false, className }))}
          initial={{ rotateX: 0, rotateY: 0, scale: 1 }}
          whileHover={{
            rotateX: -5,
            rotateY: 5,
            scale: 1.02,
            transition: { type: "spring", stiffness: 300, damping: 20 },
          }}
          whileTap={{ scale: 0.98 }}
          style={{ perspective: 1000, transformStyle: "preserve-3d" }}
          {...props}
        >
          <motion.div style={{ transform: "translateZ(20px)" }}>
            {children}
          </motion.div>
        </motion.div>
      )
    }

    return (
      <motion.div
        ref={ref}
        className={cn(cardVariants({ variant, hover3d, className }))}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 100, damping: 20 }}
        {...props}
      >
        {children}
      </motion.div>
    )
  }
)
MotionCard.displayName = "MotionCard"

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof cardVariants>
>(({ className, variant, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(cardVariants({ variant, className }))}
    {...props}
  />
))
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn("font-semibold leading-none tracking-tight", className)}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export {
  MotionCard,
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
}