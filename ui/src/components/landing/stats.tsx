import { motion } from "framer-motion"
import { Brain, Target } from "lucide-react"

const stats = [
  {
    label: "ML Models",
    value: "6",
    suffix: "",
    description: "Machine Learning Models",
    icon: Brain,
    color: "cyan",
  },
  {
    label: "Accuracy",
    value: "95",
    suffix: "%+",
    description: "Prediction Accuracy",
    icon: Target,
    color: "purple",
  },
]

export function StatsSection() {
  return (
    <section className="py-16">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-2 gap-4 max-w-2xl mx-auto">
          {stats.map((stat, i) => {
            const Icon = stat.icon
            return (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="glass-card rounded-2xl p-6 text-center"
              >
                <motion.div
                  initial={{ scale: 0 }}
                  whileInView={{ scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.3 + i * 0.1, type: "spring" }}
                  className={`inline-flex p-3 rounded-xl mb-4 ${
                    stat.color === "cyan" 
                      ? "bg-cyan-500/20 text-cyan-400" 
                      : "bg-purple-500/20 text-purple-400"
                  }`}
                >
                  <Icon className="w-6 h-6" />
                </motion.div>
                
                <motion.div
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.4 + i * 0.1 }}
                  className="text-4xl font-bold gradient-text mb-1"
                >
                  {stat.value}
                  <span className="text-2xl">{stat.suffix}</span>
                </motion.div>
                
                <div className="text-sm text-muted-foreground">
                  {stat.description}
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}