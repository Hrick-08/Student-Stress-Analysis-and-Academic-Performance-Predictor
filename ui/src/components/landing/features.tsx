import { motion } from "framer-motion"
import { Brain, BarChart3 } from "lucide-react"

const features = [
  {
    icon: Brain,
    title: "Real-time Stress Analysis",
    description: "Analyze student stress levels instantly using advanced machine learning algorithms trained on comprehensive datasets.",
    color: "cyan",
  },
  {
    icon: BarChart3,
    title: "Model Comparison",
    description: "Compare 6 different ML models to find the best predictor. Visualize R² scores and RMSE metrics.",
    color: "purple",
  },
  {
    icon: Brain,
    title: "Personalized Insights",
    description: "Get actionable recommendations based on individual student data for better academic outcomes.",
    color: "pink",
  },
]

export function FeaturesSection() {
  return (
    <section id="features" className="py-20">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="gradient-text">Powerful Features</span>
            <span className="text-white"> for Better Predictions</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Our AI-powered platform provides comprehensive tools to analyze and predict student performance.
          </p>
        </motion.div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6">
          {features.map((feature, i) => {
            const Icon = feature.icon
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                whileHover={{ y: -5 }}
                className="glass-card rounded-2xl p-6 group"
              >
                <motion.div
                  className={`inline-flex p-3 rounded-xl mb-4 ${
                    feature.color === "cyan"
                      ? "bg-cyan-500/20 text-cyan-400"
                      : feature.color === "purple"
                      ? "bg-purple-500/20 text-purple-400"
                      : "bg-pink-500/20 text-pink-400"
                  }`}
                  whileHover={{ scale: 1.1 }}
                  transition={{ type: "spring" }}
                >
                  <Icon className="w-6 h-6" />
                </motion.div>
                
                <h3 className="text-xl font-semibold mb-2 text-white group-hover:gradient-text transition-all">
                  {feature.title}
                </h3>
                
                <p className="text-muted-foreground text-sm">
                  {feature.description}
                </p>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}