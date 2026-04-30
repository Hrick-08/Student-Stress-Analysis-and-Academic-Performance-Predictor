import { motion } from "framer-motion"
import { Brain, ArrowRight, Sparkles } from "lucide-react"

interface HeroSectionProps {
  onGetStarted?: () => void
}

export function HeroSection({ onGetStarted }: HeroSectionProps) {
  return (
    <section id="home" className="min-h-screen flex items-center pt-20">
      <div className="container mx-auto px-4">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            {/* Animated Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-6"
            >
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span className="text-sm text-cyan-300">AI-Powered Prediction</span>
            </motion.div>

            {/* Main Title */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-5xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight"
            >
              <span className="text-white">Predict Student</span>
              <br />
              <span className="gradient-text">Stress & Performance</span>
              <br />
              <span className="text-white">with AI</span>
            </motion.h1>

            {/* Description */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-lg text-muted-foreground mb-8 max-w-xl"
            >
              Leverage machine learning to analyze student data and predict academic performance. 
              Make informed decisions to support student well-being and success.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="flex flex-wrap gap-4"
            >
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={onGetStarted}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-purple-600 text-white font-semibold shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-all"
              >
                <Brain className="w-5 h-5" />
                Get Started
                <ArrowRight className="w-4 h-4" />
              </motion.button>
              <motion.a
                href="#features"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-white/5 border border-white/10 text-white font-semibold hover:bg-white/10 transition-all"
              >
                Learn More
              </motion.a>
            </motion.div>
          </motion.div>

          {/* Right Graphic - 3D Brain Illustration */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="hidden lg:flex items-center justify-center"
          >
            <motion.div
              animate={{ 
                rotateX: [0, 5, 0],
                rotateY: [0, -5, 0],
              }}
              transition={{ 
                duration: 6, 
                repeat: Infinity, 
                ease: "easeInOut" 
              }}
              className="relative w-80 h-80"
            >
              {/* Glowing Brain Representation */}
              <div className="absolute inset-0 rounded-full bg-gradient-to-br from-cyan-500/20 to-purple-600/20 blur-3xl" />
              
              {/* Brain Icon with Glow */}
              <motion.div
                animate={{ 
                  boxShadow: [
                    "0 0 60px rgba(34, 211, 238, 0.4)",
                    "0 0 80px rgba(168, 85, 247, 0.4)",
                    "0 0 60px rgba(34, 211, 238, 0.4)",
                  ]
                }}
                transition={{ duration: 3, repeat: Infinity }}
                className="absolute inset-10 rounded-full bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center"
              >
                <Brain className="w-32 h-32 text-white" />
              </motion.div>

              {/* Floating Particles */}
              {[...Array(6)].map((_, i) => (
                <motion.div
                  key={i}
                  className="absolute w-3 h-3 rounded-full"
                  style={{
                    background: i % 2 === 0 ? "#22d3ee" : "#a855f7",
                    left: `${20 + i * 12}%`,
                    top: `${10 + (i % 3) * 30}%`,
                  }}
                  animate={{
                    y: [0, -20, 0],
                    opacity: [0.5, 1, 0.5],
                  }}
                  transition={{
                    duration: 2 + i * 0.3,
                    repeat: Infinity,
                    delay: i * 0.2,
                  }}
                />
              ))}
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}