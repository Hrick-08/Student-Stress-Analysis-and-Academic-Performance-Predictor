import { motion } from "framer-motion"
import { Brain, Github, Menu, X } from "lucide-react"
import { useState } from "react"

const navLinks = [
  { name: "Home", href: "#home" },
  { name: "Features", href: "#features" },
  { name: "Team", href: "#team" },
]

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false)

  const handleNavClick = (href: string) => {
    const element = document.querySelector(href)
    if (element) {
      element.scrollIntoView({ behavior: "smooth" })
    }
    setIsOpen(false)
  }

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 left-0 right-0 z-50 backdrop-blur-lg bg-background/80 border-b border-white/10"
    >
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <motion.div
            className="flex items-center gap-2 cursor-pointer"
            whileHover={{ scale: 1.02 }}
            onClick={() => handleNavClick("#home")}
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold gradient-text hidden sm:block">
              StressPredict
            </span>
          </motion.div>

          {/* Desktop Nav Links */}
          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link, i) => (
              <motion.button
                key={link.name}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * i }}
                className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                onClick={() => handleNavClick(link.href)}
              >
                {link.name}
              </motion.button>
            ))}
          </div>

          {/* GitHub Icon Only */}
          <div className="hidden md:flex items-center gap-3">
            <motion.a
              href="https://github.com/Hrick-08/stress-ml"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.1, y: -2 }}
              whileTap={{ scale: 0.95 }}
              className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
            >
              <Github className="w-5 h-5" />
            </motion.a>
          </div>

          {/* Mobile Menu Button */}
          <motion.button
            whileTap={{ scale: 0.95 }}
            className="md:hidden p-2"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </motion.button>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="md:hidden py-4 space-y-4"
          >
            {navLinks.map((link) => (
              <button
                key={link.name}
                className="block py-2 text-sm font-medium w-full text-left"
                onClick={() => handleNavClick(link.href)}
              >
                {link.name}
              </button>
            ))}
<div className="flex gap-3 pt-2">
              <a href="https://github.com/Hrick-08/stress-ml" target="_blank" className="p-2 rounded-lg bg-white/5">
                <Github className="w-5 h-5" />
              </a>
            </div>
          </motion.div>
        )}
      </div>
    </motion.nav>
  )
}