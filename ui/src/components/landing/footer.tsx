import { motion } from "framer-motion"
import { Brain, Github, Linkedin, ExternalLink } from "lucide-react"

const footerLinks = {
  Product: [
    { name: "Features", href: "#features" },
    { name: "Model Comparison", href: "#app" },
    { name: "Live Predictor", href: "#app" },
  ],
  Team: [
    { name: "About Us", href: "#team" },
    { name: "Contact", href: "#" },
  ],
  Social: [
    { name: "GitHub", href: "#", icon: Github },
    { name: "LinkedIn", href: "#", icon: Linkedin },
  ],
}

export function Footer() {
  return (
    <footer className="py-16 border-t border-white/10">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="md:col-span-1">
            <motion.div
              className="flex items-center gap-2 mb-4"
              whileHover={{ scale: 1.02 }}
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold gradient-text">
                StressPredict
              </span>
            </motion.div>
            <p className="text-sm text-muted-foreground">
              AI-powered student stress and performance prediction platform.
            </p>
          </div>

          {/* Product Links */}
          <div>
            <h4 className="font-semibold mb-4 text-white">Product</h4>
            <ul className="space-y-3">
              {footerLinks.Product.map((link) => (
                <li key={link.name}>
                  <a
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Team Links */}
          <div>
            <h4 className="font-semibold mb-4 text-white">Team</h4>
            <ul className="space-y-3">
              {footerLinks.Team.map((link) => (
                <li key={link.name}>
                  <a
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Social Links */}
          <div>
            <h4 className="font-semibold mb-4 text-white">Connect</h4>
            <div className="flex gap-3">
              {footerLinks.Social.map((link) => {
                const Icon = link.icon || ExternalLink
                return (
                  <motion.a
                    key={link.name}
                    href={link.href}
                    whileHover={{ scale: 1.1, y: -2 }}
                    className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                  >
                    <Icon className="w-5 h-5" />
                  </motion.a>
                )
              })}
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-muted-foreground">
            © 2024 StressPredict. Built with React + ML.
          </p>
          <p className="text-sm text-muted-foreground">
            Group G-10 | BE CSE AI&ML 4th Sem | Batch 2024
          </p>
        </div>
      </div>
    </footer>
  )
}