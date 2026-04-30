import { motion } from "framer-motion"
import { Github, Linkedin, Mail } from "lucide-react"

interface TeamMember {
  name: string
  role: string
  pixelColor: string
  accentColor: string
  github: string
  linkedin: string
  email: string
}

const teamMembers: TeamMember[] = [
  { 
    name: "Hritabrata Das", 
    role: "ML Engineer", 
    pixelColor: "#22d3ee", 
    accentColor: "#0891b2",
    github: "https://github.com/Hrick-08",
    linkedin: "https://www.linkedin.com/in/hritabrata-das/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BxcKr1PTUTL2qyjFUdicavA%3D%3D",
    email: "mailto:hritabratadas8@gmail.com"
  },
  { 
    name: "Abhinav Dadwal", 
    role: "ML Engineer", 
    pixelColor: "#a855f7", 
    accentColor: "#7c3aed",
    github: "https://github.com/Abh1nav7",
    linkedin: "https://www.linkedin.com/in/abhinav-dadwal-3b280a324/",
    email: "mailto:abhinavdadwal112@gmail.com"
  },
  { 
    name: "Harshit", 
    role: "ML Engineer", 
    pixelColor: "#ec4899", 
    accentColor: "#db2777",
    github: "https://github.com/harshit459",
    linkedin: "https://www.linkedin.com/in/harshit-90a0b1324/",
    email: "mailto:harshit12063@gmail.com"
  },
]

function PixelAvatar({ color, accentColor }: { color: string; accentColor: string }) {
  const pixels = [
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 2, 2, 2, 2, 1, 0],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 3, 4, 3, 3, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [0, 1, 2, 2, 2, 2, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
  ]

  const colors = ["transparent", color, accentColor, "#fcd34d"]

  return (
    <div 
      className="w-20 h-20 rounded-xl overflow-hidden"
      style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(8, 1fr)",
        gap: "1px",
        background: "rgba(255,255,255,0.1)"
      }}
    >
      {pixels.map((row, y) =>
        row.map((pixel, x) => (
          <div
            key={`${x}-${y}`}
            style={{
              background: colors[pixel],
            }}
          />
        ))
      )}
    </div>
  )
}

export function TeamSection() {
  return (
    <section id="team" className="py-20">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="gradient-text">Meet Our Team</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Group G-10 | BE CSE AI&ML 4th Sem | Batch 2024
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          {teamMembers.map((member, i) => (
            <motion.div
              key={member.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              whileHover={{ y: -5 }}
              className="glass-card rounded-2xl p-6 text-center group"
            >
              <PixelAvatar 
                color={member.pixelColor} 
                accentColor={member.accentColor} 
              />
              
              <h3 className="text-lg font-semibold mt-4 mb-1 text-white group-hover:gradient-text transition-all">
                {member.name}
              </h3>
              
              <p className="text-sm text-muted-foreground mb-4">
                {member.role}
              </p>
              
              <div className="flex justify-center gap-3">
                <motion.a
                  href={member.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.1, y: -2 }}
                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <Github className="w-4 h-4" />
                </motion.a>
                <motion.a
                  href={member.linkedin}
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.1, y: -2 }}
                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <Linkedin className="w-4 h-4 text-blue-400" />
                </motion.a>
                <motion.a
                  href={member.email}
                  whileHover={{ scale: 1.1, y: -2 }}
                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <Mail className="w-4 h-4" />
                </motion.a>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}