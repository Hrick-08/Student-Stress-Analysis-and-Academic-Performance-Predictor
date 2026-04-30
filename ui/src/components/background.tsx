import { useMemo } from "react"

interface StarBackgroundProps {
  starCount?: number
}

function StarBackground({ starCount = 150 }: StarBackgroundProps) {
  const stars = useMemo(() => {
    const generated = []
    for (let i = 0; i < starCount; i++) {
      const size = Math.random()
      let type = "small"
      let opacity = 0.3

      if (size > 0.93) {
        type = "large"
        opacity = 1
      } else if (size > 0.75) {
        type = "medium"
        opacity = 0.7
      } else {
        type = "small"
        opacity = 0.3 + Math.random() * 0.4
      }

      generated.push({
        id: i,
        left: Math.random() * 100,
        top: Math.random() * 100,
        type,
        opacity,
        delay: Math.random() * 5,
        duration: 2 + Math.random() * 3,
      })
    }
    return generated
  }, [starCount])

  return (
    <div className="star-field-layer" style={{ position: "absolute", inset: 0 }}>
      {stars.map((star) => (
        <div
          key={star.id}
          className={`absolute rounded-full ${star.type === "large" ? "w-2 h-2" : star.type === "medium" ? "w-1.5 h-1.5" : "w-1 h-1"}`}
          style={{
            left: `${star.left}%`,
            top: `${star.top}%`,
            backgroundColor: star.type === "large" ? "#ffffff" : "rgba(255,255,255,0.8)",
            opacity: star.opacity,
            boxShadow: star.type === "large" 
              ? `0 0 ${star.type === "large" ? 8 : 4}px rgba(255,255,255,0.8)` 
              : "none",
            animation: `twinkle ${star.duration}s ease-in-out ${star.delay}s infinite`,
          }}
        />
      ))}
    </div>
  )
}

interface NebulaBackgroundProps {
  intensity?: "subtle" | "medium" | "strong"
}

function NebulaBackground({ intensity = "medium" }: NebulaBackgroundProps) {
  const colors = {
    subtle: { cyan: 0.1, purple: 0.08, pink: 0.05 },
    medium: { cyan: 0.2, purple: 0.15, pink: 0.1 },
    strong: { cyan: 0.35, purple: 0.25, pink: 0.15 },
  }
  const c = colors[intensity]

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          width: "50vw",
          height: "50vw",
          borderRadius: "50%",
          background: `radial-gradient(circle, rgba(34, 211, 238, ${c.cyan}) 0%, transparent 70%)`,
          filter: "blur(60px)",
          top: "-10%",
          left: "-10%",
          animation: "nebulaFloat 20s ease-in-out infinite",
        }}
      />
      <div
        style={{
          position: "absolute",
          width: "40vw",
          height: "40vw",
          borderRadius: "50%",
          background: `radial-gradient(circle, rgba(168, 85, 247, ${c.purple}) 0%, transparent 70%)`,
          filter: "blur(60px)",
          bottom: "-10%",
          right: "-10%",
          animation: "nebulaFloat 25s ease-in-out infinite reverse",
        }}
      />
      <div
        style={{
          position: "absolute",
          width: "30vw",
          height: "30vw",
          borderRadius: "50%",
          background: `radial-gradient(circle, rgba(236, 72, 153, ${c.pink}) 0%, transparent 70%)`,
          filter: "blur(50px)",
          top: "40%",
          right: "20%",
          animation: "nebulaFloat 30s ease-in-out infinite",
        }}
      />
    </div>
  )
}

function GridOverlay() {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        backgroundImage: `
          linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px)
        `,
        backgroundSize: "50px 50px",
        animation: "gridPulse 8s ease-in-out infinite",
      }}
    />
  )
}

export { StarBackground, NebulaBackground, GridOverlay }