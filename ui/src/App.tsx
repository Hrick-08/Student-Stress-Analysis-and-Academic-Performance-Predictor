import { useState, useEffect, useRef } from "react"
import { motion } from "framer-motion"
import { Tabs, TabsList, TabsTrigger, AnimatedTabsContent } from "@/components/ui/tabs"
import { CardContent, CardDescription, CardHeader, CardTitle, MotionCard } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { SkeletonCard, SkeletonChart } from "@/components/ui/skeleton"
import { StarBackground, NebulaBackground, GridOverlay } from "@/components/background"
import {
  Navbar,
  HeroSection,
  StatsSection,
  FeaturesSection,
  TeamSection,
  Footer,
} from "./components/landing"
import { Slider } from "@/components/ui/slider"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"
import { Brain, Activity, Heart, GraduationCap, TrendingUp, AlertCircle, CheckCircle, Loader2 } from "lucide-react"
import { StudentInput, PredictionResult, ModelInfo, DEFAULT_STUDENT_INPUT, MODELS_LIST } from "./types/api"
import { fetchModels, predictWithModel, predictAllModels } from "./services/api"

function App() {
  const [activeTab, setActiveTab] = useState("models")
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState("AdaBoost")
  const [formData, setFormData] = useState<StudentInput>(DEFAULT_STUDENT_INPUT)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [allPredictions, setAllPredictions] = useState<Record<string, PredictionResult | null>>({})
  const [predicting, setPredicting] = useState(false)
  const predictionRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadModels()
  }, [])

  async function loadModels() {
    try {
      setLoading(true)
      const response = await fetchModels()
      setModels(response.models)
      setError(null)
    } catch (err) {
      setError("Failed to connect to API. Make sure the backend is running on port 8001.")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  function updateFormField(field: keyof StudentInput, value: number) {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  async function handlePredict() {
    try {
      setPredicting(true)
      const result = await predictWithModel(formData, selectedModel)
      setPrediction(result)

      const allResult = await predictAllModels(formData)
      setAllPredictions(allResult.predictions)
      
      predictionRef.current?.scrollIntoView({ behavior: "smooth", block: "center" })
    } catch (err) {
      console.error("Prediction error:", err)
    } finally {
      setPredicting(false)
    }
  }

  const r2Colors = ["#22d3ee", "#3b82f6", "#8b5cf6", "#ec4899", "#f97316", "#eab308"]
  const sortedModels = [...models].sort((a, b) => b.test_r2 - a.test_r2)

  const performanceColors: Record<string, string> = {
    "At Risk": "bg-red-500/20 text-red-400 border-red-500/30",
    "Average": "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    "Performing": "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  }

  const scoreToPercentage = (score: number) => Math.round((score / 5) * 100)

  const getInterpretation = (band: string, data: StudentInput) => {
    if (band === "At Risk") {
      const concerns = []
      if (data.stress_level === 2) concerns.push("high stress")
      if (data.anxiety_level > 15) concerns.push("high anxiety")
      if (data.sleep_quality === 0) concerns.push("poor sleep")
      if (data.self_esteem < 10) concerns.push("low self-esteem")
      return concerns.length > 0 ? `Key concerns: ${concerns.join(", ")}` : "Multiple lifestyle factors need attention"
    } else if (band === "Average") {
      const improvements = []
      if (data.sleep_quality < 2) improvements.push("sleep quality")
      if (data.social_support < 3) improvements.push("social support")
      return improvements.length > 0 ? `Focus on: ${improvements.join(", ")}` : "Small lifestyle changes can help"
    }
    return "Great balance! Keep up current practices"
  }

  return (
    <div className="min-h-screen">
      {/* Background */}
      <div style={{ position: "fixed", inset: 0, overflow: "hidden", background: "radial-gradient(ellipse at bottom, #1a1a2e 0%, #0a0a15 100%)", zIndex: 0 }}>
        <StarBackground starCount={250} />
        <NebulaBackground intensity="medium" />
        <GridOverlay />
        <div 
          style={{
            position: "absolute",
            width: "50vw",
            height: "50vw",
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(34, 211, 238, 0.15) 0%, transparent 70%)",
            filter: "blur(80px)",
            top: "-15%",
            right: "-10%",
            animation: "ambientGlow 10s ease-in-out infinite",
          }} 
        />
        <div 
          style={{
            position: "absolute",
            width: "40vw",
            height: "40vw",
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(168, 85, 247, 0.12) 0%, transparent 70%)",
            filter: "blur(80px)",
            bottom: "-10%",
            left: "-10%",
            animation: "ambientGlow 15s ease-in-out infinite reverse",
          }} 
        />
      </div>

      {/* Content */}
      <div style={{ position: "relative", zIndex: 1 }}>
        {/* Navigation */}
        <Navbar />

        {/* Hero Section */}
        <HeroSection onGetStarted={() => {
          document.getElementById('app')?.scrollIntoView({ behavior: 'smooth' })
        }} />

        {/* Stats Section */}
        <StatsSection />

        {/* Features Section */}
        <FeaturesSection />

        {/* Team Section */}
        <TeamSection />

        {/* App Section - Model Comparison Tab */}
        <section id="app" className="py-20">
          <div className="container mx-auto px-4">
            {/* Section Header */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center mb-10"
            >
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                <span className="gradient-text">Start Predicting</span>
                <span className="text-white"> Today</span>
              </h2>
              <p className="text-muted-foreground max-w-xl mx-auto">
                Use our AI-powered prediction tool to analyze student stress and performance.
              </p>
            </motion.div>
            </div>

            {/* Tabs for App */}
            {error && (
          <MotionCard className="mb-8 border-red-500/30 bg-red-500/10" hover3d>
            <CardContent className="flex items-center gap-3 p-4">
              <AlertCircle className="w-5 h-5 text-red-400" />
              <span className="text-red-300">{error}</span>
              <Button variant="outline" size="sm" onClick={loadModels} className="ml-auto">
                Retry
              </Button>
            </CardContent>
          </MotionCard>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="max-w-6xl mx-auto">
          <TabsList className="grid w-full max-w-md mx-auto mb-8 grid-cols-2">
            <TabsTrigger value="models" className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Model Comparison
            </TabsTrigger>
            <TabsTrigger value="predictor" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Live Predictor
            </TabsTrigger>
          </TabsList>

          <AnimatedTabsContent value="models">
            {loading ? (
              <div className="grid gap-6">
                <SkeletonChart />
                <SkeletonChart />
                <SkeletonCard />
              </div>
            ) : (
              <div className="grid gap-6">
                <MotionCard className="glass-card" hover3d>
                  <CardHeader>
                    <CardTitle className="text-2xl flex items-center gap-2">
                      <Activity className="w-6 h-6 text-cyan-400" />
                      Model Performance Comparison
                    </CardTitle>
                    <CardDescription>
                      R² Score measures how well models predict (higher is better)
                    </CardDescription>
                  </CardHeader>
<CardContent>
                    <div className="h-[400px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart 
                          data={sortedModels} 
                          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                          <XAxis dataKey="name" stroke="#94a3b8" tick={{ fill: "#94a3b8" }} />
                          <YAxis stroke="#94a3b8" tick={{ fill: "#94a3b8" }} domain={[0, 1]} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "rgba(15, 23, 42, 0.9)",
                              border: "1px solid rgba(255,255,255,0.1)",
                              borderRadius: "8px",
                            }}
                          />
                          <Bar 
                            dataKey="test_r2" 
                            name="Test R²" 
                            radius={[8, 8, 0, 0]}
                            isAnimationActive={true}
                            animationDuration={1200}
                          >
                            {sortedModels.map((_, index) => (
                              <Cell key={index} fill={r2Colors[index % r2Colors.length]} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </MotionCard>

                <MotionCard className="glass-card" hover3d>
                  <CardHeader>
                    <CardTitle className="text-2xl">Root Mean Square Error (RMSE)</CardTitle>
                    <CardDescription>
                      RMSE measures prediction error (lower is better)
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[300px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart 
                          data={sortedModels} 
                          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                          <XAxis dataKey="name" stroke="#94a3b8" tick={{ fill: "#94a3b8" }} />
                          <YAxis stroke="#94a3b8" tick={{ fill: "#94a3b8" }} domain={[0.5, 1]} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "rgba(15, 23, 42, 0.9)",
                              border: "1px solid rgba(255,255,255,0.1)",
                              borderRadius: "8px",
                            }}
                          />
                          <Bar dataKey="rmse" name="RMSE" radius={[8, 8, 0, 0]} fill="#f97316" isAnimationActive={true} animationDuration={1500}></Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
</CardContent>
                </MotionCard>

                <MotionCard className="glass-card" hover3d>
                  <CardHeader>
                    <CardTitle className="text-xl">Detailed Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b border-white/10">
                            <th className="text-left py-3 px-4 text-muted-foreground">Model</th>
                            <th className="text-right py-3 px-4 text-muted-foreground">Test R²</th>
                            <th className="text-right py-3 px-4 text-muted-foreground">RMSE</th>
                            <th className="text-center py-3 px-4 text-muted-foreground">Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {sortedModels.map((model) => (
                            <tr key={model.name} className="border-b border-white/5 hover:bg-white/5">
                              <td className="py-3 px-4 font-medium">{model.name}</td>
                              <td className="text-right py-3 px-4 font-mono">{model.test_r2.toFixed(4)}</td>
                              <td className="text-right py-3 px-4 font-mono">{model.rmse.toFixed(4)}</td>
                              <td className="text-center py-3 px-4">
                                {model.name === "AdaBoost" && (
                                  <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-cyan-500/20 text-cyan-400 text-sm">
                                    <CheckCircle className="w-3 h-3" />
                                    Recommended
                                  </span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </MotionCard>
              </div>
            )}
          </AnimatedTabsContent>

          <AnimatedTabsContent value="predictor">
            <div className="grid lg:grid-cols-2 gap-6">
              <MotionCard className="glass-card" hover3d>
                <CardHeader>
                  <CardTitle className="text-2xl flex items-center gap-2">
                    <GraduationCap className="w-6 h-6 text-purple-400" />
                    Student Information
                  </CardTitle>
                  <CardDescription>
                    Adjust sliders to input student data
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="flex items-center gap-4 p-4 rounded-lg bg-white/5 border border-white/10">
                    <Brain className="w-5 h-5 text-cyan-400" />
                    <div className="flex-1">
                      <label className="text-sm font-medium">Model Selection</label>
                    </div>
                    <Select value={selectedModel} onValueChange={setSelectedModel}>
                      <SelectTrigger className="w-[200px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {MODELS_LIST.map((model) => (
                          <SelectItem key={model.name} value={model.name}>
                            {model.name} {model.recommended && "★"}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <FormSection title="Mental Health & Stress" icon={Brain} color="text-rose-400">
                    <FormSlider
                      label="Anxiety Level"
                      field="anxiety_level"
                      value={formData.anxiety_level}
                      min={0}
                      max={21}
                      onChange={updateFormField}
                    />
                    <FormSlider
                      label="Depression"
                      field="depression"
                      value={formData.depression}
                      min={0}
                      max={27}
                      onChange={updateFormField}
                    />
                    <FormSlider
                      label="Self Esteem"
                      field="self_esteem"
                      value={formData.self_esteem}
                      min={0}
                      max={30}
                      onChange={updateFormField}
                    />
                  </FormSection>

                  <FormSection title="Environmental & Social" icon={Heart} color="text-emerald-400">
                    <FormSlider
                      label="Safety"
                      field="safety"
                      value={formData.safety}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                    <FormSlider
                      label="Basic Needs"
                      field="basic_needs"
                      value={formData.basic_needs}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                    <FormSlider
                      label="Living Conditions"
                      field="living_conditions"
                      value={formData.living_conditions}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                    <FormSlider
                      label="Social Support"
                      field="social_support"
                      value={formData.social_support}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                    <FormSlider
                      label="Teacher-Student Relationship"
                      field="teacher_student_relationship"
                      value={formData.teacher_student_relationship}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                  </FormSection>

                  <FormSection title="Physical Health" icon={Activity} color="text-orange-400">
                    <FormSlider
                      label="Headache Frequency"
                      field="headache"
                      value={formData.headache}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                    <FormSlider
                      label="Breathing Problems"
                      field="breathing_problem"
                      value={formData.breathing_problem}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                    <SelectField
                      label="Sleep Quality"
                      value={formData.sleep_quality.toString()}
                      onChange={(v) => updateFormField("sleep_quality", parseInt(v))}
                      options={[
                        { value: "0", label: "Poor" },
                        { value: "1", label: "Average" },
                        { value: "2", label: "Good" },
                      ]}
                    />
                    <SelectField
                      label="Blood Pressure"
                      value={formData.blood_pressure.toString()}
                      onChange={(v) => updateFormField("blood_pressure", parseInt(v))}
                      options={[
                        { value: "1", label: "Low" },
                        { value: "2", label: "Normal" },
                        { value: "3", label: "High" },
                      ]}
                    />
                  </FormSection>

                  <FormSection title="Academic & Lifestyle" icon={GraduationCap} color="text-violet-400">
                    <FormSlider
                      label="Peer Pressure"
                      field="peer_pressure"
                      value={formData.peer_pressure}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                    <FormSlider
                      label="Study Load"
                      field="study_load"
                      value={formData.study_load}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                    <FormSlider
                      label="Future Career Concerns"
                      field="future_career_concerns"
                      value={formData.future_career_concerns}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                    <FormSlider
                      label="Extracurricular Activities"
                      field="extracurricular_activities"
                      value={formData.extracurricular_activities}
                      min={0}
                      max={5}
                      onChange={updateFormField}
                    />
                    <SelectField
                      label="Stress Level"
                      value={formData.stress_level.toString()}
                      onChange={(v) => updateFormField("stress_level", parseInt(v))}
                      options={[
                        { value: "0", label: "Low" },
                        { value: "1", label: "Medium" },
                        { value: "2", label: "High" },
                      ]}
                    />
                  </FormSection>

                  <Button
                    variant="glow"
                    onClick={handlePredict}
                    disabled={predicting}
                    className="w-full py-6 text-lg"
                  >
                    {predicting ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Predicting...
                      </>
                    ) : (
                      "Predict Performance"
                    )}
</Button>
                </CardContent>
              </MotionCard>

              <div className="space-y-6">
                {prediction && (
                  <MotionCard ref={predictionRef} className="glass-card glow-border glow-border-cyan glow-border-animated" hover3d>
                    <CardHeader>
                      <CardTitle className="text-2xl flex items-center gap-2">
                        <CheckCircle className="w-6 h-6 text-cyan-400" />
                        Prediction Result
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-center mb-6">
                        <div className="text-7xl font-bold gradient-text mb-2">
                          {scoreToPercentage(prediction.predicted_score)}%
                        </div>
                        <div className="text-muted-foreground">Performance Score</div>
                      </div>

                      <div className="flex justify-center mb-6">
                        <span className={`px-4 py-2 rounded-full border ${performanceColors[prediction.performance_band]}`}>
                          {prediction.performance_band}
                        </span>
                      </div>

                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between py-2 border-b border-white/10">
                          <span className="text-muted-foreground">Raw Prediction</span>
                          <span>{prediction.predicted_score.toFixed(2)} / 5.0</span>
                        </div>
                        <div className="flex justify-between py-2 border-b border-white/10">
                          <span className="text-muted-foreground">Model Used</span>
                          <span>{prediction.model_name}</span>
                        </div>
                        <div className="flex justify-between py-2 border-b border-white/10">
                          <span className="text-muted-foreground">Model R²</span>
                          <span>{prediction.model_r2.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between py-2">
                          <span className="text-muted-foreground">Model RMSE</span>
                          <span>{prediction.model_rmse.toFixed(4)}</span>
                        </div>
                      </div>

                      <div className="mt-6 p-4 rounded-lg bg-white/5 border border-white/10">
                        <p className="text-sm text-muted-foreground mb-2">Interpretation</p>
                        <p>{getInterpretation(prediction.performance_band, formData)}</p>
                      </div>
                    </CardContent>
                  </MotionCard>
                )}

                {Object.keys(allPredictions).length > 0 && (
                  <MotionCard className="glass-card" hover3d>
                    <CardHeader>
                      <CardTitle className="text-xl">All Models Comparison</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-white/10">
                              <th className="text-left py-2 px-3 text-muted-foreground">Model</th>
                              <th className="text-right py-2 px-3 text-muted-foreground">Score</th>
                              <th className="text-center py-2 px-3 text-muted-foreground">Band</th>
                              <th className="text-right py-2 px-3 text-muted-foreground">R²</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(allPredictions)
                              .filter(([_, pred]) => pred !== null)
                              .map(([name, pred]) => {
                                const sorted = Object.entries(allPredictions)
                                  .filter(([_, p]) => p !== null)
                                  .sort((a, b) => {
                                    const order = { "At Risk": 0, "Average": 1, "Performing": 2 }
                                    return (order[a[1]!.performance_band] || 999) - (order[b[1]!.performance_band] || 999)
                                  })
                                const rank = sorted.findIndex(([n]) => n === name)
                                return (
                                  <tr key={name} className="border-b border-white/5">
                                    <td className="py-2 px-3 font-medium">
                                      {name}
                                      {rank === 0 && <span className="ml-2 text-cyan-400">★</span>}
                                    </td>
                                    <td className="text-right py-2 px-3 font-mono">
                                      {scoreToPercentage(pred!.predicted_score)}%
                                    </td>
                                    <td className="text-center py-2 px-3">
                                      <span className={`px-2 py-1 rounded text-xs ${performanceColors[pred!.performance_band]}`}>
                                        {pred!.performance_band}
                                      </span>
                                    </td>
                                    <td className="text-right py-2 px-3 font-mono">
                                      {pred!.model_r2.toFixed(4)}
                                    </td>
                                  </tr>
                                )
                              })}
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </MotionCard>
                )}

                {!prediction && (
                  <MotionCard className="glass-card" hover3d>
                    <CardContent className="py-20 text-center">
                      <Brain className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
                      <p className="text-muted-foreground">Fill in the form and click predict to see results</p>
                    </CardContent>
                  </MotionCard>
                )}
              </div>
            </div>
          </AnimatedTabsContent>
          </Tabs>
        </section>

        {/* Footer */}
        <Footer />
      </div>
    </div>
  )
}

interface FormSectionProps {
  title: string
  icon: React.ElementType
  color: string
  children: React.ReactNode
}

function FormSection({ title, icon: Icon, color, children }: FormSectionProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Icon className={`w-4 h-4 ${color}`} />
        {title}
      </div>
      <div className="space-y-4 pl-6 border-l-2 border-white/10">{children}</div>
    </div>
  )
}

interface FormSliderProps {
  label: string
  field: keyof StudentInput
  value: number
  min: number
  max: number
  onChange: (field: keyof StudentInput, value: number) => void
}

function FormSlider({ label, field, value, min, max, onChange }: FormSliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-medium">{value}</span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={1}
        onValueChange={(val) => onChange(field, val[0])}
        className="py-2"
      />
    </div>
  )
}

interface SelectFieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
}

function SelectField({ label, value, onChange, options }: SelectFieldProps) {
  return (
    <div className="space-y-2">
      <div className="text-sm text-muted-foreground">{label}</div>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-9">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

export default App