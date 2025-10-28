import React, { useState, useEffect, useRef } from 'react';
import { BarChart3, Layers, CheckCircle2, XCircle, TrendingUp, Database, Users, Globe } from 'lucide-react';

const Framework3DVisual = () => {
  const [activeLayer, setActiveLayer] = useState(null);
  const [rotationAngle, setRotationAngle] = useState(0);
  const [showParticles, setShowParticles] = useState(true);
  const canvasRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setRotationAngle(prev => (prev + 0.5) % 360);
    }, 50);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!canvasRef.current || !showParticles) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    const particles = Array.from({ length: 50 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 2 + 1
    }));
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
      
      particles.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;
        
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      });
      
      requestAnimationFrame(animate);
    };
    
    animate();
  }, [showParticles]);

  const layers = [
    { 
      id: 1, 
      name: "Discovery", 
      icon: Database,
      color: "#667eea",
      metrics: ["47 patterns", "50K permutations", "Range 1-2000"],
      depth: 0
    },
    { 
      id: 2, 
      name: "Statistical", 
      icon: BarChart3,
      color: "#764ba2",
      metrics: ["BF: 12.4-21.2", "FDR correction", "CI 95%"],
      depth: 1
    },
    { 
      id: 3, 
      name: "Diachronic", 
      icon: TrendingUp,
      color: "#f093fb",
      metrics: ["2000+ years", "100% stability", "3 manuscripts"],
      depth: 2
    },
    { 
      id: 4, 
      name: "Cross-cultural", 
      icon: Globe,
      color: "#4facfe",
      metrics: ["4 corpora", "AS: 1.00", "Distinctive"],
      depth: 3
    },
    { 
      id: 5, 
      name: "Expert Review", 
      icon: Users,
      color: "#43e97b",
      metrics: ["13 experts", "Consensus 7.5-8.4", "Validation"],
      depth: 4
    }
  ];

  const patterns = [
    { name: "תולדות", value: 12.4, status: "valid", color: "#10b981" },
    { name: "Sum 1260", value: 18.7, status: "valid", color: "#10b981" },
    { name: "Sum 1335", value: 14.9, status: "valid", color: "#10b981" },
    { name: "Sum 490", value: 21.2, status: "valid", color: "#10b981" },
    { name: "Rejected", value: 43, status: "rejected", color: "#ef4444" }
  ];

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8 relative overflow-hidden">
      <canvas 
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ opacity: showParticles ? 1 : 0 }}
      />
      
      <div className="relative z-10 max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4 tracking-tight">
            🔬 Multi-Level Validation Architecture
          </h1>
          <p className="text-xl text-purple-200">
            From Data Chaos to Validated Certainty
          </p>
          <button 
            onClick={() => setShowParticles(!showParticles)}
            className="mt-4 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-all"
          >
            {showParticles ? "Hide" : "Show"} particles
          </button>
        </div>

        {/* Funnel principal 3D */}
        <div className="mb-16 perspective-1000">
          <div 
            className="relative mx-auto transition-transform duration-500"
            style={{ 
              transform: `rotateY(${rotationAngle * 0.3}deg) rotateX(${Math.sin(rotationAngle * 0.02) * 5}deg)`,
              transformStyle: 'preserve-3d'
            }}
          >
            {layers.map((layer, idx) => {
              const Icon = layer.icon;
              const width = 100 - (idx * 15);
              const isActive = activeLayer === layer.id;
              
              return (
                <div
                  key={layer.id}
                  className="relative mx-auto mb-6 cursor-pointer transition-all duration-300 group"
                  style={{ 
                    width: `${width}%`,
                    transform: `translateZ(${isActive ? 50 : layer.depth * 10}px)`,
                  }}
                  onMouseEnter={() => setActiveLayer(layer.id)}
                  onMouseLeave={() => setActiveLayer(null)}
                >
                  <div 
                    className="relative p-6 rounded-2xl backdrop-blur-lg border-2 transition-all duration-300"
                    style={{
                      background: isActive 
                        ? `linear-gradient(135deg, ${layer.color}dd, ${layer.color}99)`
                        : `linear-gradient(135deg, ${layer.color}88, ${layer.color}55)`,
                      borderColor: isActive ? '#ffffff' : `${layer.color}66`,
                      boxShadow: isActive 
                        ? `0 20px 60px ${layer.color}66, 0 0 40px ${layer.color}44 inset`
                        : `0 10px 30px ${layer.color}33`
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className={`p-3 bg-white/20 rounded-xl transition-transform ${isActive ? 'scale-110' : ''}`}>
                          <Icon className="w-8 h-8 text-white" />
                        </div>
                        <div>
                          <h3 className="text-2xl font-bold text-white mb-1">
                            Layer {layer.id}: {layer.name}
                          </h3>
                          <div className="flex gap-3 flex-wrap">
                            {layer.metrics.map((metric, i) => (
                              <span 
                                key={i}
                                className="px-3 py-1 bg-white/20 rounded-full text-sm text-white font-medium"
                              >
                                {metric}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                      <div className={`text-4xl font-bold text-white transition-opacity ${isActive ? 'opacity-100' : 'opacity-50'}`}>
                        {layer.id}
                      </div>
                    </div>
                  </div>
                  
                  {/* Connecteur */}
                  {idx < layers.length - 1 && (
                    <div className="absolute left-1/2 -bottom-6 transform -translate-x-1/2 z-0">
                      <div className="w-1 h-6 bg-gradient-to-b from-purple-400 to-transparent"></div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Résultats - Visualisation circulaire */}
        <div className="bg-white/5 backdrop-blur-lg rounded-3xl p-8 border border-white/10">
          <h2 className="text-3xl font-bold text-white mb-8 text-center flex items-center justify-center gap-3">
            <Layers className="w-8 h-8" />
            Filtering Outcomes
          </h2>
          
          <div className="grid grid-cols-5 gap-6 mb-8">
            {patterns.map((pattern, idx) => (
              <div
                key={idx}
                className="relative group"
              >
                <div 
                  className="relative aspect-square rounded-2xl flex flex-col items-center justify-center transition-all duration-300 hover:scale-110"
                  style={{
                    background: `conic-gradient(${pattern.color} ${(pattern.value / 50) * 360}deg, rgba(255,255,255,0.1) 0deg)`,
                    boxShadow: `0 10px 40px ${pattern.color}66`
                  }}
                >
                  <div className="absolute inset-2 bg-slate-900 rounded-xl flex flex-col items-center justify-center p-4">
                    {pattern.status === 'valid' ? (
                      <CheckCircle2 className="w-8 h-8 text-green-400 mb-2" />
                    ) : (
                      <XCircle className="w-8 h-8 text-red-400 mb-2" />
                    )}
                    <div className="text-2xl font-bold text-white mb-1">
                      {pattern.value}
                    </div>
                    <div className="text-xs text-center text-white/70 font-medium">
                      {pattern.name}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Final statistics */}
          <div className="grid grid-cols-4 gap-6">
            <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 p-6 rounded-xl border border-green-500/30">
              <div className="text-4xl font-bold text-green-400 mb-2">4</div>
              <div className="text-sm text-green-200">Validated Patterns</div>
            </div>
            <div className="bg-gradient-to-br from-red-500/20 to-red-600/20 p-6 rounded-xl border border-red-500/30">
              <div className="text-4xl font-bold text-red-400 mb-2">43</div>
              <div className="text-sm text-red-200">Rejected Patterns</div>
            </div>
            <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/20 p-6 rounded-xl border border-purple-500/30">
              <div className="text-4xl font-bold text-purple-400 mb-2">91%</div>
              <div className="text-sm text-purple-200">Discrimination Rate</div>
            </div>
            <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 p-6 rounded-xl border border-blue-500/30">
              <div className="text-4xl font-bold text-blue-400 mb-2">87%</div>
              <div className="text-sm text-blue-200">Replication Success</div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-purple-300 text-sm">
          <p>© 2025 Benseddik Ahmed | ORCID: 0009-0005-6308-8171 | DOI: 10.5281/zenodo.17443361</p>
        </div>
      </div>
    </div>
  );
};

export default Framework3DVisual;