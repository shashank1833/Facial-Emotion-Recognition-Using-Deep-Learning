import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { 
  Upload, 
  Camera, 
  Video, 
  Activity, 
  History, 
  BarChart3, 
  Settings, 
  Brain, 
  Zap, 
  ShieldCheck, 
  RefreshCw,
  X
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:5000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [activeTab, setActiveTab] = useState('analyze');
  const [history, setHistory] = useState([]);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [analysisSteps, setAnalysisSteps] = useState([]);
  const [isLive, setIsLive] = useState(false);
  const [isLiveAnalyzing, setIsLiveAnalyzing] = useState(false);
  const [liveResult, setLiveResult] = useState(null);
  const [liveError, setLiveError] = useState(null);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const liveVideoRef = useRef(null);
  const canvasRef = useRef(null);
  const liveCanvasRef = useRef(null);
  const liveIntervalRef = useRef(null);

  // Check backend health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/health`);
        if (response.data.status === 'healthy') {
          setBackendStatus('ready');
        }
      } catch (err) {
        setBackendStatus('error');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const clearAllData = () => {
    setHistory([]);
    setAnalysisResult(null);
    setAnalysisSteps([]);
    setLiveResult(null);
    localStorage.removeItem('emotion_history');
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setFileType(file.type.startsWith('video') ? 'video' : 'image');
    setAnalysisResult(null);
    setAnalysisSteps([]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file);
    }
  };

  const clearUpload = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setFileType(null);
    setAnalysisResult(null);
    setAnalysisSteps([]);
  };

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return null;
    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg');
  };

  const startLiveDetection = async () => {
    try {
      setLiveError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720, facingMode: 'user' } 
      });
      
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream;
        setIsLive(true);
      }
    } catch (err) {
      console.error('Failed to start camera:', err);
      setLiveError('Camera access denied or not available.');
    }
  };

  const manualAnalyzeLive = async () => {
    if (!liveVideoRef.current || !liveCanvasRef.current || backendStatus !== 'ready' || isLiveAnalyzing) return;
    
    setIsLiveAnalyzing(true);
    const canvas = liveCanvasRef.current;
    const video = liveVideoRef.current;
    
    // Ensure video is playing and has dimensions
    if (video.videoWidth === 0) {
      setIsLiveAnalyzing(false);
      return;
    }
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg').split(',')[1];
    
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        image: imageData
      });
      setLiveResult(response.data);
      
      // Add to history
      const newRecord = {
        id: Date.now(),
        emotion: response.data.emotion,
        type: 'live',
        time: new Date().toLocaleTimeString()
      };
      setHistory(prev => [newRecord, ...prev].slice(0, 10));
    } catch (err) {
      console.error('Live analysis failed:', err);
      alert('Analysis failed. Please ensure backend is running.');
    } finally {
      setIsLiveAnalyzing(false);
    }
  };

  const stopLiveDetection = () => {
    setIsLive(false);
    if (liveIntervalRef.current) {
      clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
    }
    if (liveVideoRef.current && liveVideoRef.current.srcObject) {
      const tracks = liveVideoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      liveVideoRef.current.srcObject = null;
    }
    setLiveResult(null);
  };

  useEffect(() => {
    // Stop live detection when switching tabs
    if (activeTab !== 'live' && isLive) {
      stopLiveDetection();
    }
  }, [activeTab]);

  const runAnalysis = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    setAnalysisSteps(['detect']);
    
    let imageData = '';
    
    if (fileType === 'video') {
      imageData = captureFrame();
    } else {
      // Convert image to base64
      imageData = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(selectedFile);
      });
    }

    // Simulate pipeline steps
    setTimeout(() => setAnalysisSteps(prev => [...prev, 'landmarks']), 800);
    setTimeout(() => setAnalysisSteps(prev => [...prev, 'features']), 1600);
    setTimeout(() => setAnalysisSteps(prev => [...prev, 'classify']), 2400);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        image: imageData.split(',')[1]
      });
      
      setTimeout(() => {
        setAnalysisResult(response.data);
        const newRecord = {
          id: Date.now(),
          emotion: response.data.emotion,
          confidence: response.data.confidence,
          type: fileType,
          time: new Date().toLocaleTimeString()
        };
        setHistory(prev => [newRecord, ...prev].slice(0, 10));
        setIsAnalyzing(false);
      }, 3000);

    } catch (err) {
      console.error('Analysis failed:', err);
      setIsAnalyzing(false);
      alert('Analysis failed. Please ensure backend is running.');
    }
  };

  const steps = [
    { id: 'detect', label: 'Face Detection', icon: <ShieldCheck className="w-4 h-4" /> },
    { id: 'landmarks', label: 'Landmark Mapping', icon: <Activity className="w-4 h-4" /> },
    { id: 'features', label: 'Feature Extraction', icon: <Brain className="w-4 h-4" /> },
    { id: 'classify', label: 'Emotion Classification', icon: <Zap className="w-4 h-4" /> },
  ];

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans">
      {/* Background Glows */}
      <div className="glow-bg w-[500px] h-[500px] bg-indigo-600 top-[-100px] left-[-100px]" />
      <div className="glow-bg w-[400px] h-[400px] bg-purple-600 bottom-[-100px] right-[-100px]" />

      {/* Sidebar Navigation */}
      <aside className="w-20 lg:w-72 bg-slate-950/50 backdrop-blur-2xl border-r border-white/5 flex flex-col z-50 transition-all duration-500">
        <div className="p-4 lg:p-8">
          <div className="flex items-center space-x-3 mb-10 lg:mb-12">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20 animate-pulse">
              <Brain className="text-white w-6 h-6" />
            </div>
            <div className="hidden lg:block">
              <span className="text-lg font-bold text-white tracking-tight block leading-none">Aura AI</span>
              <span className="text-[10px] text-indigo-400 font-bold tracking-widest uppercase">Emotion Engine</span>
            </div>
          </div>

          <nav className="space-y-1.5 lg:space-y-2">
            {[
              { id: 'analyze', label: 'Analyze', icon: <Zap className="w-5 h-5" /> },
              { id: 'live', label: 'Live Detection', icon: <Camera className="w-5 h-5" /> },
              { id: 'history', label: 'Records', icon: <History className="w-5 h-5" /> },
              { id: 'stats', label: 'Metrics', icon: <BarChart3 className="w-5 h-5" /> }
            ].map(item => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3 px-3 py-3 lg:px-4 lg:py-3.5 rounded-xl lg:rounded-2xl text-sm font-semibold transition-all duration-300 ${
                  activeTab === item.id 
                  ? 'bg-indigo-500/10 text-white shadow-[0_0_20px_rgba(99,102,241,0.1)] border border-indigo-500/20' 
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/[0.02]'
                }`}
              >
                {item.icon}
                <span className="hidden lg:block">{item.label}</span>
                {activeTab === item.id && <div className="ml-auto w-1.5 h-1.5 bg-indigo-500 rounded-full shadow-[0_0_10px_#6366f1]" />}
              </button>
            ))}
          </nav>
        </div>

        <div className="mt-auto p-4 lg:p-8">
          <div className="hidden lg:flex items-center space-x-3 p-4 rounded-2xl bg-white/[0.02] border border-white/5">
            <div className={`w-2 h-2 rounded-full ${backendStatus === 'ready' ? 'bg-emerald-500 shadow-[0_0_10px_#10b981]' : 'bg-amber-500 animate-pulse shadow-[0_0_10px_#f59e0b]'}`} />
            <div>
              <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
                {backendStatus === 'ready' ? 'Core Online' : 'Linking...'}
              </p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-grow relative overflow-y-auto scroll-smooth">
        <div className="max-w-7xl mx-auto px-4 py-8 lg:px-12 lg:py-16">
          
          {activeTab === 'live' && (
            <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 space-y-10 lg:space-y-12">
              <header className="px-2">
                <h2 className="text-3xl lg:text-4xl font-black text-white tracking-tight mb-2">Live Detection</h2>
                <p className="text-slate-500 font-medium text-sm lg:text-base">Real-time neural emotion streaming via active camera feed.</p>
              </header>

              <div className="grid grid-cols-12 gap-6 lg:gap-12">
                <div className="col-span-12 lg:col-span-8">
                  <div className="relative aspect-video rounded-[2.5rem] glass-card-3d overflow-hidden border border-white/5">
                    <video 
                      ref={liveVideoRef}
                      autoPlay 
                      playsInline 
                      muted
                      className="w-full h-full object-cover"
                    />
                    <canvas ref={liveCanvasRef} className="hidden" />
                    
                    {isLive && <div className="scan-line" />}
                    
                    <div className="absolute top-6 right-6">
                      <div className={`px-4 py-2 rounded-full backdrop-blur-xl border flex items-center space-x-2 ${isLive ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-red-500/10 border-red-500/20 text-red-400'}`}>
                        <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
                        <span className="text-[10px] font-black uppercase tracking-widest">{isLive ? 'Live Feed' : 'Offline'}</span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4 lg:gap-6">
                    {!isLive ? (
                      <button 
                        onClick={startLiveDetection}
                        className="btn-3d px-12 py-4 text-lg"
                      >
                        INITIALIZE CAMERA
                      </button>
                    ) : (
                      <>
                        <button 
                          onClick={manualAnalyzeLive}
                          disabled={isLiveAnalyzing || backendStatus !== 'ready'}
                          className={`btn-3d px-12 py-4 text-lg flex items-center space-x-3 ${isLiveAnalyzing ? 'opacity-70 cursor-not-allowed' : ''}`}
                        >
                          {isLiveAnalyzing ? (
                            <>
                              <RefreshCw className="w-5 h-5 animate-spin" />
                              <span>ANALYZING...</span>
                            </>
                          ) : (
                            <>
                              <Zap className="w-5 h-5" />
                              <span>ANALYZE EMOTION</span>
                            </>
                          )}
                        </button>
                        <button 
                          onClick={stopLiveDetection}
                          className="bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 px-12 py-4 rounded-2xl font-bold transition-all"
                        >
                          TERMINATE SESSION
                        </button>
                      </>
                    )}
                  </div>
                  {liveError && <p className="text-red-400 text-center mt-4 text-sm font-bold">{liveError}</p>}
                </div>

                <div className="col-span-12 lg:col-span-4">
                  <div className={`glass-card-3d rounded-[2.5rem] p-8 h-full flex flex-col items-center justify-center text-center transition-all duration-500 ${liveResult ? 'opacity-100' : 'opacity-30'}`}>
                    <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] mb-10">Session Analysis</h3>
                    {liveResult ? (
                       <div className="space-y-8">
                         <span className="text-6xl font-black text-white tracking-tighter uppercase italic block">{liveResult.emotion}</span>
                       </div>
                     ) : (
                      <div className="flex flex-col items-center">
                        {isLiveAnalyzing ? (
                          <div className="w-12 h-12 border-2 border-indigo-500 rounded-full animate-spin mb-4" />
                        ) : (
                          <div className="w-12 h-12 border-2 border-dashed border-white/10 rounded-full mb-4 flex items-center justify-center">
                            <Zap className="w-6 h-6 text-white/20" />
                          </div>
                        )}
                        <p className="text-xs text-slate-600 font-bold uppercase tracking-widest">
                          {isLiveAnalyzing ? 'Processing...' : 'Awaiting Capture'}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'analyze' && (
            <div className="space-y-10 lg:space-y-12">
              {/* Hero Section */}
              <header className="relative px-2">
                <div className="absolute -top-24 -left-24 w-64 h-64 bg-indigo-500/10 blur-[100px] rounded-full" />
                <h1 className="text-4xl lg:text-6xl font-black text-white tracking-tight mb-4 glow-text leading-tight">
                  Facial <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">Intelligence</span>
                </h1>
                <p className="text-slate-400 text-lg font-medium max-w-2xl leading-relaxed">
                  Analyze human emotions in real-time using our proprietary 3D neural mapping technology.
                </p>
              </header>

              <div className="grid grid-cols-12 gap-6 lg:gap-12">
                {/* Interaction Zone */}
                <div className="col-span-12 lg:col-span-8 space-y-6 lg:space-y-8">
                  <div 
                    onDrop={handleDrop}
                    onDragOver={(e) => e.preventDefault()}
                    className={`
                      relative aspect-video lg:aspect-[16/9] rounded-[2rem] lg:rounded-[2.5rem] glass-card-3d overflow-hidden cursor-pointer group
                      ${previewUrl ? 'border-indigo-500/30' : 'border-dashed border-white/10 hover:border-white/20'}
                    `}
                    onClick={() => !isAnalyzing && !previewUrl && fileInputRef.current?.click()}
                  >
                    <input 
                      type="file" 
                      ref={fileInputRef} 
                      onChange={handleFileChange} 
                      className="hidden" 
                      accept="image/*,video/*" 
                    />
                    
                    {isAnalyzing && <div className="scan-line" />}

                    {!previewUrl ? (
                      <div className="absolute inset-0 flex flex-col items-center justify-center p-8 lg:p-12">
                        <div className="w-16 h-16 lg:w-20 lg:h-20 bg-indigo-500/10 rounded-2xl lg:rounded-3xl flex items-center justify-center mb-6 group-hover:scale-110 group-hover:bg-indigo-500/20 transition-all duration-700">
                          <Upload className="w-6 h-6 lg:w-8 lg:h-8 text-indigo-400" />
                        </div>
                        <h3 className="text-xl lg:text-2xl font-bold text-white mb-2">Neural Input</h3>
                        <p className="text-slate-500 text-center text-sm lg:text-base font-medium max-w-xs">
                          Drag & drop media files or click to initialize analysis engine.
                        </p>
                        <div className="mt-8 flex space-x-6 opacity-40">
                          <div className="flex items-center space-x-2 text-[10px] font-black uppercase tracking-widest"><Camera className="w-4 h-4" /> <span>JPG/PNG</span></div>
                          <div className="flex items-center space-x-2 text-[10px] font-black uppercase tracking-widest"><Video className="w-4 h-4" /> <span>MP4/MOV</span></div>
                        </div>
                      </div>
                    ) : (
                      <>
                        {fileType === 'video' ? (
                          <video 
                            ref={videoRef}
                            src={previewUrl} 
                            className="w-full h-full object-cover"
                            controls={!isAnalyzing}
                            autoPlay
                            muted
                            loop
                          />
                        ) : (
                          <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
                        )}
                        
                        <div className="absolute inset-0 bg-gradient-to-t from-[#020617] via-transparent to-transparent opacity-60" />
                        
                        {/* 3D Landmark Overlay Simulation */}
                        {isAnalyzing && (
                          <div className="absolute inset-0 pointer-events-none">
                            {[...Array(30)].map((_, i) => (
                              <div 
                                key={i} 
                                className="landmark-dot" 
                                style={{ 
                                  top: `${30 + Math.random() * 40}%`, 
                                  left: `${30 + Math.random() * 40}%`,
                                  animationDelay: `${Math.random() * 1.5}s`
                                }} 
                              />
                            ))}
                          </div>
                        )}

                        {!isAnalyzing && (
                          <button 
                            onClick={clearUpload}
                            className="absolute top-6 right-6 lg:top-8 lg:right-8 w-10 h-10 lg:w-12 lg:h-12 bg-white/10 hover:bg-red-500/20 text-white hover:text-red-400 rounded-full flex items-center justify-center backdrop-blur-xl transition-all z-30 border border-white/10"
                          >
                            <X className="w-5 h-5" />
                          </button>
                        )}
                      </>
                    )}
                    <canvas ref={canvasRef} className="hidden" />
                  </div>

                  <div className="flex flex-col sm:flex-row items-center justify-between p-6 lg:p-8 glass-card-3d rounded-[1.5rem] lg:rounded-[2rem] gap-6">
                    <div className="flex items-center space-x-4 lg:space-x-5">
                      <div className="w-12 h-12 lg:w-14 lg:h-14 rounded-xl lg:rounded-2xl bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                        <Activity className="w-6 h-6 lg:w-7 lg:h-7 text-indigo-400" />
                      </div>
                      <div>
                        <h4 className="font-bold text-white text-base lg:text-lg">Engine Parameters</h4>
                        <p className="text-[10px] text-indigo-400/60 font-black uppercase tracking-widest">Temporal Analysis Active</p>
                      </div>
                    </div>
                    <button 
                      onClick={runAnalysis}
                      disabled={!selectedFile || isAnalyzing}
                      className={`
                        btn-3d min-w-full sm:min-w-[200px] text-base lg:text-lg py-3 lg:py-4
                        ${!selectedFile || isAnalyzing ? 'opacity-50 cursor-not-allowed grayscale' : ''}
                      `}
                    >
                      {isAnalyzing ? (
                        <span className="flex items-center justify-center space-x-2">
                          <RefreshCw className="w-5 h-5 animate-spin" />
                          <span>ANALYZING...</span>
                        </span>
                      ) : (
                        'RUN INFERENCE'
                      )}
                    </button>
                  </div>
                </div>

                {/* Metrics & Results */}
                <div className="col-span-12 lg:col-span-4 space-y-6 lg:space-y-8">
                  {/* Pipeline Status */}
                  <div className="glass-card-3d rounded-[2rem] lg:rounded-[2.5rem] p-6 lg:p-8 space-y-6 lg:space-y-8">
                    <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em]">Processing Pipeline</h3>
                    <div className="space-y-4 lg:space-y-5">
                      {steps.map((step, idx) => {
                        const isCompleted = analysisSteps.includes(step.id);
                        const isCurrent = isAnalyzing && analysisSteps[analysisSteps.length - 1] === step.id;
                        
                        return (
                          <div key={step.id} className="flex items-center space-x-3 lg:space-x-4 group">
                            <div className={`w-9 h-9 lg:w-10 lg:h-10 rounded-lg lg:rounded-xl flex items-center justify-center transition-all duration-500 ${
                              isCompleted ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 
                              isCurrent ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 animate-pulse' :
                              'bg-white/5 text-slate-700 border border-white/5'
                            }`}>
                              {isCompleted ? <ShieldCheck className="w-5 h-5" /> : step.icon}
                            </div>
                            <div className="flex-grow">
                              <p className={`text-sm font-bold transition-all duration-300 ${isCompleted ? 'text-white' : 'text-slate-600'}`}>
                                {step.label}
                              </p>
                            </div>
                            {isCurrent && <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full shadow-[0_0_8px_#818cf8]" />}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Prediction Result */}
                  <div className={`glass-card-3d rounded-[2rem] lg:rounded-[2.5rem] p-6 lg:p-8 transition-all duration-700 ${
                    analysisResult ? 'opacity-100 scale-100' : 'opacity-30 scale-95 grayscale'
                  }`}>
                    <div className="text-center">
                      <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] mb-6 lg:mb-10">Neural Prediction</h3>
                      
                      {analysisResult ? (
                        <div className="space-y-6 lg:space-y-8">
                          <div className="relative inline-block">
                            <div className="absolute inset-0 bg-indigo-500/20 blur-3xl rounded-full" />
                            <span className="relative text-5xl lg:text-6xl font-black text-white tracking-tighter uppercase italic">
                              {analysisResult.emotion}
                            </span>
                          </div>
                        </div>
                      ) : (
                        <div className="py-8 lg:py-12 flex flex-col items-center">
                          <div className="w-10 h-10 lg:w-12 lg:h-12 border-2 border-dashed border-white/10 rounded-full animate-spin mb-4" />
                          <p className="text-xs lg:text-sm text-slate-600 font-bold uppercase tracking-widest">Awaiting Neural Input</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'history' && (
            <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 space-y-10 lg:space-y-12">
              <header className="px-2">
                <h2 className="text-3xl lg:text-4xl font-black text-white tracking-tight mb-2">Neural Records</h2>
                <p className="text-slate-500 font-medium text-sm lg:text-base">Historical emotion analysis data indexed by temporal signature.</p>
              </header>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
                {history.length > 0 ? history.map(item => (
                  <div key={item.id} className="glass-card-3d p-6 lg:p-8 rounded-[1.5rem] lg:rounded-[2rem] space-y-6 group">
                    <div className="flex justify-between items-start">
                      <div>
                        <span className="text-xl lg:text-2xl font-black text-white uppercase italic tracking-tighter group-hover:text-indigo-400 transition-colors">
                          {item.emotion}
                        </span>
                        <div className="flex items-center space-x-2 mt-1">
                          {item.type === 'video' ? <Video className="w-3 h-3 text-slate-500" /> : <Camera className="w-3 h-3 text-slate-500" />}
                          <span className="text-[10px] font-bold text-slate-600 uppercase tracking-widest">{item.type}</span>
                        </div>
                      </div>
                      <span className="text-[10px] font-black text-indigo-400/40 uppercase">{item.time}</span>
                    </div>
                  </div>
                )) : (
                  <div className="col-span-full py-24 lg:py-32 text-center glass-card-3d rounded-[2rem] lg:rounded-[3rem]">
                    <History className="w-10 h-10 lg:w-12 lg:h-12 text-slate-800 mx-auto mb-6" />
                    <p className="text-slate-600 text-xs lg:text-sm font-bold uppercase tracking-[0.2em]">No records found in neural database</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'stats' && (
            <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 space-y-10 lg:space-y-12">
              <header className="px-2">
                <h2 className="text-3xl lg:text-4xl font-black text-white tracking-tight mb-2">Neural Analytics</h2>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <p className="text-slate-500 font-medium text-sm lg:text-base">Advanced statistical breakdown of your emotion analysis sessions.</p>
                  <button 
                    onClick={clearAllData}
                    className="flex items-center space-x-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 rounded-xl transition-all text-[10px] font-black uppercase tracking-widest"
                  >
                    <RefreshCw className="w-3 h-3" />
                    <span>Purge Neural Cache</span>
                  </button>
                </div>
              </header>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
                <div className="glass-card-3d p-6 lg:p-8 rounded-[1.5rem] lg:rounded-[2rem] space-y-4">
                  <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Total Scans</p>
                  <p className="text-4xl lg:text-5xl font-black text-white italic">{history.length}</p>
                  <div className="flex items-center space-x-2 text-emerald-400">
                    <Activity className="w-3 h-3" />
                    <span className="text-[10px] font-bold uppercase">Active Engine</span>
                  </div>
                </div>

                <div className="glass-card-3d p-6 lg:p-8 rounded-[1.5rem] lg:rounded-[2rem] space-y-4">
                  <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Top Emotion</p>
                  <p className="text-2xl lg:text-3xl font-black text-white italic uppercase tracking-tighter">
                    {history.length > 0 
                      ? (() => {
                          const counts = history.reduce((acc, curr) => {
                            acc[curr.emotion] = (acc[curr.emotion] || 0) + 1;
                            return acc;
                          }, {});
                          return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
                        })()
                      : 'None'}
                  </p>
                  <div className="flex items-center space-x-2 text-purple-400">
                    <Zap className="w-3 h-3" />
                    <span className="text-[10px] font-bold uppercase">Dominant State</span>
                  </div>
                </div>

                <div className="glass-card-3d p-6 lg:p-8 rounded-[1.5rem] lg:rounded-[2rem] space-y-4">
                  <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Data Points</p>
                  <p className="text-4xl lg:text-5xl font-black text-white italic">{history.length * 68}</p>
                  <div className="flex items-center space-x-2 text-amber-400">
                    <ShieldCheck className="w-3 h-3" />
                    <span className="text-[10px] font-bold uppercase">Landmarks Indexed</span>
                  </div>
                </div>
              </div>

              {/* Emotion Distribution Bar Chart (CSS-based) */}
              <div className="glass-card-3d p-8 lg:p-12 rounded-[2.5rem] lg:rounded-[3rem] space-y-10">
                <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] text-center">Emotion Distribution Profile</h3>
                
                <div className="space-y-6 max-w-3xl mx-auto">
                  {['HAPPY', 'SAD', 'ANGRY', 'SURPRISED', 'NEUTRAL'].map(emotion => {
                    const count = history.filter(h => h.emotion === emotion).length;
                    const percentage = history.length > 0 ? (count / history.length) * 100 : 0;
                    
                    return (
                      <div key={emotion} className="space-y-2">
                        <div className="flex justify-between text-[10px] font-black uppercase tracking-widest">
                          <span className="text-slate-400">{emotion}</span>
                          <span className="text-white">{count} instances</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

        </div>
      </main>

      {/* Floating Settings/Status Bar (Optional) */}
      <div className="fixed bottom-6 right-6 flex items-center space-x-3 z-50">
        <button className="w-12 h-12 bg-white/5 hover:bg-white/10 backdrop-blur-xl border border-white/10 rounded-full flex items-center justify-center transition-all">
          <Settings className="w-5 h-5 text-slate-400" />
        </button>
      </div>
    </div>
  );
}

export default App;
