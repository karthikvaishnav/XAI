import React, { useState, useEffect, useRef } from "react";
import axios from 'axios';
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import GaugeChart from 'react-gauge-chart';
import DecisionTreeViz from './DecisionTreeViz';

const Plot = createPlotlyComponent(Plotly);

// --- THEME & STYLES ---
const theme = {
  bg: "#020617",
  glass: "rgba(15, 23, 42, 0.7)",
  accent: "#22d3ee",
  accentGlow: "rgba(34, 211, 238, 0.3)",
  text: "#f8fafc",
  grid: "#1e293b",
  danger: "#f43f5e",
  success: "#10b981",
  flame: "#ff5722",
  flameCore: "#ffeb3b"
};

const cardStyle = {
  background: theme.glass,
  backdropFilter: "blur(12px)",
  border: "1px solid rgba(34, 211, 238, 0.2)",
  boxShadow: `0 8px 32px 0 rgba(0, 0, 0, 0.8), inset 0 0 15px ${theme.accentGlow}`,
  padding: "20px",
  borderRadius: "8px",
  color: theme.text,
  position: "relative",
  zIndex: 2,
  marginBottom: "20px",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden"
};

const btnStyle = {
  padding: "12px 24px",
  background: "transparent",
  color: theme.accent,
  border: `1px solid ${theme.accent}`,
  borderRadius: "4px",
  fontSize: "12px",
  fontWeight: "bold",
  textTransform: "uppercase",
  letterSpacing: "2px",
  cursor: "pointer",
  boxShadow: `0 0 10px ${theme.accentGlow}`,
  transition: "0.3s all ease",
  marginTop: "10px"
};

const inputStyle = {
  width: "100%",
  boxSizing: "border-box",
  padding: "12px",
  background: "#1e293b",
  color: theme.text,
  border: `1px solid ${theme.accent}`,
  borderRadius: "4px",
  fontFamily: "monospace",
  marginBottom: "10px"
};

const checkboxStyle = {
  marginRight: "10px",
  accentColor: theme.accent,
  cursor: "pointer"
};

// --- CHART LAYOUT CONFIG ---
const sciFiPlotLayout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: theme.text, family: "monospace", size: 12 },
  xaxis: {
    gridcolor: theme.grid,
    zerolinecolor: theme.accent,
    tickfont: { color: theme.text, size: 11 },
    automargin: true
  },
  yaxis: {
    gridcolor: theme.grid,
    zerolinecolor: theme.accent,
    tickfont: { color: theme.text, size: 11 },
    automargin: true
  },
  margin: { t: 40, b: 50, l: 60, r: 20 },
  scene: {
    xaxis: { gridcolor: theme.grid, backgroundcolor: "rgba(0,0,0,0)", showbackground: false, color: theme.text },
    yaxis: { gridcolor: theme.grid, backgroundcolor: "rgba(0,0,0,0)", showbackground: false, color: theme.text },
    zaxis: { gridcolor: theme.grid, backgroundcolor: "rgba(0,0,0,0)", showbackground: false, color: theme.text },
  },
  legend: { font: { color: theme.text } }
};

// --- 🌟 NEW ANIMATION COMPONENTS ---

// 1. STAR FIELD BACKGROUND
const StarField = () => {
  return (
    <div className="star-container">
      {[...Array(50)].map((_, i) => (
        <div
          key={i}
          className={`star ${i % 3 === 0 ? 'twinkle' : ''}`}
          style={{
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 5}s`
          }}
        />
      ))}
      <div className="shooting-star" style={{ top: '10%', left: '80%' }} />
      <div className="shooting-star" style={{ top: '40%', left: '90%', animationDelay: '4s' }} />
    </div>
  );
};

// // 2. SCI-FI RINGS (HALO)
// const SciFiHalo = () => {
//   return (
//     <div className="halo-container">
//       <div className="halo-ring dashed-ring"></div>
//       <div className="halo-ring inner-pulse"></div>
//       <div className="halo-ring thin-ring"></div>
//     </div>
//   );
// };

// // 3. TOP DATA STREAM (BLINKING DOTS)
// const TopDataStream = () => {
//   const dots = [...Array(40)];
//   return (
//     <div className="data-stream-container">
//       {dots.map((_, i) => (
//         <div 
//           key={i} 
//           className="data-dot" 
//           style={{
//             animationDuration: `${1 + Math.random() * 2}s`,
//             animationDelay: `${Math.random() * 2}s` 
//           }}
//         />
//       ))}
//     </div>
//   );
// };


function App() {
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [filePath, setFilePath] = useState("");

  // Model Config
  const [targetCol, setTargetCol] = useState("");
  const [selectedModel, setSelectedModel] = useState('rf');
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [polyDegree, setPolyDegree] = useState(2);

  const [result, setResult] = useState(null);

  // Explainability State
  const [explanation, setExplanation] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [simValues, setSimValues] = useState({});
  const [simProbability, setSimProbability] = useState(0);
  const [showSimulator, setShowSimulator] = useState(false);
  const [featureMeta, setFeatureMeta] = useState({});
  const [clickedIndex, setClickedIndex] = useState(null);

  const [narrative, setNarrative] = useState("")
  const [isSpeaking, setIsSpeaking] = useState(false)

  const [isBoosting, setIsBoosting] = useState(false);
  const simulatorRef = useRef(null);
  // Manual Prediction State
  const [manualInputs, setManualInputs] = useState({});
  const [manualResult, setManualResult] = useState(null);

  const handleManualPredict = async () => {
    try {
      const res = await axios.post('http://localhost:5000/api/predict_manual', {
        inputs: manualInputs,
        model_type: selectedModel
      });
      setManualResult(res.data);
    } catch (err) { alert("Prediction Error"); }
  };

  const speakText = (text) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // Stop any previous speech
      const utterance = new SpeechSynthesisUtterance(text);

      // select a decent voice if available
      const voices = window.speechSynthesis.getVoices();
      utterance.voice = voices.find(v => v.lang.includes('en')) || voices[0];
      utterance.pitch = 1;
      utterance.rate = 1;

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);

      window.speechSynthesis.speak(utterance);
    } else {
      console.warn("Text-to-speech not supported in this browser.");
    }
  };

  const generateNarrative = (expData, baseValue) => {
    // 1. Sort features by absolute impact (highest impact first)
    const sortedFeatures = [...expData].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));

    // 2. Identify top positive and negative drivers
    const topDriver = sortedFeatures[0];
    const secondDriver = sortedFeatures[1];

    // 3. Construct the Script (Simulating GenAI)
    let script = `Here is the analysis for this data point. `;

    if (Math.abs(topDriver.shap_value) > 0.1) {
      script += `The most critical factor is ${topDriver.feature}, with a value of ${topDriver.value}. `;
      if (topDriver.shap_value > 0) {
        script += `This significantly pushes the prediction outcome higher. `;
      } else {
        script += `This pulls the prediction outcome lower. `;
      }
    }

    if (secondDriver) {
      script += `Also, ${secondDriver.feature} plays a major role. `;
    }

    script += `Overall, the local factors shift the baseline value of ${baseValue.toFixed(2)} to the final prediction.`;

    setNarrative(script);
    speakText(script);
  };


  const handleFileUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post('http://localhost:5000/api/uploads', formData);
      setColumns(res.data.columns);
      setFilePath(res.data.filePath);
      setStep(2);
    } catch (err) { alert('Upload Failed'); }
  };

  const handleFeatureToggle = (col) => {
    if (selectedFeatures.includes(col)) {
      setSelectedFeatures(selectedFeatures.filter(c => c !== col));
    } else {
      setSelectedFeatures([...selectedFeatures, col]);
    }
  };

  const handleTrain = async () => {
    setIsBoosting(true);
    setLoading(true);
    setTimeout(async () => {
      try {
        const res = await axios.post('http://localhost:5000/api/train', {
          file_path: filePath,
          target_column: targetCol,
          model_type: selectedModel,
          selected_features: selectedFeatures,
          poly_degree: polyDegree
        });
        setResult(res.data);
        setStep(3);
      } catch (err) {
        alert('Training failed: ' + (err.response?.data?.detail || "Unknown Error"));
        setIsBoosting(false);
      }
      setLoading(false);
    }, 2000);
  };

  const handlePointClick = async (data) => {
    if (result.task === 'regression') return;
    window.speechSynthesis.cancel()

    const point = data.points[0];
    const originalIndex = point.customdata;
    if (originalIndex === undefined) return;

    setClickedIndex(originalIndex);
    setExplaining(true);
    setExplanation(null);
    setShowSimulator(true);
    setNarrative("")

    setTimeout(() => { simulatorRef.current?.scrollIntoView({ behavior: "smooth" }); }, 100);

    try {
      const res = await axios.post('http://localhost:5000/api/explain', {
        index: originalIndex, model_type: selectedModel
      });

      const expData = res.data.explanation;
      const baseValue = res.data.base_value;

      const shapValues = expData.map(d => d.shap_value);
      const prediction = baseValue + shapValues.reduce((a, b) => a + b, 0);
      const values = [baseValue, ...shapValues, prediction];

      setExplanation({
        type: "waterfall",
        measure: ["absolute", ...Array(expData.length).fill("relative"), "total"],
        x: ["Base", ...expData.map(d => `${d.feature} = ${d.value.toFixed(2)}`), "Prediction"],
        y: values,
        connector: { line: { color: "rgba(255,255,255,0.4)", dash: "dot" } },
        increasing: { marker: { color: theme.success } },
        decreasing: { marker: { color: theme.danger } },
        totals: { marker: { color: theme.accent } },
        textfont: { color: "#fff", size: 13 },
        cliponaxis: false
      });

      generateNarrative(expData, baseValue)

      const featureMeta = {};
      expData.forEach(item => {
        const v = item.value;
        featureMeta[item.feature] = {
          original: v,
          min: v - Math.abs(v) * 1.5 - 0.1,
          max: v + Math.abs(v) * 1.5 + 0.1,
          step: Math.max(0.01, Math.abs(v) * 0.05)
        };
      });
      setFeatureMeta(featureMeta);
      setSimValues(expData.reduce((acc, item) => ({ ...acc, [item.feature]: item.value }), {}));
      runSimulation(expData.reduce((acc, item) => ({ ...acc, [item.feature]: item.value }), {}));

    } catch (err) {
      console.error(err);
      alert("Failed to explain this point");
    } finally {
      setExplaining(false);
    }
  };

  const handleSliderChange = (feature, value) => {
    const newValues = { ...simValues, [feature]: parseFloat(value) };
    setSimValues(newValues);
    runSimulation(newValues);
  };

  const runSimulation = async (features) => {
    try {
      const res = await axios.post('http://localhost:5000/api/simulate', {
        features: features, model_type: selectedModel
      });
      setSimProbability(res.data.probability);
    } catch (err) { console.error(err); }
  };

 // --- REGRESSION DASHBOARD ---
  const renderRegressionDashboard = () => {
    const { metrics, visuals, coefficients } = result;
    return (
      <div style={{ width: "100%", animation: "fadeIn 0.5s" }}>
        <div style={{ ...cardStyle, flexDirection: "row", justifyContent: "space-around", textAlign: "center" }}>
          <div><div style={{ color: theme.accent, fontSize: "10px" }}>R² SCORE</div><div style={{ fontSize: "24px", fontWeight: "bold" }}>{metrics.r2.toFixed(3)}</div></div>
          <div><div style={{ color: theme.danger, fontSize: "10px" }}>RMSE</div><div style={{ fontSize: "24px" }}>{metrics.rmse.toFixed(2)}</div></div>
          <div><div style={{ color: theme.flame, fontSize: "10px" }}>MAE</div><div style={{ fontSize: "24px" }}>{metrics.mae.toFixed(2)}</div></div>
        </div>

        <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>
          {/* A. ACTUAL VS PREDICTED */}
          <div style={{ ...cardStyle, flex: "1 1 450px" }}>
            <h4 style={{ margin: "0 0 10px 0", color: theme.accent }}>A // PREDICTION_ACCURACY</h4>
            <Plot
              data={[
                {
                  x: visuals.actual_vs_pred.y_true,
                  y: visuals.actual_vs_pred.y_pred,
                  mode: 'markers', type: 'scatter', name: 'Predictions',
                  marker: { color: theme.accent, opacity: 0.6 }
                },
                {
                  x: [Math.min(...visuals.actual_vs_pred.y_true), Math.max(...visuals.actual_vs_pred.y_true)],
                  y: [Math.min(...visuals.actual_vs_pred.y_true), Math.max(...visuals.actual_vs_pred.y_true)],
                  mode: 'lines', name: 'Ideal Fit',
                  line: { color: theme.danger, dash: 'dash' }
                }
              ]}
              layout={{ ...sciFiPlotLayout, height: 350, margin: { l: 40, r: 20, t: 30, b: 40 } }}
              useResizeHandler={true}
              style={{ width: "100%", height: "100%" }}
            />
          </div>

          {/* B. RESIDUALS */}
          <div style={{ ...cardStyle, flex: "1 1 450px" }}>
            <h4 style={{ margin: "0 0 10px 0", color: theme.accent }}>B // ERROR_DISTRIBUTION</h4>
            <Plot
              data={[{
                x: visuals.actual_vs_pred.y_pred,
                y: visuals.actual_vs_pred.y_true.map((t, i) => t - visuals.actual_vs_pred.y_pred[i]),
                mode: 'markers', type: 'scatter',
                marker: { color: theme.danger, opacity: 0.7 }
              },
              {
                x: [Math.min(...visuals.actual_vs_pred.y_pred), Math.max(...visuals.actual_vs_pred.y_pred)],
                y: [0, 0], mode: 'lines', line: { color: '#fff' }
              }]}
              layout={{ ...sciFiPlotLayout, height: 350, margin: { l: 40, r: 20, t: 30, b: 40 } }}
              useResizeHandler={true}
              style={{ width: "100%", height: "100%" }}
            />
          </div>
        </div>

        {/* C. TOPOLOGY SCAN (Handles 1D, 2D, and 3D+) */}
        <div style={{ ...cardStyle, height: "500px" }}>
          <h4 style={{ margin: "0 0 10px 0", color: theme.accent }}>C // FIT_TOPOLOGY_SCAN</h4>
          {visuals.regression_line ? (
            <Plot
              data={[
                {
                  x: visuals.scatter_raw.x, y: visuals.scatter_raw.y,
                  mode: 'markers', name: 'Data', marker: { color: theme.accent, opacity: 0.5 }
                },
                {
                  x: visuals.regression_line.x, y: visuals.regression_line.y,
                  mode: 'lines', name: 'Model', line: { color: theme.flame, width: 3 }
                }
              ]}
              layout={{ ...sciFiPlotLayout, height: 450 }}
              useResizeHandler={true} style={{ width: "100%", height: "100%" }}
            />
          ) : visuals.surface ? (
            <Plot
              data={[
                {
                  x: visuals.scatter_3d.x, y: visuals.scatter_3d.y, z: visuals.scatter_3d.z,
                  mode: 'markers', type: 'scatter3d', marker: { size: 3, color: theme.accent }
                },
                {
                  x: visuals.surface.x, y: visuals.surface.y, z: visuals.surface.z,
                  type: 'surface', opacity: 0.6, colorscale: 'Viridis'
                }
              ]}
              layout={{ ...sciFiPlotLayout, height: 450, margin: { l: 0, r: 0, t: 0, b: 0 } }}
              useResizeHandler={true} style={{ width: "100%", height: "100%" }}
            />
          ) : visuals.scatter_3d ? (
             /* NEW: Fallback for 3+ Features (Show PCA Data Cloud) */
             <Plot
              data={[{
                  x: visuals.scatter_3d.x, y: visuals.scatter_3d.y, z: visuals.scatter_3d.z,
                  mode: 'markers', type: 'scatter3d', 
                  marker: { 
                      size: 4, 
                      color: visuals.scatter_3d.target, // Color by actual value
                      colorscale: 'Viridis',
                      opacity: 0.8 
                  }
              }]}
              layout={{ 
                  ...sciFiPlotLayout, 
                  height: 450, 
                  margin: { l: 0, r: 0, t: 0, b: 0 },
                  scene: { ...sciFiPlotLayout.scene, xaxis:{title:'PCA1'}, yaxis:{title:'PCA2'}, zaxis:{title:'PCA3'} }
              }}
              useResizeHandler={true} style={{ width: "100%", height: "100%" }}
            />
          ) : (
            <div style={{ padding: "50px", textAlign: "center", opacity: 0.5, color: theme.danger }}>
              &gt; UNABLE TO RENDER TOPOLOGY
            </div>
          )}
        </div>

        {/* D. COEFFICIENTS */}
        <div style={{ ...cardStyle }}>
          <h4 style={{ margin: "0 0 10px 0", color: theme.accent }}>D // COEFFICIENT_VECTOR</h4>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: "10px" }}>
            {Object.entries(coefficients).map(([key, val]) => (
              <div key={key} style={{ background: "rgba(0,0,0,0.3)", padding: "5px", borderRadius: "4px" }}>
                <div style={{ fontSize: "10px", color: theme.accent }}>{key}</div>
                <div style={{ color: val > 0 ? theme.success : theme.danger }}>{typeof val === 'number' ? val.toFixed(4) : val}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // --- MAIN RENDER ---
  return (
    <div style={{
      backgroundColor: theme.bg, minHeight: "100vh", width: "100vw", padding: "20px", boxSizing: "border-box",
      fontFamily: "monospace", color: theme.text, backgroundImage: `radial-gradient(circle at center, #0f172a 0%, #020617 100%)`,
      display: "flex", flexDirection: "column", overflowX: "hidden", position: "relative"
    }}>

      {/* 🌟 1. BACKGROUND STARS */}
      <StarField />

      <div className="scanline"></div>

      {/* ROCKET ANIMATION */}
      {(step === 2 || isBoosting) && (
        <div className={`rocket-container ${isBoosting ? 'launching' : 'hovering'}`}>
          <div className="rocket">
            <div className="rocket-body"><div className="window"></div></div>
            <div className="fin fin-left"></div><div className="fin fin-right"></div>
            <div className={`exhaust-flame ${isBoosting ? 'boost' : ''}`}></div>
          </div>
        </div>
      )}

      <header style={{ width: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", margin: "20px 0 40px 0", flexShrink: 0, position: "relative", zIndex: 10 }}>
        <div className="terminal-container"><h1 className="terminal-text">XAI_COMMAND_DECK</h1></div>
        <div style={{ fontSize: "10px", color: theme.accent, marginTop: "10px", letterSpacing: "2px" }}>
          [ SYSTEM_STATUS: {loading ? "BOOST_SEQUENCER_ACTIVE" : "READY"} ]
        </div>
      </header>

      <main style={{ flexGrow: 1, display: "flex", flexDirection: "column", zIndex: 10, maxWidth: "1600px", margin: "0 auto", width: "100%", position: "relative" }}>

        {/* STEP 1: UPLOAD */}
        {step === 1 && (
          <div style={{ position: "relative", minHeight: "400px", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>

            {/* 🌟 2. HALO & DATA STREAM */}
            {/* <TopDataStream /> */}
            {/* <SciFiHalo /> */}

            <div style={{ ...cardStyle, width: "100%", maxWidth: "500px", textAlign: "center", position: "relative", zIndex: 2 }}>
              <h2 style={{ color: theme.accent, marginBottom: "20px", fontSize: "14px" }}>&gt; INITIALIZE_DATA_STREAM</h2>
              <input type="file" onChange={(e) => setFile(e.target.files[0])} accept=".csv" style={{ ...inputStyle, cursor: "pointer" }} />
              <button onClick={handleFileUpload} style={btnStyle} disabled={!file}>ESTABLISH LINK</button>
            </div>
          </div>
        )}

        {/* STEP 2: CONFIGURATION */}
        {step === 2 && (
          <div style={{ position: "relative", minHeight: "400px", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>

            {/* 🌟 2. HALO & DATA STREAM */}
            {/* <TopDataStream /> */}
            {/* <SciFiHalo /> */}

            <div style={{ ...cardStyle, width: "100%", maxWidth: "600px", position: "relative", zIndex: 2 }}>
              <h2 style={{ color: theme.accent, marginBottom: "20px", fontSize: "14px", textAlign: "center" }}>&gt; CONFIGURE_CORE</h2>

              <div style={{ marginBottom: "15px" }}>
                <label style={{ fontSize: '10px', display: 'block', marginBottom: '5px', opacity: 0.7 }}>TARGET_VECTOR (Y)</label>
                <select style={inputStyle} onChange={(e) => setTargetCol(e.target.value)}>
                  <option value="">-- Select --</option>
                  {columns.map(col => <option key={col} value={col}>{col}</option>)}
                </select>
              </div>

              {/* FEATURE SELECTION */}
              <div style={{ marginBottom: "15px" }}>
                <label style={{ fontSize: '10px', display: 'block', marginBottom: '5px', opacity: 0.7 }}>INPUT_VECTORS (X)</label>
                <div style={{ height: "150px", overflowY: "scroll", background: "rgba(0,0,0,0.3)", padding: "10px", border: `1px solid ${theme.grid}` }}>
                  {(() => {
                    const availableFeatures = columns.filter(c => c !== targetCol);
                    const isAllSelected = availableFeatures.length > 0 && selectedFeatures.length === availableFeatures.length;
                    return (
                      <>
                        <div style={{ display: "flex", alignItems: "center", marginBottom: "10px", borderBottom: `1px solid ${theme.grid}`, paddingBottom: "5px" }}>
                          <input type="checkbox" style={checkboxStyle} checked={isAllSelected}
                            onChange={() => {
                              if (isAllSelected) setSelectedFeatures([]);
                              else setSelectedFeatures(availableFeatures);
                            }}
                          />
                          <span style={{ fontSize: "12px", fontWeight: "bold", color: theme.accent }}>SELECT ALL_FEATURES</span>
                        </div>
                        {availableFeatures.map(c => (
                          <div key={c} style={{ display: "flex", alignItems: "center", marginBottom: "5px" }}>
                            <input type="checkbox" style={checkboxStyle} onChange={() => handleFeatureToggle(c)} checked={selectedFeatures.includes(c)} />
                            <span style={{ fontSize: "12px" }}>{c}</span>
                          </div>
                        ))}
                      </>
                    );
                  })()}
                </div>
              </div>

              <div style={{ marginBottom: "25px" }}>
                <label style={{ fontSize: '10px', display: 'block', marginBottom: '5px', opacity: 0.7 }}>MODEL_ARCHITECTURE</label>
                <select style={inputStyle} value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                  <optgroup label="Regression (New)">
                    <option value="linear">Linear Regression</option>
                    <option value="ridge">Ridge (L2)</option>
                    <option value="lasso">Lasso (L1)</option>
                    <option value="poly">Polynomial Regression</option>
                  </optgroup>
                  <optgroup label="Classification (Legacy)">
                    <option value="rf">Random Forest</option>
                    <option value="logistic">Logistic Regression</option>
                    <option value="dt">Decision Tree</option>
                  </optgroup>
                </select>
              </div>

              {selectedModel === 'poly' && (
                <div style={{ marginBottom: "25px", border: `1px solid ${theme.danger}`, padding: "10px" }}>
                  <label style={{ fontSize: '10px', display: 'block', marginBottom: '5px', color: theme.danger }}>POLYNOMIAL DEGREE: {polyDegree}</label>
                  <input type="range" min="2" max="5" value={polyDegree} className="scifi-slider"
                    onChange={(e) => setPolyDegree(parseInt(e.target.value))}
                  />
                </div>
              )}

              <button onClick={handleTrain} style={{ ...btnStyle, width: "100%" }} disabled={!targetCol || loading}>
                {loading ? "IGNITION IN PROGRESS..." : "EXECUTE_SEQUENCE"}
              </button>
            </div>
          </div>
        )}

        {/* STEP 3: RESULTS & DASHBOARD */}
        {step === 3 && result && (
          <div style={{ width: "100%" }}>
            <button onClick={() => { setStep(2); setIsBoosting(false); setShowSimulator(false); }} style={{ ...btnStyle, marginBottom: "20px" }}>&lt; RE_CONFIG</button>

            {result.task === 'regression' ? renderRegressionDashboard() : (() => {
              const { metrics, confusion_matrix, feature_importance } = result;
              return (
                <div style={{ width: "100%", animation: "fadeIn 0.5s" }}>
                  {/* METRICS ROW */}
                  <div style={{ ...cardStyle, flexDirection: "row", justifyContent: "space-around", textAlign: "center" }}>
                    <div><div style={{ color: theme.accent, fontSize: "10px" }}>ACCURACY</div><div style={{ fontSize: "24px", fontWeight: "bold" }}>{(metrics.accuracy * 100).toFixed(1)}%</div></div>
                    <div><div style={{ color: theme.success, fontSize: "10px" }}>PRECISION</div><div style={{ fontSize: "24px" }}>{metrics.precision.toFixed(2)}</div></div>
                    <div><div style={{ color: "#fab005", fontSize: "10px" }}>RECALL</div><div style={{ fontSize: "24px" }}>{metrics.recall.toFixed(2)}</div></div>
                    <div><div style={{ color: theme.danger, fontSize: "10px" }}>F1 SCORE</div><div style={{ fontSize: "24px" }}>{metrics.f1.toFixed(2)}</div></div>
                  </div>

                  {selectedModel === 'dt' && (
                    <div style={{ marginBottom: "20px" }}>
                      <h4 style={{ color: theme.accent, margin: "0 0 10px 0", letterSpacing: "2px" }}>// LOGIC_TREE_BLUEPRINT</h4>
                      <DecisionTreeViz filePath={filePath} targetCol={targetCol} />
                    </div>
                  )}

                  <div style={{ display: "flex", gap: "20px", flexWrap: "wrap", marginBottom: "20px" }}>
                    <div style={{ ...cardStyle, flex: "1 1 500px", height: "500px" }}>
                      <h4 style={{ margin: "0 0 10px 0", color: theme.accent }}>CONFUSION_MATRIX_HEATMAP</h4>
                      <Plot
                        data={[{ z: confusion_matrix.z, x: confusion_matrix.x, y: confusion_matrix.y, type: 'heatmap', colorscale: 'Viridis', showscale: true }]}
                        layout={{ ...sciFiPlotLayout, title: 'Predicted vs Actual', xaxis: { title: 'Predicted Class', side: 'bottom', color: theme.text }, yaxis: { title: 'Actual Class', autorange: 'reversed', color: theme.text }, margin: { l: 60, r: 20, t: 50, b: 50 } }}
                        useResizeHandler={true} style={{ width: "100%", height: "100%" }}
                      />
                    </div>
                    <div style={{ ...cardStyle, flex: "1 1 500px", height: "500px" }}>
                      <h4 style={{ color: theme.accent, margin: "0 0 10px 0" }}> GLOBAL_FEATURE_IMPORTANCE</h4>
                      <div style={{ flexGrow: 1 }}>
                        <Plot
                          data={[{ x: feature_importance.map(i => i.feature), y: feature_importance.map(i => i.importance), type: 'bar', marker: { color: theme.accent, opacity: 0.7, line: { color: theme.text, width: 1 } } }]}
                          layout={{ ...sciFiPlotLayout, autosize: true, margin: { l: 30, r: 10, t: 10, b: 60 } }}
                          style={{ width: "100%", height: "100%" }} useResizeHandler={true}
                        />
                      </div>
                    </div>
                    <div style={{ ...cardStyle, flex: "1 1 500px", height: "500px" }}>
                      <h4 style={{ color: theme.accent, margin: "0 0 10px 0" }}> 3D_MANIFOLD_PROJECTION</h4>
                      <div style={{ flexGrow: 1 }}>
                        <Plot
                          data={(() => {
                            const uniqueTargets = [...new Set(result.scatter_data.map(d => d.target))];
                            return uniqueTargets.map(targetVal => {
                              const group = result.scatter_data.filter(d => d.target === targetVal);
                              return { x: group.map(d => d.x), y: group.map(d => d.y), z: group.map(d => d.z), mode: 'markers', type: 'scatter3d', name: `Class ${targetVal}`, marker: { size: 5, opacity: 0.8 } };
                            });
                          })()}
                          layout={{ ...sciFiPlotLayout, autosize: true, margin: { l: 0, r: 0, t: 0, b: 0 } }}
                          style={{ width: "100%", height: "100%" }} useResizeHandler={true}
                        />
                      </div>
                    </div>
                  </div>

                  <div style={{ ...cardStyle, height: "750px", marginBottom: "20px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}><h4 style={{ color: theme.accent, margin: 0 }}> INTERACTIVE_2D_PROJECTION</h4><span style={{ fontSize: "12px", color: theme.success, animation: "blink 2s infinite" }}>&gt;&gt; CLICK_A_DATA_POINT_TO_ANALYZE</span></div>
                    <div style={{ flexGrow: 1, marginTop: "10px" }}>
                      <Plot
                        onInitialized={(figure, graphDiv) => { graphDiv.on('plotly_click', (eventData) => { const point = eventData.points[0]; if (point && point.customdata !== undefined) { handlePointClick({ points: [{ customdata: point.customdata }] }); } }); }}
                        data={(() => {
                          const uniqueTargets = [...new Set(result.scatter_data1.map(d => d.target))];
                          return uniqueTargets.map(targetVal => {
                            const group = result.scatter_data1.filter(d => d.target === targetVal);
                            return { x: group.map(d => d.x), y: group.map(d => d.y), customdata: group.map(d => d.original_index), mode: "markers", type: "scatter", name: `Class ${targetVal}`, marker: { size: 12, opacity: 0.7, line: { width: 1, color: theme.text } } };
                          });
                        })()}
                        layout={{ ...sciFiPlotLayout, autosize: true, clickmode: "event", hovermode: "closest", margin: { l: 80, r: 20, t: 30, b: 80 }, xaxis: { ...sciFiPlotLayout.xaxis, title: { text: "Principal Component 1", font: { size: 14, color: theme.accent } } }, yaxis: { ...sciFiPlotLayout.yaxis, title: { text: "Principal Component 2", font: { size: 14, color: theme.accent } } } }}
                        style={{ width: "100%", height: "100%" }} useResizeHandler={true}
                      />
                    </div>
                  </div>
                </div>
              );
            })()}

            {showSimulator && result.task !== 'regression' && (
              <div ref={simulatorRef} style={{ ...cardStyle, border: `2px solid ${theme.accent}`, boxShadow: `0 0 30px ${theme.accentGlow}`, animation: "fadeIn 0.5s ease-in-out" }}>

                {/* Header */}
                <div style={{ display: "flex", borderBottom: `1px solid ${theme.grid}`, paddingBottom: "10px", marginBottom: "20px", alignItems: "center", justifyContent: "space-between" }}>
                  <h2 style={{ color: theme.accent, margin: 0 }}>ANALYSIS_CONSOLE // ID: {clickedIndex}</h2>
                  <button onClick={() => {
                    const originalValues = Object.fromEntries(Object.entries(featureMeta).map(([f, meta]) => [f, meta.original]));
                    setSimValues(originalValues); runSimulation(originalValues);
                  }} style={{ ...btnStyle, marginTop: 0, padding: "8px 16px" }}>RESET_SIMULATION</button>
                </div>

                {/* MAIN CONTENT ROW */}
                <div style={{ display: "flex", flexWrap: "wrap", gap: "30px" }}>

                  {/* 🟢 LEFT COLUMN: Waterfall Chart + AI Narrator */}
                  <div style={{ flex: "1 1 500px", display: "flex", flexDirection: "column", gap: "20px" }}>

                    {/* 1. Waterfall Chart */}
                    <div style={{ paddingRight: "10px" }}>
                      <h4 style={{ color: theme.text, marginTop: 0 }}>// DECISION_LOGIC_BREAKDOWN</h4>
                      {explaining ? (
                        <div style={{ padding: "50px", textAlign: "center", color: theme.accent }}>&gt; DECRYPTING_MODEL_LOGIC...</div>
                      ) : explanation && (
                        <Plot
                          data={[{ ...explanation }]}
                          key={clickedIndex}
                          layout={{
                            ...sciFiPlotLayout,
                            height: 400, // Reduced height slightly to fit narrator better
                            margin: { l: 60, r: 30, t: 30, b: 140 },
                            yaxis: { title: "SHAP Impact", zeroline: true, zerolinecolor: theme.accent },
                            xaxis: { tickangle: -35, automargin: true }
                          }}
                          style={{ width: "100%", height: "400px" }}
                          useResizeHandler={true}
                        />
                      )}
                    </div>

                    {/* 2. AI Narrator (Moved Here) */}
                    <div style={{ width: "100%", animation: "fadeIn 0.8s" }}>
                      <div style={{
                        background: "rgba(2, 6, 23, 0.8)",
                        border: `1px solid ${theme.success}`,
                        boxShadow: `0 0 20px rgba(16, 185, 129, 0.1)`,
                        borderRadius: "6px",
                        padding: "20px",
                        position: "relative",
                        overflow: "hidden"
                      }}>
                        <div style={{ position: "absolute", top: 0, right: 0, width: "20px", height: "20px", borderTop: `2px solid ${theme.success}`, borderRight: `2px solid ${theme.success}` }} />
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "15px", borderBottom: `1px solid ${theme.grid}`, paddingBottom: "10px" }}>
                          <h4 style={{ margin: 0, color: theme.success, fontSize: "12px", letterSpacing: "3px", display: "flex", alignItems: "center", gap: "10px" }}>
                            <span style={{ fontSize: "16px" }}>⚡</span> NEURAL_NARRATIVE_LINK
                          </h4>
                          {isSpeaking && <div style={{ color: theme.success, fontSize: "10px", animation: "blink 1s infinite" }}>[ TRANSMITTING_AUDIO_DATA... ]</div>}
                        </div>

                        <div style={{ display: "flex", gap: "20px", alignItems: "flex-start" }}>
                          <div style={{ flexGrow: 1 }}>
                            <p style={{ fontFamily: "monospace", fontSize: "14px", lineHeight: "1.7", color: theme.text, margin: 0, borderLeft: `3px solid ${theme.success}`, paddingLeft: "20px", textShadow: "0 0 5px rgba(0,0,0,0.5)" }}>
                              "{narrative || "AWAITING_DATA_STREAM_ANALYSIS..."}"
                            </p>
                          </div>
                          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", minWidth: "150px" }}>
                            <div style={{ display: "flex", alignItems: "flex-end", height: "30px", gap: "3px", marginBottom: "15px" }}>
                              {isSpeaking ? [...Array(8)].map((_, i) => <div key={i} className="audio-bar" style={{ animationDelay: `${i * 0.1}s` }} />) : <div style={{ height: "2px", width: "100%", background: theme.grid }} />}
                            </div>
                            <button onClick={() => speakText(narrative)} style={{
                              background: isSpeaking ? "rgba(244, 63, 94, 0.1)" : "rgba(16, 185, 129, 0.1)",
                              border: `1px solid ${isSpeaking ? theme.danger : theme.success}`,
                              color: isSpeaking ? theme.danger : theme.success,
                              padding: "10px 16px", fontSize: "11px", fontWeight: "bold", letterSpacing: "1px", cursor: "pointer", textTransform: "uppercase", width: "100%", transition: "all 0.3s ease", boxShadow: isSpeaking ? `0 0 10px ${theme.danger}` : "none"
                            }}>
                              {isSpeaking ? "■ HALT_STREAM" : "▶ PLAY_LOG"}
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>

                  </div>

                  {/* 🔵 RIGHT COLUMN: Simulator (Thinner) */}
                  <div style={{ flex: "0 0 300px", borderLeft: `1px solid ${theme.grid}`, paddingLeft: "20px" }}>
                    <h4 style={{ color: theme.text, marginTop: 0 }}>// PREDICTION_SIMULATOR</h4>
                    <div style={{ textAlign: "center", marginBottom: "30px", background: "rgba(0,0,0,0.3)", padding: "10px", borderRadius: "8px" }}>
                      <GaugeChart id="gauge-chart1" nrOfLevels={3} colors={[theme.danger, "#FFC371", theme.success]} arcWidth={0.3} percent={simProbability} textColor={theme.text} needleColor={theme.accent} needleBaseColor={theme.accent} />
                      <h3 style={{ color: theme.accent, margin: "5px 0" }}>PROBABILITY: {(simProbability * 100).toFixed(1)}%</h3>
                    </div>
                    <div style={{ maxHeight: "600px", overflowY: "auto", paddingRight: "10px" }}>
                      {Object.keys(simValues).map(feature => (
                        <div key={feature} style={{ marginBottom: "20px" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: theme.accent, marginBottom: "8px" }}>
                            <span>{feature}</span><span style={{ color: theme.text }}>{simValues[feature].toFixed(2)}</span>
                          </div>
                          <input type="range" className="scifi-slider"
                            min={featureMeta[feature]?.min ?? -10}
                            max={featureMeta[feature]?.max ?? 10}
                            step={featureMeta[feature]?.step ?? 0.1}
                            value={simValues[feature]}
                            onChange={(e) => handleSliderChange(feature, parseFloat(e.target.value))}
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                </div>
              </div>
            )}
            {/* 🎮 NEW: MANUAL PREDICTION TERMINAL */}
            <div style={{ ...cardStyle, border: `1px solid ${theme.accent}`, marginTop: "30px", animation: "fadeIn 0.8s" }}>
              <div style={{ borderBottom: `1px solid ${theme.grid}`, paddingBottom: "15px", marginBottom: "20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <h4 style={{ margin: 0, color: theme.accent, letterSpacing: "2px" }}>&gt; MANUAL_OVERRIDE_TERMINAL</h4>
                  <div style={{ fontSize: "10px", color: theme.text, opacity: 0.7 }}>Inject raw data values for real-time inference</div>
                </div>
                {manualResult && (
                  <div style={{ textAlign: "right", animation: "popIn 0.3s" }}>
                    <div style={{ fontSize: "10px", color: theme.text }}>CALCULATED OUTPUT</div>
                    <div style={{ fontSize: "24px", fontWeight: "bold", color: theme.success, textShadow: `0 0 15px ${theme.success}` }}>
                      {typeof manualResult.prediction === 'number' ? manualResult.prediction.toFixed(2) : manualResult.prediction}
                    </div>
                    {!manualResult.is_regression && (
                      <div style={{ fontSize: "10px", color: theme.accent }}>CONFIDENCE: {(manualResult.confidence * 100).toFixed(1)}%</div>
                    )}
                  </div>
                )}
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "15px" }}>
                {/* INPUTS GENERATOR */}
                {(selectedFeatures.length > 0 ? selectedFeatures : columns.filter(c => c !== targetCol)).map(col => (
                  <div key={col}>
                    <label style={{ fontSize: "10px", color: theme.accent, display: "block", marginBottom: "5px" }}>{col.toUpperCase()}</label>
                    <input
                      type="text"
                      style={{ ...inputStyle, marginBottom: 0, border: `1px solid ${theme.grid}` }}
                      placeholder="VALUE"
                      onChange={(e) => setManualInputs({ ...manualInputs, [col]: e.target.value })}
                    />
                  </div>
                ))}
              </div>

              <button onClick={handleManualPredict} style={{ ...btnStyle, background: theme.accent, color: theme.bg, marginTop: "20px", width: "100%" }}>
                RUN_INFERENCE_PROTOCOL
              </button>
            </div>
          </div>
        )}
      </main>

      {/* 🌟 CSS ANIMATIONS */}
      <style>{`
      /* --- 🌟 AUDIO WAVE ANIMATION --- */
        .audio-wave {
          display: flex;
          align-items: center;
          gap: 3px;
          height: 12px;
        }
        .wave-bar {
          width: 3px;
          height: 100%;
          background-color: ${theme.danger};
          animation: waveform 0.8s infinite ease-in-out;
        }
        .wave-bar:nth-child(1) { animation-delay: -0.4s; }
        .wave-bar:nth-child(2) { animation-delay: -0.2s; }
        .wave-bar:nth-child(3) { animation-delay: 0s; }

        @keyframes waveform {
          0%, 100% { height: 30%; }
          50% { height: 100%; }
        }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(34, 211, 238, 0.3); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #22d3ee; }
        body { margin: 0; overflow-x: hidden; background: #020617; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        input[type=range].scifi-slider { -webkit-appearance: none; width: 100%; background: transparent; }
        input[type=range].scifi-slider:focus { outline: none; }
        input[type=range].scifi-slider::-webkit-slider-runnable-track { width: 100%; height: 6px; cursor: pointer; background: ${theme.grid}; border-radius: 2px; border: 1px solid ${theme.accentGlow}; box-shadow: inset 0 0 5px #000; }
        input[type=range].scifi-slider::-webkit-slider-thumb { height: 18px; width: 18px; border-radius: 50%; background: ${theme.bg}; border: 2px solid ${theme.accent}; cursor: pointer; -webkit-appearance: none; margin-top: -7px; box-shadow: 0 0 10px ${theme.accent}; transition: transform 0.1s; }
        input[type=range].scifi-slider:focus::-webkit-slider-thumb { background: ${theme.accent}; transform: scale(1.2); }
        .rocket-container { position: fixed; right: 15%; z-index: 999; will-change: transform; }
        .hovering { bottom: 15%; animation: hover-shake 3s infinite ease-in-out; }
        .launching { animation: launch-boost-final 2s forwards cubic-bezier(.62,-0.01,.23,1); }
        .rocket { width: 60px; height: 140px; }
        .rocket-body { width: 60px; height: 110px; background: #cbd5e1; border-radius: 50% 50% 10% 10%; border: 2px solid #94a3b8; position: relative; }
        .window { width: 20px; height: 20px; background: ${theme.accent}; border-radius: 50%; position: absolute; top: 25%; left: 50%; transform: translateX(-50%); box-shadow: 0 0 15px ${theme.accent}; }
        .fin { width: 30px; height: 45px; background: #64748b; position: absolute; bottom: 15px; z-index: -1; }
        .fin-left { left: -18px; border-radius: 100% 0 0 0; }
        .fin-right { right: -18px; border-radius: 0 100% 0 0; }
        .exhaust-flame { position: absolute; top: 90%; left: 50%; transform: translateX(-50%); width: 25px; height: 60px; background: linear-gradient(to bottom, ${theme.flameCore}, ${theme.flame}, transparent); border-radius: 0 0 50% 50%; filter: blur(2px); animation: flicker 0.1s infinite alternate; }
        .exhaust-flame.boost { height: 200px; width: 50px; background: linear-gradient(to bottom, #ffffff, ${theme.flameCore}, ${theme.flame}, transparent); filter: blur(6px); }
        @keyframes hover-shake { 0%, 100% { transform: translateY(0) rotate(0deg); } 50% { transform: translateY(-15px) rotate(1deg); } }
        @keyframes launch-boost-final { 0% { transform: translateY(0); bottom: 15%; } 15% { transform: translateY(20px); bottom: 15%; } 100% { transform: translateY(-150vh); bottom: 15%; } }
        @keyframes flicker { 0% { opacity: 0.8; transform: translateX(-50%) scaleX(0.9); } 100% { opacity: 1; transform: translateX(-50%) scaleX(1.1); } }
        .terminal-text { font-size: clamp(1.2rem, 5vw, 2.5rem); letter-spacing: 8px; color: ${theme.accent}; text-shadow: 0 0 10px ${theme.accentGlow}; white-space: nowrap; overflow: hidden; border-right: 4px solid ${theme.accent}; animation: typing 4s steps(16, end) infinite, blink-cursor 0.75s step-end infinite; }
        @keyframes typing { 0%, 100% { width: 0 } 50%, 90% { width: 100% } }
        @keyframes blink-cursor { from, to { border-color: transparent } 50% { border-color: ${theme.accent} } }
        @keyframes blink { 50% { opacity: 0; } }
        .scanline { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%); background-size: 100% 4px; z-index: 1000; pointer-events: none; opacity: 0.3; }

        /* --- 🌟 STAR BACKGROUND ANIMATION --- */
        .star-container {
          position: fixed; top: 0; left: 0; width: 100%; height: 100%;
          z-index: 0; pointer-events: none;
        }
        .star {
          position: absolute; width: 2px; height: 2px;
          background: white; border-radius: 50%; opacity: 0.5;
        }
        .twinkle { animation: twinkle-pulse 3s infinite ease-in-out; }
        @keyframes twinkle-pulse {
          0%, 100% { opacity: 0.3; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.5); }
        }
        .shooting-star {
          position: absolute; width: 2px; height: 2px;
          background: linear-gradient(90deg, white, transparent);
          border-radius: 50%; opacity: 0;
          animation: shoot 8s infinite linear;
        }
        @keyframes shoot {
          0% { transform: translateX(0) translateY(0) scale(1); opacity: 1; }
          10% { transform: translateX(-500px) translateY(500px) scale(0); opacity: 0; }
          100% { opacity: 0; }
        }

        /* --- 🌟 HALO ANIMATION --- */
        .halo-container {
          position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
          width: 600px; height: 600px; z-index: 0; pointer-events: none;
          display: flex; align-items: center; justify-content: center;
        }
        .halo-ring { position: absolute; border-radius: 50%; }
        .dashed-ring {
          width: 500px; height: 500px; border: 1px dashed ${theme.accent};
          opacity: 0.2; animation: spin 20s linear infinite;
        }
        .inner-pulse {
          width: 350px; height: 350px; border: 1px solid ${theme.accent};
          opacity: 0.1; box-shadow: 0 0 20px ${theme.accent};
          animation: pulse-ring 4s ease-in-out infinite;
        }
        .thin-ring {
          width: 420px; height: 420px; border-top: 1px solid ${theme.success};
          border-bottom: 1px solid ${theme.success}; opacity: 0.3;
          animation: spin-reverse 15s linear infinite;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        @keyframes spin-reverse { 100% { transform: rotate(-360deg); } }
        @keyframes pulse-ring {
          0%, 100% { transform: scale(0.9); opacity: 0.1; }
          50% { transform: scale(1.05); opacity: 0.2; }
        }

        /* --- 🌟 TOP DATA STREAM ANIMATION --- */
        .data-stream-container {
          position: absolute; top: -60px; left: 50%; transform: translateX(-50%);
          width: 90%; display: flex; justify-content: space-between; z-index: 0; pointer-events: none;
        }
        .data-dot {
          width: 4px; height: 4px; border-radius: 50%; background-color: ${theme.accent};
          opacity: 0.2; box-shadow: 0 0 5px ${theme.accent};
          animation: data-blink infinite alternate;
        }
        @keyframes data-blink {
          0% { opacity: 0.1; transform: scale(0.8); }
          100% { opacity: 1; transform: scale(1.2); box-shadow: 0 0 8px ${theme.accent}; }
        }
      `}</style>
    </div>
  );
}

export default App;