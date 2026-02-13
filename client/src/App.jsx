import { useState } from "react";
import axios from 'axios';
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { useEffect, useRef } from "react";

const Plot = createPlotlyComponent(Plotly);


function App(){
  const [loading,setLoading] = useState(false)

  const [step,setStep] = useState(1)
  const [file,setFile] = useState(null)
  const [columns,setColumns] = useState([])
  const [filePath, setFilePath] = useState("")

  const [targetCol,setTargetCol] = useState("")
  const [selectedModel, setSelectedModel] = useState('rf')

  const [result,setResult] = useState(null)

  const [explanation, setExplanation] = useState(null);
  const [explaining, setExplaining] = useState(false)

  const plotRef = useRef(null);

  useEffect(() => {
  if (!plotRef.current) return;

  const gd = plotRef.current;

  const handler = (event) => {
    console.log("✅ plotly_click fired");

    const point = event.points[0];
    const originalIndex = point.customdata;

    handlePointClick({
      points: [{ customdata: originalIndex }],
    });
  };

  gd.on("plotly_click", handler);

  // cleanup (important in React)
  return () => {
    gd.removeListener("plotly_click", handler);
  };
}, [result]);



  const handleFileUpload = async() =>{
    if(!file) return 
    const formData = new FormData();
    formData.append('file',file)

    try{
      const res = await axios.post('http://localhost:5000/api/uploads',formData)
      setColumns(res.data.columns)
      setFilePath(res.data.filePath)
      setStep(2)
    }catch(err){
      console.error(err)
      alert('upload Failed')
    }

  }

  const handleTrain = async()=>{
    setLoading(true)
    try{
      const res = await axios.post('http://localhost:5000/api/train',{
        file_path : filePath,
        target_column : targetCol,
        model_type : selectedModel
      })
      setResult(res.data)
      setStep(3)
    }catch(err){
      console.error(err)
      alert('training failed')
    }
    setLoading(false)
  }
  

  // const [accuracy, setAccuracy] = useState(null)
  // const [chartData, setChartData] = useState(null)
  // const [scatterPlot, setScatterPlot] = useState(null)
  const handlePointClick = async(data)=>{
    console.log('point clicked', data);

  const point = data.points[0];
  const originalIndex = point.customdata;

  if (originalIndex === undefined) {
    console.log("No customdata found");
    alert("No index found for this point");
    return;
  }

  console.log(`Trying to explain index: ${originalIndex} | model: ${selectedModel}`);

  setExplaining(true);
  setExplanation(null);

  try {
    const res = await axios.post('http://localhost:5000/api/explain', {
      index: originalIndex,
      model_type: selectedModel
    });

    console.log("Explain response:", res.data);  // ← add this

    const expData = res.data.explanation;
    const baseValue = res.data.base_value;

      setExplanation({
        type: "waterfall",
        orientation: "h",
        measure: Array(expData.length).fill("relative"), 
        y: expData.map(d => `${d.feature} = ${d.value.toFixed(2)}`), // Label: "Age = 50"
        x: expData.map(d => d.shap_value), // Value: Impact
        connector: { mode: "between", line: { width: 4, color: "rgb(0, 0, 0)", dash: 0 } },
        decreasing: { marker: { color: "#ff4d4d" } }, // Red
        increasing: { marker: { color: "#28a745" } }, // Green
        base: baseValue
      });
    }catch(err){
      console.error(err);
      alert("failed to explain this point")
      if (err.response) {
      // Server responded with error (4xx / 5xx)
      console.log("Response status:", err.response.status);
      console.log("Response data:", err.response.data);
      alert(`Server error: ${err.response.status} - ${JSON.stringify(err.response.data)}`);
    } else if (err.request) {
      // No response received (network / CORS / server down)
      console.log("No response received:", err.request);
      alert("Cannot reach the backend. Is the server running at http://localhost:5000?");
    } else {
      // Other error (e.g. bad config)
      alert(`Unexpected error: ${err.message}`);
    }
  } finally {
    setExplaining(false);
  }
  }
  


  return(
     <div style={{ padding: "40px", fontFamily: "sans-serif", maxWidth: "1200px", margin: "0 auto" }}>
      <h1>Advanced XAI Workbench</h1>

      {step === 1 && (
        <div style={cardStyle}>
          <h2>Step 1: Upload Document</h2>
          <p>Upload a CSV file</p>
          <input type="file" onChange={(e)=> setFile(e.target.files[0])} accept=".csv" />
          <button onClick={handleFileUpload} style={btnStyle} disabled={!file}>Upload & Analyze</button>
        </div>
      )}

      {step === 2 && (
        <div style={cardStyle}>
          <h2>Step 2 : Configuration</h2>

          <div style={{marginBottom:"20px"}}>
            <label><strong>Select Target Column (What to predict):</strong></label>
            <select  
              style={{ display: "block", width: "100%", padding: "10px", marginTop: "5px" }}
              onChange={(e)=> setTargetCol(e.target.value)}
            >
              <option value="">-- Select Column --</option>
              {columns.map(col=><option key ={col} value={col}>{col}</option>)}
            </select>
          </div>

          <div style={{ marginBottom: "20px" }}>
            <label><strong>Select Model Architecture:</strong></label>
            <select 
              style={{ display: "block", width: "100%", padding: "10px", marginTop: "5px" }}
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="rf">Random Forest (Best for Complexity)</option>
              <option value="logistic">Logistic Regression (Best for Simplicity)</option>
              <option value="dt">Decision Tree (Best for Rules)</option>
              
            </select>
          </div>
          
          <button onClick={handleTrain} style={btnStyle} disabled={!targetCol || loading}>
            {loading? "Training AI..": "Train Model"}
          </button>
        </div>
      )}
        
      {step === 3 && result && (
        <div>
          <button onClick={() => setStep(2)} style={{ ...btnStyle, background: "#666", marginBottom: "20px" }}>
            ← Go Back & Change Model
          </button>

          <div style={cardStyle}>
            <h2>Model Results: {selectedModel.toUpperCase()}</h2>
            <h3>Accuracy: {(result.accuracy * 100).toFixed(2)}%</h3>

            {/* FLEX CONTAINER FOR SIDE-BY-SIDE CHARTS */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "20px", marginTop: "20px" }}>
              
              {/* 1. Feature Importance Bar Chart */}
              <div style={{ flex: "1 1 500px", minWidth: "400px" }}>
                <h4>Feature Importance</h4>
                <Plot
                  data={[{
                    x: result.feature_importance.map(i => i.feature),
                    y: result.feature_importance.map(i => i.importance),
                    type: 'bar',
                    marker: { color: '#8884d8' }
                  }]}
                  layout={{clickmode:"event+select", width: undefined, height: 400, autosize: true, title: "Top Features" }}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler={true}
                />
              </div>

              
              {/* 2. 3D Scatter Plot */}
              <div style={{ flex: "1 1 500px", minWidth: "400px" }}>
                <h4>3D Data Clusters (PCA)</h4>
                
                
                <Plot
                  data={(() => {
                    const uniqueTargets = [...new Set(result.scatter_data.map(d => d.target))];
                    return uniqueTargets.map(targetVal => {
                      const group = result.scatter_data.filter(d => d.target === targetVal);
                      return {
                        x: group.map(d => d.x),
                        y: group.map(d => d.y),
                        z: group.map(d => d.z),
                        customdata: group.map(d => d.original_index), // <--- CRITICAL: Passing the ID
                        mode: 'markers',
                        type: 'scatter3d',
                        name: `Class ${targetVal}`,
                        marker: { size: 7, opacity: 0.8 ,line: { width: 1, color: '#333' }},
                        hoverinfo: 'name+text', // Show info on hover
                        text: group.map(d => `ID: ${d.original_index}`) // Text to show on hover
                      };
                    });
                  })()}
                  
                  layout={{ 
                    autosize: true,
                    height: 450,
                    title: "Data Manifold",
                    
                    scene: {
                      xaxis: { title: 'PCA 1' },
                      yaxis: { title: 'PCA 2' },
                      zaxis: { title: 'PCA 3' },
                      
                    },
                    margin: { l: 0, r: 0, b: 0, t: 30 },
                    
                  }}
                  
                                  
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler={true}
                />
              </div>

              <div style={{flex : "1 1 500px", minWidth:"400px"}}>
                  <h4>2D Scatter Plot(PCA)</h4>
                  
                  <Plot
  onInitialized={(figure, graphDiv) => {
    // This is the native Plotly graph div
    graphDiv.on('plotly_click', (eventData) => {
      console.log("✅ plotly_click event FIRED in 2D scatter!", eventData);

      if (!eventData?.points?.[0]) {
        console.log("No point data in event");
        return;
      }

      const point = eventData.points[0];
      const originalIndex = point.customdata;

      if (originalIndex === undefined || originalIndex === null) {
        console.log("No customdata found on clicked point");
        alert("Couldn't identify the clicked point");
        return;
      }

      console.log(`Clicked point index: ${originalIndex}`);

      // Trigger your explanation function
      handlePointClick({
        points: [{ customdata: originalIndex }]
      });
    });
  }}
  data={(() => {
    const uniqueTargets = [...new Set(result.scatter_data1.map(d => d.target))];
    return uniqueTargets.map(targetVal => {
      const group = result.scatter_data1.filter(d => d.target === targetVal);
      return {
        x: group.map(d => d.x),
        y: group.map(d => d.y),
        customdata: group.map(d => d.original_index),
        mode: "markers",
        type: "scatter",
        name: `Class ${targetVal}`,
        marker: {
          size: 12,                        // bigger markers = easier clicking
          opacity: 0.9,
          line: { width: 2, color: '#000' } // border helps hit detection
        },
        text: group.map(d => `ID: ${d.original_index} | Class: ${targetVal}`),
        hoverinfo: "name+text+x+y"
      };
    });
  })()}
  layout={{
    title: "2D PCA Projection – Click a point to explain",
    height: 500,
    autosize: true,
    clickmode: "event",                    // Critical: 'event' or 'event+select'
    hovermode: "closest",
    xaxis: { title: "PCA 1" },
    yaxis: { title: "PCA 2" },
    dragmode: "pan"                        // Avoid 'select' or 'lasso' if not needed
  }}
  style={{ width: "100%", height: "100%" }}
  useResizeHandler={true}
/>


              </div>
              {/* 3. SHAP Waterfall Chart (Appears on Click) */}
              <div style={{ flex: "1 1 100%", marginTop: "30px", borderTop: "2px solid #eee", paddingTop: "20px" }}>
                {explaining && <p>Thinking... (Running SHAP Analysis)</p>}
                
                {explanation && (
                  <div>
                    <h3>🔬 Why this prediction? (Local Explainability)</h3>
                    <p>Green bars pushed the prediction UP (Positive). Red bars pushed it DOWN.</p>
                    <Plot
                      data={[explanation]}
                      layout={{
                        title: "Prediction Logic (Waterfall)",
                        height: 500,
                        autosize: true,
                        xaxis: { title: "Impact on Probability" },
                        yaxis: { automargin: true }
                      }}
                      style={{ width: "100%" }}
                      useResizeHandler={true}
                    />
                  </div>
                )}
              </div>
            </div>
            
          </div>
        </div>
      )}
     </div>
  )
} 


const cardStyle = {
  background: "#fff",
  padding: "30px",
  borderRadius: "10px",
  boxShadow: "0 4px 10px rgba(0,0,0,0.1)",
  marginBottom: "20px",
  color: "black"
};

const btnStyle = {
  marginTop: "20px",
  padding: "10px 20px",
  background: "#007bff",
  color: "white",
  border: "none",
  borderRadius: "5px",
  fontSize: "16px",
  cursor: "pointer"
};


export default App