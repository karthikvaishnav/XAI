import React, { useState, useEffect } from 'react';
import Tree from 'react-d3-tree';
import axios from 'axios';

// Ivory & Indigo Theme Colors
const theme = {
  glass: "rgba(255, 255, 255, 0.6)",
  accent: "#6366f1",       // Indigo
  accentSecondary: "#8b5cf6", // Purple
  text: "#1e293b",         
  nodeFill: "#f8fafc",     
  leafFill: "#10b981",     
  link: "#6366f1"
};

const containerStyles = {
  width: '100%',
  height: '600px',
  background: "#ffffff",
  border: `1px solid rgba(0, 0, 0, 0.05)`,
  borderRadius: '16px',
  overflow: 'hidden',
  position: 'relative',
  boxShadow: "0 10px 40px -10px rgba(0, 0, 0, 0.05)"
};



export default function DecisionTreeViz({ filePath, targetCol }) {
  const [treeData, setTreeData] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchTree = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:5000/api/decision_tree', {
        file_path: filePath,
        target_column: targetCol,
        model_type: "dt" 
      }); 
      setTreeData([res.data.tree_structure]);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  useEffect(() => {
    if (filePath && targetCol) fetchTree();
  }, [filePath, targetCol]);

  // Custom Node Rendering to control Text Color
  const renderCustomNodeElement = ({ nodeDatum, toggleNode }) => {
    const isLeaf = !nodeDatum.children || nodeDatum.children.length === 0;
    
    return (
      <g>
        {/* Node Circle */}
        <circle 
          r="15" 
          onClick={toggleNode} 
          fill={isLeaf ? theme.leafFill : theme.nodeFill} 
          stroke={theme.accent}
          strokeWidth="2"
          style={{ cursor: 'pointer', filter: `drop-shadow(0 0 5px ${theme.accent})` }}
        />
        
        {/* Main Label (Split Condition) - DARK TEXT */}
        <text 
          fill="#1e293b" 
          stroke="none"
          x="22" 
          dy="-5" 
          fontSize="14px" 
          fontWeight="600"
          style={{ 
            fontFamily: "'Outfit', sans-serif", 
            textShadow: '0 1px 2px rgba(255,255,255,0.8)', 
            pointerEvents: 'none'
          }}
        >
          {nodeDatum.name}
        </text>

        {/* Sub Label (Gini/Samples) - ACCENT SECONDARY */}
        {nodeDatum.attributes && (
          <text 
            fill={theme.accentSecondary} 
            stroke="none"
            x="22" 
            dy="15" 
            fontSize="11px"
            style={{ 
              fontFamily: "'Outfit', sans-serif", 
              opacity: 0.8,
              pointerEvents: 'none'
            }}
          >
            Gini: {nodeDatum.attributes.gini} | N: {nodeDatum.attributes.samples}
          </text>
        )}
      </g>
    );
  };


  return (
    <div style={{ marginTop: '30px', ...containerStyles }}>
      {loading && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: theme.accent, fontFamily: "'Outfit', sans-serif", fontWeight: '600' }}>
           &gt; GENERATING_LOGIC_MATRIX...
        </div>
      )}
      
      {!loading && treeData && (
        <Tree 
          data={treeData} 
          orientation="vertical"
          pathFunc="step" 
          translate={{ x: 600, y: 50 }} 
          renderCustomNodeElement={renderCustomNodeElement}
          pathClassFunc={() => 'custom-link'} 
          nodeSize={{ x: 250, y: 150 }} // Increases spacing between nodes
        />
      )}
      
      {/* GLOBAL OVERRIDES FOR D3 TREE */}
      <style>{`
        .custom-link {
          stroke: ${theme.accent} !important;
          stroke-width: 2px !important;
          opacity: 0.15;
          fill: none;
        }
        .rd3t-label__title {
          fill: #1e293b !important;
          font-family: 'Outfit', sans-serif;
        }
        .rd3t-label__attributes {
          fill: ${theme.accentSecondary} !important;
          font-family: 'Outfit', sans-serif;
        }
      `}</style>


    </div>
  );
}