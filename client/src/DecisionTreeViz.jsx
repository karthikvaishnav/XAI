import React, { useState, useEffect } from 'react';
import Tree from 'react-d3-tree';
import axios from 'axios';

// Sci-Fi Theme Colors
const theme = {
  glass: "rgba(15, 23, 42, 0.6)",
  accent: "#22d3ee",       // Cyan
  text: "#f8fafc",         // White-ish
  nodeFill: "#0f172a",     // Dark Blue
  leafFill: "#10b981",     // Green
  link: "#22d3ee"
};

const containerStyles = {
  width: '100%',
  height: '600px',
  background: theme.glass,
  border: `1px solid rgba(34, 211, 238, 0.1)`,
  borderRadius: '8px',
  overflow: 'hidden',
  position: 'relative',
  backdropFilter: "blur(12px)"
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
        
        {/* Main Label (Split Condition) - FORCED WHITE FILL */}
        <text 
          fill="#ffffff" 
          stroke="none"
          x="22" 
          dy="-5" 
          fontSize="14px" 
          fontWeight="bold"
          style={{ 
            fontFamily: 'monospace', 
            textShadow: '2px 2px 2px #000000', // Stroke shadow for readability
            pointerEvents: 'none'
          }}
        >
          {nodeDatum.name}
        </text>

        {/* Sub Label (Gini/Samples) - FORCED CYAN FILL */}
        {nodeDatum.attributes && (
          <text 
            fill={theme.accent} 
            stroke="none"
            x="22" 
            dy="15" 
            fontSize="11px"
            style={{ 
              fontFamily: 'monospace', 
              opacity: 0.9,
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
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: theme.accent, fontFamily: 'monospace' }}>
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
          opacity: 0.6;
          fill: none;
          animation: flow 2s linear infinite;
        }
        /* Safeties to force text color if custom render fails */
        .rd3t-label__title {
          fill: #ffffff !important;
          font-family: monospace;
        }
        .rd3t-label__attributes {
          fill: ${theme.accent} !important;
          font-family: monospace;
        }
      `}</style>
    </div>
  );
}