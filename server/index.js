const express = require("express")
const cors = require("cors")
const axios = require("axios")
const multer = require("multer")
const fs = require('fs')
const path = require('path')

const app = express()
const port = 5000

app.use(cors())
app.use(express.json())

const upload = multer({ dest: 'uploads/' })

app.listen(port, () => {
    console.log("app listening at port 5000")
})

app.get('/', (req, res) => {
    res.send('Node server working')
})

app.post('/api/uploads', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "file not found" })
        }

        const filePath = path.resolve(req.file.path)

        const pythonResponse = await axios.post('http://127.0.0.1:8000/load-data', {
            file_path: filePath
        })

        res.json({
            columns: pythonResponse.data.columns,
            filePath: filePath
        })

    } catch (error) {
        const detail = error.response?.data?.detail || error.message || "Failed to process file";
        console.error('upload error', detail);
        res.status(500).json({ error: detail })
    }
})

app.post('/api/train', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:8000/train', req.body)

        res.json(response.data)
    } catch (error) {
        const detail = error.response?.data?.detail || error.message || "Training failed";
        console.error("error training the model:", detail);
        res.status(500).json({ error: detail })
    }
})

app.post('/api/get-prediction', async (req, res) => {
    try {
        const userMessage = req.body.message

        const pythonResponse = await axios.post('http://127.0.0.1:8000/predict', {
            message: userMessage
        })

        res.json(pythonResponse.data)

    } catch (error) {
        const detail = error.response?.data?.detail || error.message || "Failed to connect to ml engine";
        console.log('error connecting to ml engine:', detail);
        res.status(500).json({ error: detail })
    }
})

app.post('/api/explain', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:8000/explain', req.body);
        res.json(response.data);

    } catch (error) {
        const detail = error.response?.data?.detail || error.message || "Explanation failed";
        console.error("error explaining:", detail);
        res.status(500).json({ error: detail })
    }
})

app.post('/api/simulate', async (req, res) => {
    try {
        const response = await axios.post(
            'http://127.0.0.1:8000/simulate',
            req.body
        )
        res.json(response.data)
    } catch (error) {
        const detail = error.response?.data?.detail || error.message || "Simulation failed";
        console.error("simulation error:", detail);
        res.status(500).json({ error: detail })
    }
})

app.post('/api/decision_tree', async (req, res) => {
    try {
        console.log("Tree route hit! Forwarding to Python...");
        const response = await axios.post('http://127.0.0.1:8000/decision_tree', req.body);
        res.json(response.data);
    } catch (error) {
        const detail = error.response?.data?.detail || error.message || "Tree generation failed";
        console.error("Tree error:", detail);
        res.status(500).json({ error: detail });
    }
});

app.post('/api/predict_manual', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:8000/predict_manual', req.body);
        res.json(response.data);
    } catch (error) {
        const detail = error.response?.data?.detail || error.message || "Prediction failed";
        console.error("predict_manual error:", detail);
        res.status(500).json({ error: detail });
    }
});

app.post('/api/decision_boundary', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:8000/decision_boundary', req.body);
        res.json(response.data);
    } catch (error) {
        const detail = error.response?.data?.detail || error.message || "Decision boundary failed";
        console.error("Boundary error:", detail);
        res.status(500).json({ error: detail });
    }
});

app.post('/api/compare', async (req, res) => {
    try {
        const response = await axios.post('http://127.0.0.1:8000/compare', req.body);
        res.json(response.data);
    } catch (error) {
        const detail = error.response?.data?.detail || error.message || "Comparison failed";
        console.error("Compare error:", detail);
        res.status(500).json({ error: detail });
    }
});