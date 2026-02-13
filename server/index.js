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

const upload = multer({dest : 'uploads/'})

app.listen(port, ()=>{
    console.log("app listening at port 5000")
})

app.get('/',(req,res)=>{
    res.send('Node server working')
})

app.post('/api/uploads',upload.single('file'),async (req,res)=>{
    try{
        if(!req.file){
            return res.status(400).json({error:"file not found"})
        }

        const filePath = path.resolve(req.file.path)

        const pythonResponse = await axios.post('http://127.0.0.1:8000/load-data',{
            file_path :filePath
        })

        res.json({
            columns : pythonResponse.data.columns,
            filePath: filePath
        })

    }catch(error){
        console.error('upload error', error.message)
        res.status(500).json({error : "Failed to process file"})
    }
})

app.post('/api/train',async(req,res)=>{
    try{
        const response = await axios.post('http://127.0.0.1:8000/train',req.body)

        res.json(response.data)
    }catch(error){
        console.error("error training the model: ",error.message)
        res.status(500).json({error:"training failed"})
    }
})

app.post('/api/get-prediction',async (req,res)=>{
    try{
        const userMessage = req.body.message

        const pythonResponse = await axios.post('http://127.0.0.1:8000/predict',{
            message : userMessage
        })
        
        res.json(pythonResponse.data)

    }catch(error){
        console.log('error connecting to ml engine')
        res.status(500).json({error:"Failed to connect to ml engine"})
    }
})

app.post('/api/explain', async(req,res)=>{
    try{
        const response = await axios.post('http://127.0.0.1:8000/explain',req.body);
        res.json(response.data);

    }catch(error){
        console.error("error explaining:",error.message)
        res.status(500).json({error: "Explaination failed"})
    }
})
