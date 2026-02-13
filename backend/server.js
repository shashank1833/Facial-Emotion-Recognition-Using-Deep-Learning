const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const port = 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));

// Path to Python bridge
const pythonPath = 'python';
const scriptPath = path.join(__dirname, 'inference_bridge.py');

// Initialize Python process
let pythonProcess = null;

function startPythonBridge() {
    console.log('Starting Python bridge...');
    pythonProcess = spawn(pythonPath, [scriptPath], {
        cwd: __dirname,
        env: { ...process.env, PYTHONPATH: path.join(__dirname, '..', 'src') }
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}. Restarting...`);
        setTimeout(startPythonBridge, 2000);
    });
}

startPythonBridge();

// Queue to handle multiple requests sequentially to the single Python process
let requestQueue = [];
let isProcessing = false;

async function processQueue() {
    if (isProcessing || requestQueue.length === 0) return;
    
    isProcessing = true;
    const { reqData, res } = requestQueue.shift();

    try {
        const responsePromise = new Promise((resolve, reject) => {
            const onData = (data) => {
                pythonProcess.stdout.removeListener('data', onData);
                resolve(data.toString());
            };
            pythonProcess.stdout.on('data', onData);
            
            // Timeout if Python takes too long (increased to 30s for slow model loading)
            setTimeout(() => {
                pythonProcess.stdout.removeListener('data', onData);
                reject(new Error('Inference timeout'));
            }, 30000);
        });

        pythonProcess.stdin.write(JSON.stringify(reqData) + '\n');
        
        const result = await responsePromise;
        const jsonResult = JSON.parse(result);
        
        if (jsonResult.error) {
            res.status(400).json(jsonResult);
        } else {
            res.json(jsonResult);
        }
    } catch (err) {
        console.error('Queue error:', err);
        res.status(500).json({ error: err.message });
    } finally {
        isProcessing = false;
        processQueue();
    }
}

app.post('/predict', (req, res) => {
    if (!req.body.image) {
        return res.status(400).json({ error: 'No image data provided' });
    }

    requestQueue.push({ reqData: req.body, res });
    processQueue();
});

app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        backend: 'nodejs',
        python_bridge: pythonProcess ? 'active' : 'inactive'
    });
});

app.listen(port, () => {
    console.log(`Node.js backend listening at http://localhost:${port}`);
});
