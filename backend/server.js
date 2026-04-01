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
const pythonPath = process.platform === 'win32' ? 'python' : 'python3';
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
        const msg = data.toString();
        // Ignore MediaPipe proto output to keep logs clean
        if (msg.includes('node {') || msg.includes('calculator:') || msg.includes('input_stream:')) return;
        console.error(`Python Error: ${msg}`);
    });

    pythonProcess.stdout.on('data', (data) => {
        const msg = data.toString();
        console.log(`Python Output: ${msg}`);
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
            let buffer = '';
            const onData = (data) => {
                const str = data.toString();
                buffer += str;
                
                // Try to find a complete JSON object in the buffer
                const lines = buffer.split('\n');
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i].trim();
                    if (line.startsWith('{') && line.endsWith('}')) {
                        try {
                            const parsed = JSON.parse(line);
                            // If it has emotion or error, it's our result
                            if (parsed.emotion || parsed.error) {
                                pythonProcess.stdout.removeListener('data', onData);
                                resolve(line);
                                return;
                            }
                        } catch (e) {
                            // Not a valid JSON or not our JSON, ignore
                        }
                    }
                }
                // Keep the last partial line in the buffer
                buffer = lines[lines.length - 1];
            };
            pythonProcess.stdout.on('data', onData);
            
            // Timeout if Python takes too long
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

app.get('/', (req, res) => {
    res.send('Aura AI Backend is running. Use /health for status or /predict for inference.');
});

app.listen(port, () => {
    console.log(`Node.js backend listening at http://localhost:${port}`);
});
