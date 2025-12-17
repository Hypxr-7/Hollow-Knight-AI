using System;
using System.IO;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Collections.Concurrent;
using UnityEngine;
using Modding;

namespace GameAgent
{
    public class GameAgentMod : Mod, ITogglableMod
    {
        // Configuration
        public string pythonScriptPath = @"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\src\agent.py";
        public string modelPath = @"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\model";

        private bool isAIActive = false;
        private volatile bool waitingForPrediction = false;

        private Process pythonProcess;
        private Thread readerThread;
        private Thread processingThread;
        private volatile bool threadRunning = false;

        // CRITICAL FIX: Match data collection resolution!
        private int targetWidth = 640;
        private int targetHeight = 360;

        // Reusable texture to avoid GC allocs
        private Texture2D captureTexture;

        // Data container
        private struct FrameData
        {
            public byte[] pixels;
            public float playerX;
            public float playerY;
            public float enemyX;
            public float enemyY;
        }

        // Queue for offloading processing
        private ConcurrentQueue<FrameData> frameQueue = new ConcurrentQueue<FrameData>();

        public GameAgentMod() : base("Game Agent") { }
        public override string GetVersion() => "T3.2-Resolution-Fixed";

        public override void Initialize()
        {
            ModHooks.HeroUpdateHook -= OnHeroUpdate;
            ModHooks.HeroUpdateHook += OnHeroUpdate;
            Log("AI Inference Mod initialized! Press 'P' to toggle AI");
        }

        public void OnHeroUpdate()
        {
            if (Input.GetKeyDown(KeyCode.P))
            {
                ToggleAI();
            }

            // DYNAMIC SYNC: Only capture if Python has finished the previous frame
            if (isAIActive && !waitingForPrediction)
            {
                CaptureAndEnqueue();
                waitingForPrediction = true; // Block until Python replies
            }
        }

        private void ToggleAI()
        {
            if (isAIActive)
            {
                isAIActive = false;
                waitingForPrediction = false;
                Log("AI DEACTIVATED");
                StopPythonProcess();
                CleanupTextures();
            }
            else
            {
                isAIActive = true;
                waitingForPrediction = false;
                Log("AI ACTIVATED");
                StartPythonProcess();
                InitializeTextures();
            }
        }

        private void InitializeTextures()
        {
            if (captureTexture == null)
            {
                captureTexture = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
            }
        }

        private void CleanupTextures()
        {
            if (captureTexture != null) { UnityEngine.Object.Destroy(captureTexture); captureTexture = null; }
        }

        private void StartPythonProcess()
        {
            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo()
                {
                    FileName = @"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\.venv\Scripts\python.exe",
                    Arguments = $"\"{pythonScriptPath}\" \"{modelPath}\"",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    RedirectStandardInput = true,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8
                };

                pythonProcess = new Process { StartInfo = startInfo };
                pythonProcess.ErrorDataReceived += (sender, e) => { if (!string.IsNullOrEmpty(e.Data)) Log($"Python stderr: {e.Data}"); };
                pythonProcess.Start();
                pythonProcess.BeginErrorReadLine();

                Log("Waiting for Python READY...");
                bool gotReady = false;
                for (int i = 0; i < 20; i++)
                {
                    string line = pythonProcess.StandardOutput.ReadLine();
                    if (line == "READY") { gotReady = true; break; }
                    if (pythonProcess.HasExited) break;
                }

                if (gotReady)
                {
                    Log("Python ready!");
                    threadRunning = true;
                    
                    // Reader Thread (reads prediction keys)
                    readerThread = new Thread(ReadPythonOutput) { IsBackground = true };
                    readerThread.Start();

                    // Processing Thread (processes images and sends to Python)
                    processingThread = new Thread(ProcessFramesLoop) { IsBackground = true };
                    processingThread.Start();
                }
                else
                {
                    Log("Failed to start Python.");
                    StopPythonProcess();
                    isAIActive = false;
                }
            }
            catch (Exception ex)
            {
                Log($"Start Error: {ex.Message}");
                isAIActive = false;
            }
        }

        private void StopPythonProcess()
        {
            threadRunning = false;
            if (pythonProcess != null && !pythonProcess.HasExited)
            {
                try 
                { 
                    byte[] quitBytes = Encoding.UTF8.GetBytes("QUIT\n");
                    pythonProcess.StandardInput.BaseStream.Write(quitBytes, 0, quitBytes.Length);
                    pythonProcess.StandardInput.BaseStream.Flush();
                } 
                catch {}
                pythonProcess.WaitForExit(1000);
                if (!pythonProcess.HasExited) pythonProcess.Kill();
            }
            frameQueue = new ConcurrentQueue<FrameData>(); // Clear queue
        }

        // --- Main Thread: Capture Only ---
        private void CaptureAndEnqueue()
        {
            try
            {
                // MATCH DATA COLLECTION METHOD:
                // 1. Capture full screen directly
                int screenWidth = Screen.width;
                int screenHeight = Screen.height;
                
                Texture2D fullScreenshot = new Texture2D(screenWidth, screenHeight, TextureFormat.RGB24, false);
                fullScreenshot.ReadPixels(new Rect(0, 0, screenWidth, screenHeight), 0, 0);
                fullScreenshot.Apply();

                // 2. Scale using RenderTexture (same as data collection)
                RenderTexture renderTexture = RenderTexture.GetTemporary(targetWidth, targetHeight);
                Graphics.Blit(fullScreenshot, renderTexture);

                captureTexture.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
                captureTexture.Apply();
                
                RenderTexture.active = null;
                RenderTexture.ReleaseTemporary(renderTexture);
                UnityEngine.Object.Destroy(fullScreenshot);

                // 3. Get raw RGB24 data (3 bytes per pixel)
                byte[] rawRgb = captureTexture.GetRawTextureData(); 
                
                // 4. Get position data
                Vector3 playerPos = HeroController.instance != null ? HeroController.instance.transform.position : Vector3.zero;
                GameObject enemy = FindCurrentEnemy();
                Vector3 enemyPos = enemy != null ? enemy.transform.position : Vector3.zero;

                FrameData frame = new FrameData
                {
                    pixels = rawRgb,
                    playerX = playerPos.x,
                    playerY = playerPos.y,
                    enemyX = enemyPos.x,
                    enemyY = enemyPos.y
                };

                // 5. Enqueue for background processing
                frameQueue.Enqueue(frame);
            }
            catch (Exception ex)
            {
                Log($"Capture Error: {ex.Message}");
                waitingForPrediction = false; // Reset on error to prevent deadlock
            }
        }

        // --- Background Thread: Process & Send ---
        private void ProcessFramesLoop()
        {
            FrameData frame;
            // No pre-allocated buffer needed for raw pass-through if we just send frame.pixels
            bool _debugLogged = false;

            while (threadRunning)
            {
                if (frameQueue.TryDequeue(out frame))
                {
                    try
                    {
                        // Pass raw RGB24 data directly to Python
                        // This ensures consistency with the training pipeline (DataCollector -> RGB -> Train -> Grayscale)
                        byte[] dataToSend = frame.pixels;
                        
                        // DEBUG: Log checksum once
                        if (!_debugLogged) {
                            int checkSum = 0;
                            for (int i = 0; i < Math.Min(100, dataToSend.Length); i++) {
                                checkSum += dataToSend[i];
                            }
                            Log($"DEBUG: RGB checksum (first 100 bytes): {checkSum}");
                            
                            // Also log some sample values
                            string sample = $"Sample pixels: [{dataToSend[0]}, {dataToSend[1]}, {dataToSend[2]}, {dataToSend[10]}, {dataToSend[100]}]";
                            Log($"DEBUG: {sample}");
                            _debugLogged = true;
                        }
                        
                        if (pythonProcess != null && !pythonProcess.HasExited)
                        {
                            // Protocol: Send Header -> Flush -> Send Raw Bytes -> Flush
                            string pX = frame.playerX.ToString(System.Globalization.CultureInfo.InvariantCulture);
                            string pY = frame.playerY.ToString(System.Globalization.CultureInfo.InvariantCulture);
                            string eX = frame.enemyX.ToString(System.Globalization.CultureInfo.InvariantCulture);
                            string eY = frame.enemyY.ToString(System.Globalization.CultureInfo.InvariantCulture);

                            // Header now implies RGB data (width * height * 3 bytes)
                            string header = $"PREDICT_RAW:{targetWidth}:{targetHeight}:{pX}:{pY}:{eX}:{eY}\n";
                            byte[] headerBytes = Encoding.UTF8.GetBytes(header);
                            
                            pythonProcess.StandardInput.BaseStream.Write(headerBytes, 0, headerBytes.Length);
                            pythonProcess.StandardInput.BaseStream.Flush();
                            
                            pythonProcess.StandardInput.BaseStream.Write(dataToSend, 0, dataToSend.Length);
                            pythonProcess.StandardInput.BaseStream.Flush();
                        }
                    }
                    catch (Exception ex)
                    {
                        Log($"Processing Error: {ex.Message}");
                    }
                }
                else
                {
                    Thread.Sleep(1); // Prevent CPU spin
                }
            }
        }

        private void ReadPythonOutput()
        {
            while (threadRunning && pythonProcess != null && !pythonProcess.HasExited)
            {
                try
                {
                    string line = pythonProcess.StandardOutput.ReadLine();
                    if (line == null) break;
                    if (line.StartsWith("KEYS:")) 
                    { 
                        Log(line); 
                        waitingForPrediction = false; // Python is ready for more!
                    } 
                    else if (line.StartsWith("ERROR:")) Log($"PyErr: {line}");
                } catch { break; }
            }
        }

        // --- Helper Methods for Enemy Detection ---
        private GameObject FindCurrentEnemy()
        {
            try
            {
                var healthManagers = GameObject.FindObjectsOfType<HealthManager>();
                if (healthManagers.Length == 0) return null;

                return FindClosestEnemyToPlayer(healthManagers);
            }
            catch
            {
                return null;
            }
        }

        private GameObject FindClosestEnemyToPlayer(HealthManager[] healthManagers)
        {
            var hero = HeroController.instance;
            Vector3 heroPos = hero?.transform.position ?? Vector3.zero;

            HealthManager closest = null;
            float minDistance = float.MaxValue;

            foreach (var healthManager in healthManagers)
            {
                float distance = Vector3.Distance(heroPos, healthManager.transform.position);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    closest = healthManager;
                }
            }

            return closest?.gameObject;
        }

        public void Unload()
        {
            isAIActive = false;
            StopPythonProcess(); 
            CleanupTextures();
            ModHooks.HeroUpdateHook -= OnHeroUpdate;
        }
    }
}