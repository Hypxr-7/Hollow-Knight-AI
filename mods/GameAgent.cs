using System;
using System.IO;
using System.Diagnostics;
using System.Text;
using System.Threading;
using UnityEngine;
using Modding;

namespace GameAgent
{
    public class GameAgentMod : Mod
    {
        // Configuration
        public string pythonScriptPath = @"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\src\agent.py";
        public string modelPath = @"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\model";

        private bool isAIActive = false;
        private float inferenceTimer = 0f;
        private float inferenceInterval = 1f / 10f; // 10 FPS

        private Process pythonProcess;
        private Thread readerThread;
        private volatile bool threadRunning = false;

        private int targetWidth = 160;
        private int targetHeight = 120;

        private string lastKeysPressed = "";

        public GameAgentMod() : base("Game Agent") { }
        public override string GetVersion() => "T3.0";

        public override void Initialize()
        {
            ModHooks.HeroUpdateHook += OnHeroUpdate;
            Log("AI Inference Mod initialized! Press 'P' to toggle AI");
        }

        public void OnHeroUpdate()
        {
            // Toggle AI with P key
            if (Input.GetKeyDown(KeyCode.P))
            {
                ToggleAI();
            }

            // Run inference if active
            if (isAIActive)
            {
                inferenceTimer += Time.deltaTime;
                if (inferenceTimer >= inferenceInterval)
                {
                    RunInference();
                    inferenceTimer = 0f;
                }
            }
        }

        private void ToggleAI()
        {
            if (isAIActive)
            {
                // Stopping AI
                isAIActive = false;
                Log("AI DEACTIVATED - Releasing all keys...");
                StopPythonProcess();
            }
            else
            {
                // Starting AI
                isAIActive = true;
                Log("AI ACTIVATED");
                StartPythonProcess();
            }
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

                // Start error reader thread immediately
                pythonProcess.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        Log($"Python stderr: {e.Data}");
                    }
                };

                pythonProcess.Start();
                pythonProcess.BeginErrorReadLine();

                // Wait for READY signal with timeout
                Log("Waiting for Python READY signal...");

                bool gotReady = false;
                for (int i = 0; i < 10; i++) // Try reading up to 10 lines
                {
                    string line = pythonProcess.StandardOutput.ReadLine();
                    Log($"Python output line {i}: '{line}'");

                    if (line == "READY")
                    {
                        gotReady = true;
                        break;
                    }
                    else if (line != null && line.StartsWith("ERROR:"))
                    {
                        Log($"Python startup error: {line}");
                        break;
                    }
                    else if (line != null && line.StartsWith("FATAL:"))
                    {
                        Log($"Python fatal error: {line}");
                        break;
                    }

                    if (pythonProcess.HasExited)
                    {
                        Log($"Python process exited with code: {pythonProcess.ExitCode}");
                        break;
                    }
                }

                if (gotReady)
                {
                    Log("Python process ready!");

                    // Start reader thread
                    threadRunning = true;
                    readerThread = new Thread(ReadPythonOutput);
                    readerThread.IsBackground = true;
                    readerThread.Start();
                }
                else
                {
                    Log("Failed to get READY signal from Python");
                    StopPythonProcess();
                    isAIActive = false;
                }
            }
            catch (Exception ex)
            {
                Log($"Failed to start Python: {ex.Message}");
                isAIActive = false;
            }
        }

        private void StopPythonProcess()
        {
            try
            {
                threadRunning = false;

                if (pythonProcess != null && !pythonProcess.HasExited)
                {
                    // Send quit command to release keys
                    try
                    {
                        pythonProcess.StandardInput.WriteLine("QUIT");
                        pythonProcess.StandardInput.Flush();
                        Log("Sent QUIT command to Python");
                    }
                    catch (Exception ex)
                    {
                        Log($"Could not send QUIT: {ex.Message}");
                    }

                    // Wait for clean exit
                    if (!pythonProcess.WaitForExit(2000))
                    {
                        Log("Python didn't exit cleanly, forcing kill");
                        pythonProcess.Kill();
                    }
                    else
                    {
                        Log("Python exited cleanly");
                    }
                }

                if (readerThread != null && readerThread.IsAlive)
                {
                    readerThread.Join(1000);
                }

                Log("Python process stopped, all keys should be released");
            }
            catch (Exception ex)
            {
                Log($"Error stopping Python: {ex.Message}");
            }
            finally
            {
                pythonProcess = null;
            }
        }

        private void RunInference()
        {
            try
            {
                if (pythonProcess == null || pythonProcess.HasExited)
                {
                    Log("Python process died, restarting...");
                    StartPythonProcess();
                    return;
                }

                // Capture screenshot
                Texture2D screenshot = CaptureScreen();
                if (screenshot == null) return;

                // Convert to grayscale bytes
                byte[] grayscaleBytes = ConvertToGrayscale(screenshot);
                UnityEngine.Object.Destroy(screenshot);

                // Encode to base64
                string base64Image = Convert.ToBase64String(grayscaleBytes);

                // Send to Python
                string command = $"PREDICT:{targetWidth}:{targetHeight}:{base64Image}";
                pythonProcess.StandardInput.WriteLine(command);
                pythonProcess.StandardInput.Flush();
            }
            catch (Exception ex)
            {
                Log($"Inference error: {ex.Message}");
            }
        }

        private void ReadPythonOutput()
        {
            try
            {
                while (threadRunning && pythonProcess != null && !pythonProcess.HasExited)
                {
                    string line = pythonProcess.StandardOutput.ReadLine();
                    if (line == null) break;

                    if (line.StartsWith("KEYS:"))
                    {
                        string keys = line.Substring(5); // Remove "KEYS:"
                        lastKeysPressed = keys;

                        if (!string.IsNullOrEmpty(keys))
                        {
                            Log($"AI pressed: {keys}");
                        }
                    }
                    else if (line.StartsWith("ERROR:"))
                    {
                        Log($"Python error: {line.Substring(6)}");
                    }
                    else if (line == "QUIT_OK")
                    {
                        Log("Python acknowledged quit");
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                Log($"Reader thread error: {ex.Message}");
            }
        }

        private Texture2D CaptureScreen()
        {
            try
            {
                int screenWidth = Screen.width;
                int screenHeight = Screen.height;

                // Capture full screen
                Texture2D fullScreenshot = new Texture2D(screenWidth, screenHeight, TextureFormat.RGB24, false);
                fullScreenshot.ReadPixels(new Rect(0, 0, screenWidth, screenHeight), 0, 0);
                fullScreenshot.Apply();

                // Scale down
                RenderTexture rt = RenderTexture.GetTemporary(targetWidth, targetHeight);
                Graphics.Blit(fullScreenshot, rt);

                Texture2D scaledScreenshot = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
                RenderTexture.active = rt;
                scaledScreenshot.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
                scaledScreenshot.Apply();
                RenderTexture.active = null;

                UnityEngine.Object.Destroy(fullScreenshot);
                RenderTexture.ReleaseTemporary(rt);

                return scaledScreenshot;
            }
            catch (Exception ex)
            {
                Log($"Capture error: {ex.Message}");
                return null;
            }
        }

        private byte[] ConvertToGrayscale(Texture2D texture)
        {
            Color[] pixels = texture.GetPixels();
            byte[] grayscaleBytes = new byte[targetWidth * targetHeight];

            for (int i = 0; i < pixels.Length; i++)
            {
                float gray = pixels[i].r * 0.299f + pixels[i].g * 0.587f + pixels[i].b * 0.114f;
                grayscaleBytes[i] = (byte)(gray * 255f);
            }

            return grayscaleBytes;
        }

        public void Unload()
        {
            Log("Mod unloading - ensuring AI is stopped");
            isAIActive = false;
            StopPythonProcess();
            ModHooks.HeroUpdateHook -= OnHeroUpdate;
        }
    }
}