using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using Modding;
using UnityEngine;

namespace DataCollector
{
    public class DataCollectorMod : Mod
    {
        #region Configuration
        // Configurable save directory
        public string saveDirectory = @"C:\Users\muusm\Documents\ML_project\Hollow-Knight-AI\HKData";

        // Performance settings
        private readonly int targetWidth = 640;
        private readonly int targetHeight = 360;

        // Recording settings - Target 30 FPS
        private readonly float csvRecordingInterval = 1f / 30f; 
        private readonly float deadzoneThreshold = 0.5f;
        #endregion

        #region Recording State
        private bool isRecording = false;
        private float recordingTimer = 0f;
        private int frameCount = 0;
        private string sessionTimestamp;
        #endregion

        #region File Paths
        private string csvFilePath;
        private string framesDirectoryPath;
        #endregion

        #region Threading & Data
        private ConcurrentQueue<FrameData> writeQueue = new ConcurrentQueue<FrameData>();
        private Thread writeThread;
        private bool threadRunning = false;
        private readonly object fileLock = new object();

        private struct FrameData
        {
            public int FrameId;
            public byte[] RawData;
            public string CsvRow;
            public int Width;
            public int Height;
        }
        #endregion

        #region Reusable Resources
        private Texture2D screenTexture;
        private Texture2D resizedTexture;
        private RenderTexture renderTexture;
        #endregion

        #region Constructor and Initialization
        public DataCollectorMod() : base("Data Collector") { }

        public override string GetVersion() => "4.0";

        public override void Initialize()
        {
            ModHooks.HeroUpdateHook += OnHeroUpdate;
            
            // Initialize reusable textures once
            // Note: Screen size might change, ideally check this, but for now assume fixed
            InitializeTextures(Screen.width, Screen.height);

            try
            {
                CreateMainSaveDirectory();
                Log("Behavioral Data Collector initialized successfully!");
            }
            catch (Exception ex)
            {
                Log($"Error initializing mod: {ex.Message}");
            }
        }

        private void InitializeTextures(int screenWidth, int screenHeight)
        {
            if (screenTexture != null) UnityEngine.Object.Destroy(screenTexture);
            if (resizedTexture != null) UnityEngine.Object.Destroy(resizedTexture);
            if (renderTexture != null) renderTexture.Release();

            screenTexture = new Texture2D(screenWidth, screenHeight, TextureFormat.RGB24, false);
            resizedTexture = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
            renderTexture = new RenderTexture(targetWidth, targetHeight, 24);
        }

        private void CreateMainSaveDirectory()
        {
            if (!Directory.Exists(saveDirectory))
            {
                Directory.CreateDirectory(saveDirectory);
                Log($"Created main save directory: {saveDirectory}");
            }
        }
        #endregion

        #region Main Update Loop
        public void OnHeroUpdate()
        {
            HandleRecordingToggle();

            if (isRecording)
            {
                UpdateRecordingTimer();
            }
        }

        private void HandleRecordingToggle()
        {
            if (Input.GetKeyDown(KeyCode.O))
            {
                if (isRecording)
                {
                    StopCurrentSession();
                }
                else
                {
                    StartNewSession();
                }
            }
        }

        private void UpdateRecordingTimer()
        {
            recordingTimer += Time.deltaTime;

            if (recordingTimer >= csvRecordingInterval)
            {
                CaptureFrame();
                recordingTimer = 0f;
            }
        }
        #endregion

        #region Session Management
        private void StartNewSession()
        {
            InitializeSessionPaths();
            CreateSessionDirectories();
            WriteCSVHeader();
            ResetSessionState();

            // Start Worker Thread
            threadRunning = true;
            writeThread = new Thread(WriteLoop);
            writeThread.IsBackground = true;
            writeThread.Start();

            isRecording = true;
            Log("Recording session started! (Threaded)");
        }

        private void InitializeSessionPaths()
        {
            sessionTimestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            csvFilePath = Path.Combine(saveDirectory, $"hk_actions_{sessionTimestamp}.csv");
            framesDirectoryPath = Path.Combine(saveDirectory, $"frames_{sessionTimestamp}");
        }

        private void CreateSessionDirectories()
        {
            if (!Directory.Exists(framesDirectoryPath))
            {
                Directory.CreateDirectory(framesDirectoryPath);
                Log($"Created frames directory: {framesDirectoryPath}");
            }
        }

        private void ResetSessionState()
        {
            frameCount = 0;
            // Clear queue logic if needed, but new session new queue usually fine
            while(writeQueue.TryDequeue(out _)); 
        }

        private void StopCurrentSession()
        {
            isRecording = false;
            
            // Allow thread to finish buffer
            // In a real scenario, we might want to wait or signal end
            // For now, we let the thread run until queue empty then stop
            
            // We'll set a flag or just let it finish in background?
            // Safer to let it run. But we need to know when to stop.
            // Simplified: We stop adding, thread keeps processing until empty.
            
            Log("Stopping recording... saving remaining frames.");
            
            // We can leave the thread running to drain, or join. 
            // Since it's a game, we don't want to freeze. 
            // We will signal the thread to stop when empty.
            
            // The WriteLoop checks 'threadRunning' and 'queue.Count'.
            // We set threadRunning = false, but WriteLoop will continue until queue is empty.
            threadRunning = false; 
            
            LogSessionSummary();
        }

        private void LogSessionSummary()
        {
            Log("Recording session stopped!");
            Log($"CSV: {Path.GetFileName(csvFilePath)}");
            Log($"Frames: {Path.GetFileName(framesDirectoryPath)}");
        }
        #endregion

        #region Data Capture
        private void CaptureFrame()
        {
            try
            {
                // 1. Capture Game Data
                var playerData = GetPlayerData();
                if (playerData == null) return;
                var inputData = GetInputData();
                var enemyData = GetEnemyData();
                string csvRow = FormatDataRow(playerData, inputData, enemyData);

                // 2. Capture Screen Data (Main Thread)
                // Check if screen resolution changed
                if (screenTexture.width != Screen.width || screenTexture.height != Screen.height)
                {
                    InitializeTextures(Screen.width, Screen.height);
                }

                // Read full screen
                screenTexture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
                screenTexture.Apply();

                // Downscale using GPU (Blit)
                Graphics.Blit(screenTexture, renderTexture);
                
                // Read back small texture
                RenderTexture.active = renderTexture;
                resizedTexture.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
                resizedTexture.Apply();
                RenderTexture.active = null;

                // Get raw bytes - FAST, just a copy
                byte[] rawBytes = resizedTexture.GetRawTextureData();

                // 3. Enqueue
                writeQueue.Enqueue(new FrameData 
                { 
                    FrameId = frameCount, 
                    RawData = rawBytes, 
                    CsvRow = csvRow,
                    Width = targetWidth,
                    Height = targetHeight
                });

                frameCount++;
            }
            catch (Exception ex)
            {
                Log($"Error capturing frame: {ex.Message}");
            }
        }
        
        // ... (Keep existing Data Helper methods: GetPlayerData, GetInputData, etc.) ...
        
        private PlayerDataSnapshot GetPlayerData()
        {
            var heroController = HeroController.instance;
            if (heroController == null) return null;

            var position = heroController.transform.position;
            var health = PlayerData.instance.health;

            return new PlayerDataSnapshot
            {
                Position = position,
                Health = health
            };
        }

        private InputDataSnapshot GetInputData()
        {
            var rawInput = GetRawInputAxes();
            var processedInput = ApplyDeadzoneCorrection(rawInput);
            var actionStates = GetActionStates(processedInput);

            return new InputDataSnapshot
            {
                MovingLeft = actionStates.MovingLeft,
                MovingRight = actionStates.MovingRight,
                MovingUp = actionStates.MovingUp,
                MovingDown = actionStates.MovingDown,
                Attacking = actionStates.Attacking,
                Jumping = actionStates.Jumping,
                Dashing = actionStates.Dashing,
                Casting = actionStates.Casting
            };
        }

        private RawInputAxes GetRawInputAxes()
        {
            return new RawInputAxes
            {
                Horizontal = Input.GetAxis("Horizontal"),
                Vertical = Input.GetAxis("Vertical")
            };
        }

        private ProcessedInputAxes ApplyDeadzoneCorrection(RawInputAxes raw)
        {
            return new ProcessedInputAxes
            {
                Horizontal = Mathf.Abs(raw.Horizontal) > deadzoneThreshold ? raw.Horizontal : 0f,
                Vertical = Mathf.Abs(raw.Vertical) > deadzoneThreshold ? raw.Vertical : 0f
            };
        }

        private ActionStates GetActionStates(ProcessedInputAxes input)
        {
            var heroController = HeroController.instance;

            return new ActionStates
            {
                MovingLeft = input.Horizontal < -deadzoneThreshold || Input.GetKey(KeyCode.Joystick1Button14),
                MovingRight = input.Horizontal > deadzoneThreshold || Input.GetKey(KeyCode.Joystick1Button15),
                MovingUp = input.Vertical > deadzoneThreshold,
                MovingDown = input.Vertical < -deadzoneThreshold,
                Attacking = heroController.cState.attacking,
                Jumping = heroController.cState.jumping,
                Dashing = heroController.cState.dashing,
                Casting = heroController.cState.focusing || heroController.cState.casting || heroController.cState.spellQuake
            };
        }

        private EnemyDataSnapshot GetEnemyData()
        {
            var enemy = FindCurrentEnemy();
            var position = enemy?.transform.position ?? Vector3.zero;

            return new EnemyDataSnapshot
            {
                Position = position,
                Enemy = enemy
            };
        }
        
        private GameObject FindCurrentEnemy()
        {
            try
            {
                var healthManagers = GameObject.FindObjectsOfType<HealthManager>();
                if (healthManagers.Length == 0) return null;
                return FindClosestEnemyToPlayer(healthManagers);
            }
            catch { return null; }
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

        private string FormatDataRow(PlayerDataSnapshot player, InputDataSnapshot input, EnemyDataSnapshot enemy)
        {
            return $"{frameCount},{player.Position.x:F3},{player.Position.y:F3},{player.Health}," +
                   $"{enemy.Position.x:F3},{enemy.Position.y:F3}," +
                   $"{input.MovingLeft},{input.MovingRight},{input.MovingUp},{input.MovingDown}," +
                   $"{input.Attacking},{input.Jumping},{input.Dashing},{input.Casting}";
        }
        #endregion

        #region Threaded Worker
        private void WriteLoop()
        {
            while (threadRunning || !writeQueue.IsEmpty)
            {
                if (writeQueue.TryDequeue(out FrameData frame))
                {
                    try
                    {
                        // Save Raw Bytes
                        string framePath = Path.Combine(framesDirectoryPath, $"frame_{frame.FrameId:D6}.raw");
                        File.WriteAllBytes(framePath, frame.RawData);

                        // Append CSV Row
                        // We use a lock just in case, though this is the only writer thread
                        lock (fileLock)
                        {
                            File.AppendAllText(csvFilePath, frame.CsvRow + "\n");
                        }
                    }
                    catch
                    {
                        // Background thread error - silent to prevent crash
                    }
                }
                else
                {
                    Thread.Sleep(5); // Sleep to save CPU when empty
                }
            }
        }
        #endregion

        #region File Operations
        private void WriteCSVHeader()
        {
            const string header = "frame_id,x_position,y_position,health,enemy_x,enemy_y," +
                                "moving_left,moving_right,moving_up,moving_down," +
                                "attacking,jumping,dashing,casting";
            File.WriteAllText(csvFilePath, header + "\n");
        }
        #endregion

        #region Data Transfer Objects
        private class PlayerDataSnapshot
        {
            public Vector3 Position { get; set; }
            public int Health { get; set; }
        }

        private class InputDataSnapshot
        {
            public bool MovingLeft { get; set; }
            public bool MovingRight { get; set; }
            public bool MovingUp { get; set; }
            public bool MovingDown { get; set; }
            public bool Attacking { get; set; }
            public bool Jumping { get; set; }
            public bool Dashing { get; set; }
            public bool Casting { get; set; }
        }

        private class EnemyDataSnapshot
        {
            public Vector3 Position { get; set; }
            public GameObject Enemy { get; set; }
        }

        private class RawInputAxes
        {
            public float Horizontal { get; set; }
            public float Vertical { get; set; }
        }

        private class ProcessedInputAxes
        {
            public float Horizontal { get; set; }
            public float Vertical { get; set; }
        }

        private class ActionStates
        {
            public bool MovingLeft { get; set; }
            public bool MovingRight { get; set; }
            public bool MovingUp { get; set; }
            public bool MovingDown { get; set; }
            public bool Attacking { get; set; }
            public bool Jumping { get; set; }
            public bool Dashing { get; set; }
            public bool Casting { get; set; }
        }

        private class ScaleDimensions
        {
            public int Width { get; set; }
            public int Height { get; set; }
        }
        #endregion
    }
}