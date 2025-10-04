using System;
using System.Collections.Generic;
using System.IO;
using Modding;
using UnityEngine;

namespace DataCollector
{
    public class DataCollectorMod : Mod
    {
        #region Configuration
        // Configurable save directory
        public string saveDirectory = @"C:\Users\Abdullah\Downloads\HKData";

        // Performance settings
        private readonly int targetWidth = 640;
        private readonly int targetHeight = 360;

        // Recording settings
        private readonly float csvRecordingInterval = 1f / 3f; // 3 FPS
        private readonly float deadzoneThreshold = 0.5f;
        private readonly int bufferSaveThreshold = 100;
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

        #region Data Management
        private readonly List<string> dataBuffer = new List<string>();
        #endregion

        #region Constructor and Initialization
        public DataCollectorMod() : base("Data Collector") { }

        public override string GetVersion() => "v3.0";

        public override void Initialize()
        {
            ModHooks.HeroUpdateHook += OnHeroUpdate;

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
                CapturePlayerData();
                CaptureScreenFrame();
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

            isRecording = true;
            Log("Recording session started!");
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
            dataBuffer.Clear();
        }

        private void StopCurrentSession()
        {
            isRecording = false;
            SaveRemainingBufferedData();
            LogSessionSummary();
        }

        private void SaveRemainingBufferedData()
        {
            if (dataBuffer.Count > 0)
            {
                SaveBufferedData();
            }
        }

        private void LogSessionSummary()
        {
            Log("Recording session stopped!");
            Log($"CSV: {Path.GetFileName(csvFilePath)}");
            Log($"Frames: {Path.GetFileName(framesDirectoryPath)}");
        }
        #endregion

        #region Data Capture
        private void CapturePlayerData()
        {
            try
            {
                var playerData = GetPlayerData();
                if (playerData == null) return;

                var inputData = GetInputData();
                var enemyData = GetEnemyData();

                string dataRow = FormatDataRow(playerData, inputData, enemyData);

                dataBuffer.Add(dataRow);
                frameCount++;

                SaveBufferIfNeeded();
            }
            catch (Exception ex)
            {
                Log($"Error capturing data: {ex.Message}");
            }
        }

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

            LogDeadzoneCorrection(rawInput, frameCount);

            return new InputDataSnapshot
            {
                MovingLeft = actionStates.MovingLeft,
                MovingRight = actionStates.MovingRight,
                Attacking = actionStates.Attacking,
                Jumping = actionStates.Jumping,
                Dashing = actionStates.Dashing
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
                Attacking = heroController.cState.attacking,
                Jumping = heroController.cState.jumping,
                Dashing = heroController.cState.dashing
            };
        }

        private void LogDeadzoneCorrection(RawInputAxes raw, int currentFrame)
        {
            if (currentFrame % 30 == 0)
            {
                Log($"Deadzone applied - Raw H:{raw.Horizontal:F3} | Raw V:{raw.Vertical:F3}");
            }
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

        private string FormatDataRow(PlayerDataSnapshot player, InputDataSnapshot input, EnemyDataSnapshot enemy)
        {
            return $"{frameCount},{player.Position.x:F3},{player.Position.y:F3}," +
                   $"{enemy.Position.x:F3},{enemy.Position.y:F3}," +
                   $"{input.MovingLeft},{input.MovingRight}," +
                   $"{input.Attacking},{input.Jumping},{input.Dashing}";
        }

        private void SaveBufferIfNeeded()
        {
            if (dataBuffer.Count >= bufferSaveThreshold)
            {
                SaveBufferedData();
            }
        }
        #endregion

        #region Enemy Detection
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
        #endregion

        #region Screen Capture
        private void CaptureScreenFrame()
        {
            try
            {
                EnsureFramesDirectoryExists();

                var screenshot = CaptureAndScaleScreen();
                if (screenshot != null)
                {
                    SaveScreenshotFrame(screenshot);
                    UnityEngine.Object.Destroy(screenshot);
                }
            }
            catch (Exception ex)
            {
                HandleScreenCaptureError(ex);
            }
        }

        private void EnsureFramesDirectoryExists()
        {
            if (!Directory.Exists(framesDirectoryPath))
            {
                Directory.CreateDirectory(framesDirectoryPath);
                Log($"Created frames directory: {framesDirectoryPath}");
            }
        }

        private Texture2D CaptureAndScaleScreen()
        {
            var fullScreenshot = CaptureFullScreen();
            var scaledScreenshot = CreateScaledScreenshot(fullScreenshot);

            UnityEngine.Object.Destroy(fullScreenshot);
            return scaledScreenshot;
        }

        private Texture2D CaptureFullScreen()
        {
            int screenWidth = Screen.width;
            int screenHeight = Screen.height;

            var screenshot = new Texture2D(screenWidth, screenHeight, TextureFormat.RGB24, false);
            screenshot.ReadPixels(new Rect(0, 0, screenWidth, screenHeight), 0, 0);
            screenshot.Apply();

            return screenshot;
        }

        private Texture2D CreateScaledScreenshot(Texture2D fullScreenshot)
        {
            var scaleDimensions = CalculateScaleDimensions();

            var renderTexture = RenderTexture.GetTemporary(scaleDimensions.Width, scaleDimensions.Height);
            Graphics.Blit(fullScreenshot, renderTexture);

            var scaledScreenshot = new Texture2D(scaleDimensions.Width, scaleDimensions.Height, TextureFormat.RGB24, false);
            RenderTexture.active = renderTexture;
            scaledScreenshot.ReadPixels(new Rect(0, 0, scaleDimensions.Width, scaleDimensions.Height), 0, 0);
            scaledScreenshot.Apply();
            RenderTexture.active = null;

            RenderTexture.ReleaseTemporary(renderTexture);
            return scaledScreenshot;
        }

        private ScaleDimensions CalculateScaleDimensions()
        {
            int screenWidth = Screen.width;
            int screenHeight = Screen.height;

            float scale = Mathf.Min((float)targetWidth / screenWidth, (float)targetHeight / screenHeight);

            return new ScaleDimensions
            {
                Width = Mathf.RoundToInt(screenWidth * scale),
                Height = Mathf.RoundToInt(screenHeight * scale)
            };
        }

        private void SaveScreenshotFrame(Texture2D screenshot)
        {
            byte[] frameData = screenshot.EncodeToPNG();
            string framePath = Path.Combine(framesDirectoryPath, $"frame_{frameCount:D6}.png");

            EnsureDirectoryExists(Path.GetDirectoryName(framePath));
            File.WriteAllBytes(framePath, frameData);
        }

        private void EnsureDirectoryExists(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }
        }

        private void HandleScreenCaptureError(Exception ex)
        {
            Log($"Error capturing frame: {ex.Message}");
            Log($"Attempted path: {framesDirectoryPath}");

            TryRecreateFramesDirectory();
        }

        private void TryRecreateFramesDirectory()
        {
            try
            {
                Directory.CreateDirectory(framesDirectoryPath);
                Log("Recreated frames directory");
            }
            catch (Exception dirEx)
            {
                Log($"Failed to create directory: {dirEx.Message}");
            }
        }
        #endregion

        #region File Operations
        private void WriteCSVHeader()
        {
            const string header = "frame_id,x_position,y_position,enemy_x,enemy_y," +
                                "moving_left,moving_right," +
                                "attacking,jumping,dashing";
            File.WriteAllText(csvFilePath, header + "\n");
        }

        private void SaveBufferedData()
        {
            try
            {
                if (dataBuffer.Count > 0)
                {
                    EnsureCSVDirectoryExists();
                    File.AppendAllLines(csvFilePath, dataBuffer);
                    dataBuffer.Clear();
                }
            }
            catch (Exception ex)
            {
                Log($"Error saving CSV data: {ex.Message}");
                Log($"Attempted path: {csvFilePath}");
            }
        }

        private void EnsureCSVDirectoryExists()
        {
            string csvDirectory = Path.GetDirectoryName(csvFilePath);
            if (!Directory.Exists(csvDirectory))
            {
                Directory.CreateDirectory(csvDirectory);
                Log($"Created CSV directory: {csvDirectory}");
            }
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
            public bool Attacking { get; set; }
            public bool Jumping { get; set; }
            public bool Dashing { get; set; }
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
            public bool Attacking { get; set; }
            public bool Jumping { get; set; }
            public bool Dashing { get; set; }
        }

        private class ScaleDimensions
        {
            public int Width { get; set; }
            public int Height { get; set; }
        }
        #endregion
    }
}