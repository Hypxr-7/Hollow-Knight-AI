using System;
using System.Collections.Generic;
using System.IO;
using Modding;
using UnityEngine;

namespace DataCollector
{
    public class DataCollectorMod : Mod
    {
        // Configurable save directory
        public string saveDirectory = @"C:\Users\Abdullah\Downloads\HKData";

        private bool isRecording = false;
        private float recordingTimer = 0f;
        private float csvRecordingInterval = 1f / 3f; // 3 FPS
        private List<string> dataBuffer = new List<string>();
        private string csvFilePath;
        private string framesDirectoryPath;
        private int frameCount = 0;
        private string sessionTimestamp;


        // Performance settings
        private int targetWidth = 640;
        private int targetHeight = 360;

        public DataCollectorMod() : base("Data Collector") { }

        public override string GetVersion() => "v3.0";

        public override void Initialize()
        {
            ModHooks.HeroUpdateHook += OnHeroUpdate;

            try
            {
                // Create save directory if it doesn't exist
                if (!Directory.Exists(saveDirectory))
                {
                    Directory.CreateDirectory(saveDirectory);
                    Log($"Created main save directory: {saveDirectory}");
                }

                // Set up file paths with unique timestamps

                Log("Behavioral Data Collector initialized successfully!");

            }
            catch (Exception ex)
            {
                Log($"Error initializing mod: {ex.Message}");
            }
        }

        private void WriteCSVHeader()
        {
            string header = "frame_id,x_position,y_position,health," +
                          "moving_left,moving_right,moving_up,moving_down," +
                          "attacking,jumping,dashing,focusing,dreamnail";
            File.WriteAllText(csvFilePath, header + "\n");
        }

        public void OnHeroUpdate()
        {
            //List<string> pressedButtons = new List<string>();
            //for (int i = 0; i <= 19; i++)
            //{
            //    if (Input.GetKey((KeyCode)(350 + i))) // Joystick1Button0 = 350
            //    {
            //        pressedButtons.Add($"Button{i}");
            //    }
            //}
            //if (pressedButtons.Count > 0)
            //{
            //    Log($"Pressed: {string.Join(", ", pressedButtons)}");
            //}



            // Toggle recording with O key
            if (Input.GetKeyDown(KeyCode.O))
            {
                if (!isRecording)
                {
                    // Start new recording session
                    sessionTimestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                    csvFilePath = Path.Combine(saveDirectory, $"hk_actions_{sessionTimestamp}.csv");
                    framesDirectoryPath = Path.Combine(saveDirectory, $"frames_{sessionTimestamp}");
                    // Create frames directory
                    if (!Directory.Exists(framesDirectoryPath))
                    {
                        Directory.CreateDirectory(framesDirectoryPath);
                        Log($"Created frames directory: {framesDirectoryPath}");
                    }

                    // Write CSV header
                    WriteCSVHeader();


                    StartNewSession();
                }
                else
                {
                    // Stop current recording session
                    StopCurrentSession();
                }
            }

            // Record data when recording is active
            if (isRecording)
            {
                recordingTimer += Time.deltaTime;
                if (recordingTimer >= csvRecordingInterval)
                {
                    CapturePlayerData();
                    CaptureScreenFrame();
                    recordingTimer = 0f;
                }
            }
        }

        private void StartNewSession()
        {

            frameCount = 0; // Reset frame count for new session

            // Create unique paths for this session
            string currentTimestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            //csvFilePath = Path.Combine(saveDirectory, $"hk_actions_session_{currentTimestamp}.csv");
            //framesDirectoryPath = Path.Combine(saveDirectory, $"frames_session_{currentTimestamp}");

            // Create frames directory for this session
            if (!Directory.Exists(framesDirectoryPath))
            {
                Directory.CreateDirectory(framesDirectoryPath);
                Log($"Created frames directory: {framesDirectoryPath}");
            }

            // Write CSV header for this session
            WriteCSVHeader();

            isRecording = true;
            Log($"Recording session started!");
        }

        private void StopCurrentSession()
        {
            isRecording = false;

            // Save any remaining buffered data
            if (dataBuffer.Count > 0)
            {
                SaveBufferedData();
            }

            Log($"Recording session stopped!");
            Log($"CSV: {Path.GetFileName(csvFilePath)}");
            Log($"Frames: {Path.GetFileName(framesDirectoryPath)}");
        }

        private void CapturePlayerData()
        {
            try
            {
                // Get player controller
                var heroController = HeroController.instance;
                if (heroController == null) return;

                // Get player position
                Vector3 playerPos = heroController.transform.position;
                float xPos = playerPos.x;
                float yPos = playerPos.y;

                // Get player health
                int health = PlayerData.instance.health;

                // DEADZONE FIX: Apply deadzone to axis values to prevent drift
                // Comment out this block if you want to revert to original behavior
                float rawHorizontal = Input.GetAxis("Horizontal");
                float rawVertical = Input.GetAxis("Vertical");

                // Increase deadzone threshold - adjust this value if needed (0.5f is more aggressive)
                float deadzoneThreshold = 0.5f;

                // Apply deadzone - if within threshold, treat as zero
                float horizontalAxis = Mathf.Abs(rawHorizontal) > deadzoneThreshold ? rawHorizontal : 0f;
                float verticalAxis = Mathf.Abs(rawVertical) > deadzoneThreshold ? rawVertical : 0f;

                // Log the correction being applied
                if (frameCount % 30 == 0)
                {
                    Log($"Deadzone applied - Raw H:{rawHorizontal:F3} | Raw V:{rawVertical:F3}");
                }
                // END DEADZONE FIX

                // Controller input detection
                // Movement (old Input-based code)
                bool movingLeft  = horizontalAxis < -deadzoneThreshold || Input.GetKey(KeyCode.Joystick1Button14);
                bool movingRight = horizontalAxis > deadzoneThreshold || Input.GetKey(KeyCode.Joystick1Button15);
                //bool movingUp    = verticalAxis > deadzoneThreshold  || Input.GetKey(KeyCode.Joystick1Button12);
                //bool movingDown  = verticalAxis < -deadzoneThreshold || Input.GetKey(KeyCode.Joystick1Button13);

                // Movement (HeroController-based)
                //bool movingLeft = heroController.inputHandler.inputActions.moveVector.X < 0;
                //bool movingRight = heroController.inputHandler.inputActions.moveVector.X > 0;
                bool movingUp = heroController.cState.lookingUp;     // holding up
                bool movingDown = heroController.cState.lookingDown;   // holding down

                // Combat inputs (old Input-based code)
                //bool attacking = Input.GetKey(KeyCode.Joystick1Button0);
                //bool jumping   = Input.GetKey(KeyCode.Joystick1Button1);
                //bool dreamnail = Input.GetKey(KeyCode.Joystick1Button2);
                //bool focusing  = Input.GetKey(KeyCode.Joystick1Button3);
                //bool dashing   = Input.GetKey(KeyCode.Joystick1Button7);

                // Combat inputs (HeroController-based)
                bool attacking = heroController.cState.attacking;     // swinging nail
                bool jumping = heroController.cState.jumping;         // pressed jump
                bool dreamnail = false;     // dreamnail in use
                bool focusing = heroController.cState.focusing;       // healing
                bool dashing = heroController.cState.dashing;         // dash in progress

                // Create CSV row
                string dataRow = $"{frameCount},{xPos:F3},{yPos:F3},{health}," +
                                 $"{movingLeft},{movingRight},{movingUp},{movingDown}," +
                                 $"{attacking},{jumping},{dashing},{focusing},{dreamnail}";

                // Debug logging for input detection - can be commented out later
                //List<string> activeInputs = new List<string>();

                //activeInputs.Add($"H_Axis:{horizontalAxis:F3}");
                //activeInputs.Add($"V_Axis:{verticalAxis:F3}");

                // Old Input debug logging
                //if (Input.GetKey(KeyCode.Joystick1Button14)) activeInputs.Add("DPad_Left");
                //if (Input.GetKey(KeyCode.Joystick1Button15)) activeInputs.Add("DPad_Right");
                //if (Input.GetKey(KeyCode.Joystick1Button12)) activeInputs.Add("DPad_Up");
                //if (Input.GetKey(KeyCode.Joystick1Button13)) activeInputs.Add("DPad_Down");
                //if (Input.GetKey(KeyCode.Joystick1Button0)) activeInputs.Add("Attack");
                //if (Input.GetKey(KeyCode.Joystick1Button1)) activeInputs.Add("Jump");
                //if (Input.GetKey(KeyCode.Joystick1Button2)) activeInputs.Add("Dreamnail");
                //if (Input.GetKey(KeyCode.Joystick1Button3)) activeInputs.Add("Focus");
                //if (Input.GetKey(KeyCode.Joystick1Button7)) activeInputs.Add("Dash");

                // New HeroController-based logging
                //if (movingLeft) activeInputs.Add("RECORDING:Left");
                //if (movingRight) activeInputs.Add("RECORDING:Right");
                //if (movingUp) activeInputs.Add("RECORDING:Up");
                //if (movingDown) activeInputs.Add("RECORDING:Down");
                //if (attacking) activeInputs.Add("RECORDING:Attack");
                //if (jumping) activeInputs.Add("RECORDING:Jump");
                //if (dreamnail) activeInputs.Add("RECORDING:Dreamnail");
                //if (focusing) activeInputs.Add("RECORDING:Focus");
                //if (dashing) activeInputs.Add("RECORDING:Dash");

                //if (activeInputs.Count > 2) // More than just axis values
                //{
                //    Log($"Frame {frameCount}: {string.Join(" | ", activeInputs)}");
                //}
                // End debug logging

                // Add to buffer
                dataBuffer.Add(dataRow);
                frameCount++;

                // Save buffer periodically to avoid memory issues
                if (dataBuffer.Count >= 100)
                {
                    SaveBufferedData();
                }
            }
            catch (Exception ex)
            {
                Log($"Error capturing data: {ex.Message}");
            }
        }


        private void CaptureScreenFrame()
        {
            try
            {
                // Ensure frames directory exists
                if (!Directory.Exists(framesDirectoryPath))
                {
                    Directory.CreateDirectory(framesDirectoryPath);
                    Log($"Created frames directory: {framesDirectoryPath}");
                }

                // Capture the current screen
                Texture2D screenshot = CaptureScreen();
                if (screenshot != null)
                {
                    // Save directly to the session's frames directory
                    byte[] frameData = screenshot.EncodeToPNG();
                    string framePath = Path.Combine(framesDirectoryPath, $"frame_{frameCount:D6}.png");

                    // Ensure the directory still exists before writing file
                    string frameDir = Path.GetDirectoryName(framePath);
                    if (!Directory.Exists(frameDir))
                    {
                        Directory.CreateDirectory(frameDir);
                    }

                    File.WriteAllBytes(framePath, frameData);

                    // Clean up texture
                    UnityEngine.Object.Destroy(screenshot);
                }
            }
            catch (Exception ex)
            {
                Log($"Error capturing frame: {ex.Message}");
                Log($"Attempted path: {framesDirectoryPath}");

                // Try to recreate directory if it failed
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
        }

        private Texture2D CaptureScreen()
        {
            // Get current screen dimensions
            int screenWidth = Screen.width;
            int screenHeight = Screen.height;

            // Calculate scale to fit target resolution while maintaining aspect ratio
            float scale = Mathf.Min((float)targetWidth / screenWidth, (float)targetHeight / screenHeight);
            int scaledWidth = Mathf.RoundToInt(screenWidth * scale);
            int scaledHeight = Mathf.RoundToInt(screenHeight * scale);

            // Capture full screen
            Texture2D fullScreenshot = new Texture2D(screenWidth, screenHeight, TextureFormat.RGB24, false);
            fullScreenshot.ReadPixels(new Rect(0, 0, screenWidth, screenHeight), 0, 0);
            fullScreenshot.Apply();

            // Create scaled down version (keeping RGB for Python preprocessing)
            RenderTexture rt = RenderTexture.GetTemporary(scaledWidth, scaledHeight);
            Graphics.Blit(fullScreenshot, rt);

            Texture2D scaledScreenshot = new Texture2D(scaledWidth, scaledHeight, TextureFormat.RGB24, false);
            RenderTexture.active = rt;
            scaledScreenshot.ReadPixels(new Rect(0, 0, scaledWidth, scaledHeight), 0, 0);
            scaledScreenshot.Apply();
            RenderTexture.active = null;

            // Cleanup
            UnityEngine.Object.Destroy(fullScreenshot);
            RenderTexture.ReleaseTemporary(rt);

            return scaledScreenshot;
        }

        private void SaveBufferedData()
        {
            try
            {
                if (dataBuffer.Count > 0)
                {
                    // Ensure the CSV file's directory exists
                    string csvDir = Path.GetDirectoryName(csvFilePath);
                    if (!Directory.Exists(csvDir))
                    {
                        Directory.CreateDirectory(csvDir);
                        Log($"Created CSV directory: {csvDir}");
                    }

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


    }
}