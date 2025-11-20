# DEADFACE

Using the Google Mediapipe I created and app from where blendshapes can be tracked. 
Two modes are available. 
In Video, from uploaded video normalized Blendshape scores are written into a *.csv, a similar file than what Unreal Live Link is outputting. You can use this in Blender (great) or Unreal (experimental). 
In Stream, data is livestreamd via UDP, and can be streamed into Blender, Unreal, perhaps Unity. 
In Blender it work especially well with the FACE-It plugin.

i gave this a lot of free time, after 10 000 downloads of my previous script, i received 0 coffies, so if you want to encourage me to make my work public:<br />
<a href="https://www.buymeacoffee.com/qaanaaq" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 30px !important;width: 117px !important;" ></a>

Also some of you have reported the antivirus flagging - that is because for an indy-developer i would need to purchase certtificates so that the .executable gets certified. Also because of the previous paragraph, that is not viable for me.  BUT if you are interested i can put a tutorial together how to build it with pyinstaller from source code. (some coffee would help motivation)

<img src="https://i.imgur.com/Q23EHLr.png">


I am a visual artist, and coding is only a means to achieve results i need, lot of the code was aided by ChatGPT, some of the code is directly used from https://github.com/JimWest and his MeFaMo, and PyLiveLinkFace

# DOCUMENTATION

<h2 id="video-mode">📹 Video Mode</h2>

<!-- Image for Video Mode -->
<p>
  <img src="Video mode.png" alt="Video Mode Screenshot" style="max-width:100%; border:1px solid #444; border-radius:4px;">
</p>

<p>
  Video Mode allows you to load a prerecorded video file and extract facial animation data frame-by-frame using MediaPipe.
  Use this mode for offline analysis, testing, or generating clean neutral poses from hand-picked frames.
</p>

<h3>🟦 Load Video</h3>
<p>
  <strong>Load Video</strong><br>
  Opens a file picker to load an <code>.mp4</code>, <code>.avi</code>, <code>.mov</code>, or <code>.mkv</code> file into the app.
</p>

<h3>🟦 Start Tracking / Stop Tracking</h3>
<p>
  Begins processing the video and computing blendshapes. While running, the frame scrubber updates automatically.
  Press again to stop.
</p>

<h3>🟦 Timeline Scrubber</h3>
<p>
  Lets you manually jump to any frame in the video. The label beneath shows:<br>
  <code>currentTime / totalTime (frameNumber / totalFrames)</code>
</p>

<h3>🟦 Set Neutral (Video)</h3>
<p>
  Extracts a <em>neutral facial pose</em> from the <strong>current frame</strong> on the slider.
  Useful for customizing how expressions are interpreted compared to your personal resting face.
</p>

<h3>🟦 Zero</h3>
<p>
  Resets all neutral-pose values back to defaults (an empty/zero file).
</p>

<h3>🟦 Head / Eye Options</h3>
<p>
  <strong>Enable Head Tracking</strong><br>
  Enables estimation of head rotation and orientation from the video. Turn this off if you want only blendshape extraction.
</p>
<p>
  <strong>Symmetrical Eyes</strong><br>
  Forces both eyes to behave identically, useful if one eye is detected inconsistently.
</p>

<hr>

<h2 id="stream-mode">📡 Stream Mode</h2>

<!-- Image for Stream Mode -->
<p>
  <img src="Stream mode.png" alt="Stream Mode Screenshot" style="max-width:100%; border:1px solid #444; border-radius:4px;">
</p>

<p>
  Stream Mode uses your live camera and sends blendshape data over UDP to external applications
  (e.g. Unreal, Unity, Blender, custom tools).
</p>

<h3>🟧 Test Cam / Stop Cam</h3>
<p>
  Shows a raw camera preview to confirm the camera is working before streaming.
  Does not process blendshapes—just a visual test.
</p>

<h3>🟧 Start Streaming / Stop Streaming</h3>
<p>
  Starts sending real-time head and facial blendshape data over UDP.
  Uses the IP and port you enter in the fields below.
</p>

<h3>🟧 Set Neutral Pose</h3>
<p>
  Captures your <em>current live face</em> as the neutral reference for the streaming mode.
  Useful if your relaxed mouth/eyes differ from the default assumptions.
</p>

<h3>🟧 Zero (Stream)</h3>
<p>
  Clears the neutral pose file and resets all stored neutral calibration.
</p>

<hr>

<h2 id="advanced-controls">🎛️ Advanced Expression Controls</h2>
<p>
  These settings appear in <strong>Stream Mode</strong> and allow fine-tuning of all blendshape outputs.
</p>

<h3>Curve Response</h3>
<p>
  Adjusts global nonlinearity of facial response.
  Negative values soften or compress expressions; positive values exaggerate them.
</p>

<h3>Smoothing Filter</h3>
<p>
  Applies temporal smoothing to reduce jitter from noisy landmarks.
  A higher value means smoother motion but with slight latency.
</p>

<h3>Improved Shapes</h3>
<p>
  A toggle for an enhanced blendshape correction system (feature placeholder in the current build).
  Disabled in video mode.
</p>

<hr>

<h2 id="advanced-multipliers">🎚️ Advanced Expression Multipliers</h2>
<p>
  Inside the expandable panel (“Show Advanced”), you can individually adjust the strength of each blendshape.
</p>
<ul>
  <li>Sliders are grouped by <strong>Eyes</strong>, <strong>Brows</strong>, <strong>Mouth</strong>, and <strong>Other</strong>.</li>
  <li>Each control ranges from <strong>0 → 5×</strong> intensity.</li>
  <li>Special <strong>“-sym”</strong> sliders control symmetrical left/right shapes together.</li>
  <li>All adjustments automatically save to <code>multipliers.json</code>.</li>
  <li><strong>Settings persist across sessions</strong>, so your tuned facial behavior remains consistent over time.</li>
</ul>

<h3>Reset Multipliers</h3>
<p>
  Restores all multipliers back to <strong>1.0</strong> (default).
</p>

<h3>Hide Advanced / Show Advanced</h3>
<p>
  Collapses or expands the panel to hide or show the sliders.
</p>


