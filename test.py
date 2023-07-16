import json
import numpy as np
trajectory = np.array([[1,2,3],[4,5,6]]).tolist()
data = {
"LoopMode": "Wrap",
"FrameDuration": 0.02,
"MotionWeight": 1.0,

"Frames":
trajectory
}

with open('data.txt', 'w') as f:
    json.dump(data, f)