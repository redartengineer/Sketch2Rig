import hou
import csv
import os

#Path to the CSV file with predicted values
csv_path = os.path.expanduser("~/Documents/maya/projects/Houdini/ML/eyeballbrow_v01.0/ML folder/inferred_results_01.csv")

#Slider names (must match column order in CSV)
param_names = [
    "eyelid_top", "eyelid_top_rotate", "eyelid_bottom", "eyelid_bottom_rotate",
    "iris_size", "pupil_size", "eye_look_x", "eye_look_y",
    "eyebrow_center_y", "eyebrow_center_x", "eyebrow_scale", "eyebrow_rotate",
    "eyebrow1", "eyebrow2", "eyebrow30", "eyebrow4", "eyebrow5", "eyebrow6"
]

#Load values from the CSV (assumes header and one row of values)
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    row = next(reader)

#Apply values to Eye_UI rig
ctrl_node = hou.node("/obj/Eye_UI")
for name in param_names:
    parm = ctrl_node.parm(name)
    if parm:
        try:
            value = float(row[name])
            parm.set(value)
        except ValueError:
            print(f"⚠️ Could not convert value for {name}")
    else:
        print(f"⚠️ Parameter '{name}' not found in Eye_UI.")
