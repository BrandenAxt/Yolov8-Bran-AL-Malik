from roboflow import Roboflow
rf = Roboflow(api_key="rJJ0npH37HgN1GHb7aT5")
project = rf.workspace("maliks-workspace-qdy2r").project("corrosion-detection-hrnxb-8ktdg")
dataset = project.version(1).download("yolov8")