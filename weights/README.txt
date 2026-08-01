Model files are organized by backend, one subfolder each:

weights/yolo/     best.pt, best.onnx                          (bounding boxes + tracking)
weights/csrnet/   task_two_model_best_parta_90_120.pth.tar,    (native density map)
                  task_two_model_best_parta_120_150.pth.tar
weights/dmcount/  model_qnrf.pth (or similar, from the         (native density map)
                  DM-Count repo's pretrained_models/)
weights/p2pnet/   SHTechA.pth (or similar, from the             (point locations)
                  P2PNet repo's weights/)

When you run `python main.py`, you first pick a backend, then pick a
file from that backend's subfolder - so any number of checkpoints per
backend can sit here side by side (e.g. both CSRNet checkpoints), and
DM-Count/P2PNet's shared ".pth" extension is never ambiguous with each
other or with CSRNet's ".pth.tar".

Extra setup for DM-Count and P2PNet: unlike YOLO/CSRNet, these two are
NOT self-contained - they need a local clone of their own GitHub repo on
disk (their model architecture code is imported directly from that
clone, not reproduced in this project). See core/dmcount.py and
core/p2pnet.py for exactly why, and set the clone path either in
config/settings.yaml under `external_repos:`, or when prompted by the
CLI. P2PNet additionally needs an ImageNet-pretrained vgg16_bn backbone
available (downloaded automatically if online, or pre-placed - see the
docstring at the top of core/p2pnet.py for the offline setup command).

Using any backend other than YOLO also requires:
    pip install -r requirements-specialized.txt
