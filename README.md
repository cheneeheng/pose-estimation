# pose-estimation
Collection of different pose estimation models.
- [openpose](openpose): Inference model from CMU.
- [movenet](movenet): Model from google that runs on tf. Focus here is on speed.

## Update
**[2023.03.10]** Modified caffe in openpose to use local one. Supports cudnnv8 via cudnn_frontend api.  
**[2023.02.24]** Strongsort and OCSort are working with openpose.  
**[2023.02.23]** Deepsort and ByteTrack are working with openpose.  
**[2023.02.20]** Adding tracking to the repository. Will be moved to another repository once the tracking experiments are finished.

## TODO
Adding pose estimators:
- Mediapipe (blazepose)
- HrNet-DEKR
- AlphaPose

Adding trackers:
- CenterTrack
- FairMOT
- MOTR
- TransTrack
- qdtrack
- OMC + CSTrack
- MOTDT (Towards-Realtime-MOT)
- Pose Flow

