python3 alphapose_/main.py \
    --cfg=submodules/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint=submodules/AlphaPose/pretrained_models/fast_res50_256x192.pth \
    --indir=openpose/data/mot17/dev1/100/color \
    --outdir=openpose/data/mot17/dev1/100/color_alphapose_fastpose_poseflow \
    --save_img \
    --pose_track \
    --pose_track_model=submodules/AlphaPose/trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth \
    --profile 2>&1 | tee alphapose.txt

    # --sp \
