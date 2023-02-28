
import os
import natsort


def loop():
    n = 0
    while True:
        yield n
        n += 1


def check_input(args):
    # for wecam
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', int(args.webcam)

    # for video
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError(
                'Error: --video must refer to a video file, not directory.')

    # for detection results
    if len(args.detfile):
        if os.path.isfile(args.detfile):
            detfile = args.detfile
            return 'detfile', detfile
        else:
            raise IOError(
                'Error: --detfile must refer to a detection json file, not directory.')

    # for images
    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                im_names = files
            im_names = natsort.natsorted(im_names)
        elif len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]

        return 'image', im_names

    else:
        raise NotImplementedError


def print_finish_info(args):
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
