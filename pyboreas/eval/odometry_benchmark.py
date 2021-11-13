import argparse
import os
from pyboreas.utils.odometry import get_sequences, get_sequence_poses, get_sequence_poses_gt, compute_kitti_metrics

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='test/demo/pred/3d', type=str, help='path to prediction files')
    parser.add_argument('--gt', default='test/demo/gt', type=str, help='path to groundtruth files')
    parser.add_argument('--radar', dest='radar', action='store_true', help='evaluate radar odometry in SE(2)')
    parser.set_defaults(radar=False)
    parser.add_argument('--interp', default='', type=str, help='path to interpolation output, do not set if evaluating')
    parser.add_argument('--no-solver', dest='solver', action='store_false', help='disable solver for built-in interpolation')
    parser.set_defaults(solver=True)
    args = parser.parse_args()

    # evaluation mode
    dim = 2 if args.radar else 3
    if dim == 2:
        args.interp = ''  # force interpolation to be off for radar (2D) evaluation

    # parse sequences
    seq = get_sequences(args.pred, '.txt')
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(args.pred, seq)
    T_gt, times_gt, seq_lens_gt, crop = get_sequence_poses_gt(args.gt, seq, dim)

    # if we are interpolating...
    if args.interp:
        # can't be the same as pred
        if args.interp == args.pred:
            raise ValueError('`interp` directory path cannot be the same as the `pred` directory path')

        # make interp directory if it doesn't exist
        if not os.path.exists(args.interp):
            os.mkdir(args.interp)

    # compute errors
    t_err, r_err, _ = compute_kitti_metrics(T_gt, T_pred, times_gt, times_pred,
                                         seq_lens_gt, seq_lens_pred, seq, args.pred, dim, crop, args.interp, args.solver)

    # if we are evaluating, output print statements
    if not args.interp:
        # print out results
        print('Evaluated sequences: ', seq)
        print('Overall error: ', t_err, ' %, ', r_err, ' deg/m')