import argparse
import numpy as np
from pyboreas.utils.odometry import (
    compute_kitti_metrics,
    get_sequence_poses,
    get_sequence_poses_gt,
    get_sequences,
)
from pyboreas.utils.utils import (
    enforce_orthog,
    get_inverse_tf,
    rotation_error,
    translation_error,
    yawPitchRollToRot,
)


def eval_odom(pred, gt, radar=True):
    # evaluation mode
    dim = 2 if radar else 3

    # parse sequences
    seq = get_sequences(pred, ".txt")
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(pred, seq)
    # get corresponding groundtruth poses
    T_gt, _, seq_lens_gt, crop = get_sequence_poses_gt(gt, seq, dim)

    RotX180 = yawPitchRollToRot(0, 0, np.pi) # roll 180 deg
    '''
    for i in range(len(T_gt)):
        T_gt_inv = np.linalg.inv(T_gt[i])    # transform world->robot
        T_rot = T_gt_inv[:3,:3] @ RotX180    # first transform world->robot, then rotate upside down
        T_gt_inv[:3,:3] = T_rot              # update rotation matrix
        T_gt[i] = np.linalg.inv(T_gt_inv)    # transform robot->world
    '''
    '''
    for i in range(len(T_pred)):
        T_pred_inv = np.linalg.inv(T_pred[i])           # transform world->robot
        T_rot = T_pred_inv[:3, :3] @ RotX180            # first transform world->robot, then rotate upside down
        T_pred_inv[:3, :3] = T_rot                       # update rotation matrix
        T_pred[i] = np.linalg.inv(T_pred_inv)            # transform robot->world
    '''

    # compute errors
    t_err, r_err, _ = compute_kitti_metrics(
        T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, pred, dim, crop
    )
    '''
    import matplotlib.pyplot as plt
    print("------------- gt -------------")
    print(T_gt[0])
    print(T_gt[100])
    print(T_gt[2000])
    print("------------- pred -------------")
    print(T_pred[0])
    print(T_pred[100])
    print(T_pred[2000])
    
    # Extract X and Y coordinates from T_pred and T_gt
    x_pred = [T_pred[0][3] for T_pred in T_pred]
    y_pred = [T_pred[1][3] for T_pred in T_pred]
    #print("pred:" + str(x_pred) + "\n")

    x_gt = [T_gt[0][3] for T_gt in T_gt]
    y_gt = [T_gt[1][3] for T_gt in T_gt]
    #print("gt:" + str(x_gt) + "\n")

    # Plot the trajectories
    plt.figure(figsize=(8, 6))
    plt.plot(x_pred, y_pred, label='Predicted Trajectory', color='b', marker='o')
    plt.plot(x_gt, y_gt, label='Ground Truth Trajectory', color='g', marker='s')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Trajectory Plot')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

    # print out results
    print("Evaluated sequences: ", seq)
    print("Overall error: ", t_err, " %, ", r_err, " deg/m")
    return t_err, r_err


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred", default="test/demo/pred/3d", type=str, help="path to prediction files"
    )
    parser.add_argument(
        "--gt", default="test/demo/gt", type=str, help="path to groundtruth files"
    )
    parser.add_argument(
        "--radar",
        dest="radar",
        action="store_true",
        help="evaluate radar odometry in SE(2)",
    )
    parser.set_defaults(radar=False)
    args = parser.parse_args()

    eval_odom(args.pred, args.gt, args.radar)
