import numpy as np
from pyboreas import BoreasDataset
import cv2
import os.path as osp
import sys # to access the system
import rosbag
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge  # You'l
import rospy
from pyboreas.utils.utils import *

from std_msgs.msg import Int32, String
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from  geometry_msgs.msg import TransformStamped

class BagWriter:
  def __init__(self, path):
    self.path = path
    print("Writing to bag at path: \n"+path)
    self.bag = rosbag.Bag(self.path, 'w')

  def WriteImage(self, image, stamp, topic):
    image.header.stamp = stamp
    image.header.frame_id = "navtech"
    self.bag.write(topic, image, stamp)
  def WriteTf(self, T_tf, stamp, topic):
      tvek = TFMessage()
      tvek.transforms.append(T_tf)
      self.bag.write("/tf", tvek, stamp)
      odom = Odometry()
      odom.header.stamp = T_tf.header.stamp
      odom.header.frame_id = "world"
      odom.child_frame_id = "navtech"
      odom.pose.pose.position = T_tf.transform.translation;
      odom.pose.pose.orientation = T_tf.transform.rotation;
      self.bag.write(topic, odom, stamp)
  def Close(self):
      self.bag.close()

def ProcessBag(root):

    bd = BoreasDataset(root)
    
    rospy.init_node('radar_publisher', anonymous=True)
    pub_polar = rospy.Publisher('/Navtech/Polar', Image, queue_size=10)
    br = CvBridge()
    print("bd.sequences")
    print(bd.sequences)
    stamp=0

    for seq in bd.sequences:
        print("seq: ")
        print(seq)
        output_bag_path = osp.join(seq.seq_root, seq.ID + ".bag")
        bw = BagWriter(output_bag_path)
        
        N = len(seq.radar_frames)
        print("radar scans: " + str(N))
        #exit()
        for i in range(N):
            print("frame: " + str(i) + " / " + str(N))
            radar_frame = seq.get_radar(i)
            fft_data = 255*radar_frame.polar
            radar_polar = fft_data.astype(np.uint8)
            #height = fft_data.shape[0]
            #width = fft_data.shape[1]
            #channels = fft_data_vis.shape[2]
            #print("resolution: " + str(radar_frame.resolution))
            cv2.imshow("Polar radar", radar_polar)
            cv2.waitKey(10)
            seconds = radar_frame.timestamp_micro / 1e6
            stamp = rospy.Time.from_sec(seconds)
            image_msg = Image()
            image_msg = br.cv2_to_imgmsg(radar_polar)
            bw.WriteImage(image_msg, stamp, "/Navtech/Polar")
            
            

            pose = radar_frame.pose
            q = rotToQuaternion(pose[0:3,0:3])
            #print(pose[0:3,0:3])
            #print(q)
            #print(pose)

            t = TransformStamped()
            t.header.frame_id = "world"
            t.header.stamp = stamp
            t.child_frame_id = "navtech"
            t.transform.translation.x = pose[0,3]
            t.transform.translation.y = pose[1,3]
            t.transform.translation.z = pose[2,3]
            t.transform.rotation.w = q[3]
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            #print(t.transform.rotation)
            bw.WriteTf(t, stamp, "/gt")

            #pub_polar.publish(image_msg)
            # do something
            radar_frame.unload_data() # Memory reqs will keep increasing without this
        t = TransformStamped()
        bw.WriteTf(t, stamp, "/end")
        bw.Close()


def main() -> int:
    """Echo the input arguments to standard output"""
    root = '/mnt/external_ssd/radar_data/boreas/'
    ProcessBag(root)
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
